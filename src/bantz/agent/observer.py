"""Event-driven stderr observer (#124, #220 Sprint 3 Part 5).

Classifies terminal error streams and delivers notifications.  No polling
loop — feeds arrive via ``feed()`` or via EventBus ``stderr_line`` events.
The LLM analysis call uses ``aiohttp`` which is imported **lazily** so the
module never hard-fails when the package is absent.

Architecture::

    StderrReader  → raw lines from stderr source
    ErrorBuffer   → batches lines, deduplicates within window
    ErrorClassifier → regex pre-filter + optional LLM analysis
    Observer      → orchestrator (event-driven, no polling)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from bantz.core.event_bus import bus

log = logging.getLogger(__name__)

# ── Severity ──────────────────────────────────────────────────────────────

class Severity(str, Enum):
    IGNORE = "ignore"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

# ── Error patterns ────────────────────────────────────────────────────────

_PATTERNS: list[tuple[re.Pattern, Severity]] = [
    # Critical
    (re.compile(r"Traceback \(most recent call last\)", re.I), Severity.CRITICAL),
    (re.compile(r"(?:SIGSEGV|SIGKILL|SIGABRT|Segmentation fault)", re.I), Severity.CRITICAL),
    (re.compile(r"Out[- ]?[Oo]f[- ]?[Mm]emory|OOM|Cannot allocate memory", re.I), Severity.CRITICAL),
    (re.compile(r"(?:^|\s)FATAL(?:\s|:)", re.I), Severity.CRITICAL),
    (re.compile(r"panic:", re.I), Severity.CRITICAL),
    (re.compile(r"core dumped", re.I), Severity.CRITICAL),
    # Warning
    (re.compile(r"(?:Error|Exception):\s*.+", re.I), Severity.WARNING),
    (re.compile(r"npm ERR!", re.I), Severity.WARNING),
    (re.compile(r"(?:^|\s)error(?:\[\w+\])?:", re.I), Severity.WARNING),
    (re.compile(r"undefined reference", re.I), Severity.WARNING),
    (re.compile(r"Error response from daemon", re.I), Severity.WARNING),
    (re.compile(r"Permission denied", re.I), Severity.WARNING),
    (re.compile(r"FAILED\s+tests?/", re.I), Severity.WARNING),
    (re.compile(r"Build FAILED|BUILD FAILURE", re.I), Severity.WARNING),
    (re.compile(r"command not found", re.I), Severity.WARNING),
    # Info
    (re.compile(r"(?:^|\s)warning(?:\[\w+\])?:", re.I), Severity.INFO),
    (re.compile(r"DeprecationWarning|FutureWarning|PendingDeprecation", re.I), Severity.INFO),
    (re.compile(r"warn\[", re.I), Severity.INFO),
]

# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class ErrorEvent:
    """A classified error event ready for notification."""
    severity: Severity
    raw_text: str
    pattern_matched: str = ""
    analysis: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        key = self.pattern_matched or self.raw_text[:120]
        return hashlib.md5(key.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity.value, "raw_text": self.raw_text,
                "pattern_matched": self.pattern_matched, "analysis": self.analysis,
                "timestamp": self.timestamp}

# ── ErrorClassifier ───────────────────────────────────────────────────────

class ErrorClassifier:
    """Regex pre-filter with optional LLM escalation for critical errors."""

    def __init__(self, ollama_base_url: str = "http://localhost:11434",
                 analysis_model: str = "qwen2.5:0.5b", enable_llm: bool = True):
        self._ollama_url = ollama_base_url
        self._model = analysis_model
        self._enable_llm = enable_llm

    def classify(self, text: str) -> Optional[ErrorEvent]:
        if not text or not text.strip():
            return None
        best_severity = None
        best_pattern = ""
        for pattern, severity in _PATTERNS:
            if pattern.search(text):
                if best_severity is None or severity.value > best_severity.value:
                    best_severity = severity
                    best_pattern = pattern.pattern
                if severity == Severity.CRITICAL:
                    break
        if best_severity is None:
            return None
        return ErrorEvent(severity=best_severity, raw_text=text.strip(),
                          pattern_matched=best_pattern)

    async def analyze(self, event: ErrorEvent) -> str:
        """Ask a lightweight LLM to explain a critical error."""
        if not self._enable_llm:
            return ""
        prompt = (
            "You are a systems expert. Analyze this terminal error briefly "
            "(2-3 sentences max). Suggest a fix if obvious.\n\n"
            f"```\n{event.raw_text[:1500]}\n```"
        )
        try:
            import aiohttp  # lazy import — not always installed
            async with aiohttp.ClientSession() as sess:
                async with sess.post(
                    f"{self._ollama_url}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("response", "").strip()
        except Exception as exc:
            log.debug("Observer LLM analysis failed: %s", exc)
        return ""

# ── ErrorBuffer ───────────────────────────────────────────────────────────

class ErrorBuffer:
    """Batches stderr lines and deduplicates within a time window."""

    def __init__(self, batch_seconds: float = 5.0, dedup_window: float = 60.0):
        self._batch_sec = batch_seconds
        self._dedup_window = dedup_window
        self._lines: list[str] = []
        self._last_flush: float = time.time()
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def add_line(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def should_flush(self) -> bool:
        return (time.time() - self._last_flush) >= self._batch_sec and bool(self._lines)

    def flush(self) -> Optional[str]:
        with self._lock:
            if not self._lines:
                return None
            text = "\n".join(self._lines)
            self._lines.clear()
            self._last_flush = time.time()
            return text

    def is_duplicate(self, fingerprint: str) -> bool:
        now = time.time()
        expired = [k for k, t in self._seen.items() if now - t > self._dedup_window]
        for k in expired:
            del self._seen[k]
        if fingerprint in self._seen:
            return True
        self._seen[fingerprint] = now
        return False

    @property
    def pending_count(self) -> int:
        return len(self._lines)

# ── StderrReader ──────────────────────────────────────────────────────────

class StderrReader:
    """Reads stderr lines from a source (push-based)."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque(maxlen=5000)
        self._lock = threading.Lock()

    def push(self, line: str) -> None:
        with self._lock:
            self._queue.append(line)

    def push_lines(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self.push(line)

    def pop_all(self) -> list[str]:
        with self._lock:
            lines = list(self._queue)
            self._queue.clear()
            return lines

    @property
    def pending(self) -> int:
        return len(self._queue)

# ── Observer (event-driven, no polling) ───────────────────────────────────

_SEVERITY_ORDER = {Severity.IGNORE: 0, Severity.INFO: 1,
                   Severity.WARNING: 2, Severity.CRITICAL: 3}

class Observer:
    """Event-driven stderr observer (#220 Sprint 3 Part 5).

    No polling loop.  Errors arrive via ``feed()`` or EventBus
    ``stderr_line`` events and are processed immediately.
    """

    def __init__(self, on_error: Optional[Callable[[ErrorEvent], Any]] = None,
                 severity_threshold: str = "warning",
                 batch_seconds: float = 5.0, dedup_window: float = 60.0,
                 ollama_base_url: str = "http://localhost:11434",
                 analysis_model: str = "qwen2.5:0.5b",
                 enable_llm_analysis: bool = True):
        self.on_error = on_error
        self.threshold = Severity(severity_threshold)
        self.reader = StderrReader()
        self.buffer = ErrorBuffer(batch_seconds, dedup_window)
        self.classifier = ErrorClassifier(ollama_base_url=ollama_base_url,
                                          analysis_model=analysis_model,
                                          enable_llm=enable_llm_analysis)
        self._stop_event = threading.Event()
        self._running = False
        self._stats = {"total_lines": 0, "total_events": 0,
                       "by_severity": {s.value: 0 for s in Severity},
                       "deduplicated": 0}
        self._stats_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Subscribe to EventBus and mark as running.  No thread needed."""
        if self._running:
            log.warning("Observer already running")
            return
        self._stop_event.clear()
        self._running = True
        bus.on("stderr_line", self._on_stderr_event)
        log.info("Observer started (threshold=%s, event-driven)", self.threshold.value)
        # Process any lines that were fed before start()
        self._drain_reader()

    def stop(self) -> None:
        """Unsubscribe from EventBus and flush remaining data."""
        self._stop_event.set()
        bus.off("stderr_line", self._on_stderr_event)
        # Final flush
        self._drain_reader()
        text = self.buffer.flush()
        if text:
            self._process_batch(text)
        self._running = False
        log.info("Observer stopped")

    @property
    def running(self) -> bool:
        return self._running

    # ── EventBus handler ──────────────────────────────────────────────

    def _on_stderr_event(self, event: Any) -> None:
        """Called by EventBus when a ``stderr_line`` event arrives."""
        line = event.data.get("line", "") if hasattr(event, "data") else ""
        if line:
            self.reader.push(line)
            self._drain_and_process()

    # ── Core processing ───────────────────────────────────────────────

    def feed(self, text: str) -> None:
        """Manually feed stderr text (for hooks / pipes / direct use)."""
        self.reader.push_lines(text)
        # Counting happens in _drain_reader; just trigger processing
        self._drain_and_process()

    def _drain_reader(self) -> None:
        """Move all pending reader lines into the buffer."""
        lines = self.reader.pop_all()
        for line in lines:
            self.buffer.add_line(line)
            with self._stats_lock:
                self._stats["total_lines"] += 1

    def _drain_and_process(self) -> None:
        """Drain reader → buffer, then flush and process if ready."""
        self._drain_reader()
        if self.buffer.should_flush():
            text = self.buffer.flush()
            if text:
                self._process_batch(text)

    def _process_batch(self, text: str) -> None:
        """Classify a batch and notify if above threshold."""
        event = self.classifier.classify(text)
        if event is None:
            return
        # Severity gate
        if _SEVERITY_ORDER.get(event.severity, 0) < _SEVERITY_ORDER.get(self.threshold, 0):
            return
        # Dedup gate
        if self.buffer.is_duplicate(event.fingerprint):
            with self._stats_lock:
                self._stats["deduplicated"] += 1
            return
        # LLM analysis for critical errors (async, best-effort)
        if event.severity == Severity.CRITICAL:
            try:
                loop = asyncio.new_event_loop()
                try:
                    event.analysis = loop.run_until_complete(
                        self.classifier.analyze(event))
                finally:
                    loop.close()
            except Exception as exc:
                log.debug("LLM analysis failed: %s", exc)
        # Record stats
        with self._stats_lock:
            self._stats["total_events"] += 1
            self._stats["by_severity"][event.severity.value] += 1
        # Emit on EventBus
        bus.emit_threadsafe("observer_error", **event.to_dict())
        # Deliver notification callback
        if self.on_error:
            try:
                self.on_error(event)
            except Exception as exc:
                log.debug("Observer callback failed: %s", exc)

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {**self._stats, "running": self.running,
                    "buffer_pending": self.buffer.pending_count}

# ── Module singleton ──────────────────────────────────────────────────────

observer = Observer()
