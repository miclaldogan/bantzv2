"""
Bantz — Background stderr observer (#124).

Monitors terminal error streams (via /proc/<pid>/fd/2, PTY tap, or
PROMPT_COMMAND hook) and proactively classifies errors:

    ignore   → log silently
    info     → add to memory
    warning  → toast notification
    critical → full analysis popup via lightweight LLM

Architecture:
    StderrReader  → raw lines from stderr source
    ErrorBuffer   → batches lines, deduplicates within window
    ErrorClassifier → regex pre-filter + optional LLM analysis
    Observer      → orchestrator daemon (thread-based)
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

import aiohttp

log = logging.getLogger(__name__)

# ── Severity ─────────────────────────────────────────────────────────────

class Severity(str, Enum):
    IGNORE = "ignore"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ── Error patterns ───────────────────────────────────────────────────────

# Each tuple: (compiled regex, severity)
_PATTERNS: list[tuple[re.Pattern, Severity]] = [
    # ── Critical ──
    (re.compile(r"Traceback \(most recent call last\)", re.I), Severity.CRITICAL),
    (re.compile(r"(?:SIGSEGV|SIGKILL|SIGABRT|Segmentation fault)", re.I), Severity.CRITICAL),
    (re.compile(r"Out[- ]?[Oo]f[- ]?[Mm]emory|OOM|Cannot allocate memory", re.I), Severity.CRITICAL),
    (re.compile(r"(?:^|\s)FATAL(?:\s|:)", re.I), Severity.CRITICAL),
    (re.compile(r"panic:", re.I), Severity.CRITICAL),
    (re.compile(r"core dumped", re.I), Severity.CRITICAL),
    # ── Warning ──
    (re.compile(r"(?:Error|Exception):\s*.+", re.I), Severity.WARNING),
    (re.compile(r"npm ERR!", re.I), Severity.WARNING),
    (re.compile(r"(?:^|\s)error(?:\[\w+\])?:", re.I), Severity.WARNING),
    (re.compile(r"undefined reference", re.I), Severity.WARNING),
    (re.compile(r"Error response from daemon", re.I), Severity.WARNING),
    (re.compile(r"Permission denied", re.I), Severity.WARNING),
    (re.compile(r"FAILED\s+tests?/", re.I), Severity.WARNING),
    (re.compile(r"Build FAILED|BUILD FAILURE", re.I), Severity.WARNING),
    (re.compile(r"command not found", re.I), Severity.WARNING),
    # ── Info ──
    (re.compile(r"(?:^|\s)warning(?:\[\w+\])?:", re.I), Severity.INFO),
    (re.compile(r"DeprecationWarning|FutureWarning|PendingDeprecation", re.I), Severity.INFO),
    (re.compile(r"warn\[", re.I), Severity.INFO),
]


# ── Data classes ─────────────────────────────────────────────────────────

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
        """Hash for dedup: same pattern + first meaningful line."""
        key = self.pattern_matched or self.raw_text[:120]
        return hashlib.md5(key.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "raw_text": self.raw_text,
            "pattern_matched": self.pattern_matched,
            "analysis": self.analysis,
            "timestamp": self.timestamp,
        }


# ── ErrorClassifier ──────────────────────────────────────────────────────

class ErrorClassifier:
    """Regex pre-filter with optional LLM escalation for critical errors."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        analysis_model: str = "qwen2.5:0.5b",
        enable_llm: bool = True,
    ):
        self._ollama_url = ollama_base_url
        self._model = analysis_model
        self._enable_llm = enable_llm

    def classify(self, text: str) -> Optional[ErrorEvent]:
        """Classify a block of stderr text. Returns None for unrecognised."""
        if not text or not text.strip():
            return None

        best_severity = None
        best_pattern = ""

        for pattern, severity in _PATTERNS:
            if pattern.search(text):
                if best_severity is None or severity.value > best_severity.value:
                    best_severity = severity
                    best_pattern = pattern.pattern
                # Critical is the highest — no need to keep scanning
                if severity == Severity.CRITICAL:
                    break

        if best_severity is None:
            return None

        return ErrorEvent(
            severity=best_severity,
            raw_text=text.strip(),
            pattern_matched=best_pattern,
        )

    async def analyze(self, event: ErrorEvent) -> str:
        """Ask a lightweight LLM to explain a critical error. Returns analysis text."""
        if not self._enable_llm:
            return ""
        prompt = (
            "You are a systems expert. Analyze this terminal error briefly "
            "(2-3 sentences max). Suggest a fix if obvious.\n\n"
            f"```\n{event.raw_text[:1500]}\n```"
        )
        try:
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


# ── ErrorBuffer ──────────────────────────────────────────────────────────

class ErrorBuffer:
    """Batches stderr lines and deduplicates within a time window."""

    def __init__(self, batch_seconds: float = 5.0, dedup_window: float = 60.0):
        self._batch_sec = batch_seconds
        self._dedup_window = dedup_window
        self._lines: list[str] = []
        self._last_flush: float = time.time()
        self._seen: dict[str, float] = {}  # fingerprint → last seen time
        self._lock = threading.Lock()

    def add_line(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def should_flush(self) -> bool:
        return (time.time() - self._last_flush) >= self._batch_sec and bool(self._lines)

    def flush(self) -> Optional[str]:
        """Return batched text and reset, or None if empty."""
        with self._lock:
            if not self._lines:
                return None
            text = "\n".join(self._lines)
            self._lines.clear()
            self._last_flush = time.time()
            return text

    def is_duplicate(self, fingerprint: str) -> bool:
        """Check if this fingerprint was seen recently."""
        now = time.time()
        # Prune old entries
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


# ── StderrReader ─────────────────────────────────────────────────────────

class StderrReader:
    """
    Reads stderr lines from a source.

    Strategy priority:
    1. Direct file descriptor (e.g. /proc/<pid>/fd/2)
    2. Named pipe / FIFO
    3. Injected lines via push() for testing / PROMPT_COMMAND hook
    """

    def __init__(self):
        self._queue: deque[str] = deque(maxlen=5000)
        self._lock = threading.Lock()

    def push(self, line: str) -> None:
        """Manually inject a stderr line (from hook or pipe)."""
        with self._lock:
            self._queue.append(line)

    def push_lines(self, text: str) -> None:
        """Inject multiple lines at once."""
        for line in text.splitlines():
            if line.strip():
                self.push(line)

    def pop_all(self) -> list[str]:
        """Drain all pending lines."""
        with self._lock:
            lines = list(self._queue)
            self._queue.clear()
            return lines

    @property
    def pending(self) -> int:
        return len(self._queue)


# ── Observer (main daemon) ──────────────────────────────────────────────

class Observer:
    """
    Background stderr observer daemon.

    Runs in a dedicated thread. Polls StderrReader → ErrorBuffer → Classifier.
    Notifications are delivered via a callback:
        on_error(event: ErrorEvent)
    """

    def __init__(
        self,
        on_error: Optional[Callable[[ErrorEvent], Any]] = None,
        severity_threshold: str = "warning",
        batch_seconds: float = 5.0,
        dedup_window: float = 60.0,
        ollama_base_url: str = "http://localhost:11434",
        analysis_model: str = "qwen2.5:0.5b",
        enable_llm_analysis: bool = True,
    ):
        self.on_error = on_error
        self.threshold = Severity(severity_threshold)
        self.reader = StderrReader()
        self.buffer = ErrorBuffer(batch_seconds, dedup_window)
        self.classifier = ErrorClassifier(
            ollama_base_url=ollama_base_url,
            analysis_model=analysis_model,
            enable_llm=enable_llm_analysis,
        )
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self._stats = {
            "total_lines": 0,
            "total_events": 0,
            "by_severity": {s.value: 0 for s in Severity},
            "deduplicated": 0,
        }
        self._stats_lock = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the observer daemon thread."""
        if self._thread and self._thread.is_alive():
            log.warning("Observer already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="bantz-observer", daemon=True
        )
        self._thread.start()
        log.info("Observer started (threshold=%s)", self.threshold.value)

    def stop(self) -> None:
        """Signal the observer to stop and wait for thread to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None
        log.info("Observer stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Main loop ─────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main daemon loop — runs in dedicated thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            while not self._stop_event.is_set():
                # Drain reader → buffer
                lines = self.reader.pop_all()
                for line in lines:
                    self.buffer.add_line(line)
                    with self._stats_lock:
                        self._stats["total_lines"] += 1

                # Flush buffer when ready
                if self.buffer.should_flush():
                    text = self.buffer.flush()
                    if text:
                        self._process_batch(text)

                # Sleep briefly to avoid busy-wait (but stay responsive)
                self._stop_event.wait(timeout=0.5)

            # Final flush
            text = self.buffer.flush()
            if text:
                self._process_batch(text)

        finally:
            self._loop.close()
            self._loop = None

    def _process_batch(self, text: str) -> None:
        """Classify a batch of stderr text and notify if above threshold."""
        event = self.classifier.classify(text)
        if event is None:
            return

        # Severity gate
        _severity_order = {
            Severity.IGNORE: 0, Severity.INFO: 1,
            Severity.WARNING: 2, Severity.CRITICAL: 3,
        }
        if _severity_order.get(event.severity, 0) < _severity_order.get(self.threshold, 0):
            return

        # Dedup gate
        if self.buffer.is_duplicate(event.fingerprint):
            with self._stats_lock:
                self._stats["deduplicated"] += 1
            return

        # LLM analysis for critical errors
        if event.severity == Severity.CRITICAL and self._loop:
            try:
                event.analysis = self._loop.run_until_complete(
                    self.classifier.analyze(event)
                )
            except Exception as exc:
                log.debug("LLM analysis failed: %s", exc)

        # Record stats
        with self._stats_lock:
            self._stats["total_events"] += 1
            self._stats["by_severity"][event.severity.value] += 1

        # Deliver notification
        if self.on_error:
            try:
                self.on_error(event)
            except Exception as exc:
                log.debug("Observer notification callback failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────

    def feed(self, text: str) -> None:
        """Manually feed stderr text into the observer (for hooks / pipes)."""
        self.reader.push_lines(text)

    def stats(self) -> dict[str, Any]:
        """Return observer statistics."""
        with self._stats_lock:
            return {
                **self._stats,
                "running": self.running,
                "buffer_pending": self.buffer.pending_count,
            }


# ── Module singleton ─────────────────────────────────────────────────────

observer = Observer()
