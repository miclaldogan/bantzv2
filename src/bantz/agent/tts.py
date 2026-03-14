"""
Bantz — Streaming TTS Pipeline (#131)

Sentence-by-sentence text-to-speech using Piper + aplay.
Designed for audio morning briefings with interrupt support.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                 TTSEngine                           │
    │                                                     │
    │  text → sentence_split → Queue<str>                 │
    │                             ↓                       │
    │         ┌── synthesize(sentence) ──→ wav bytes ──┐  │
    │         │   (Piper subprocess)                   │  │
    │         └────────────────────────────────────────┘  │
    │                             ↓                       │
    │         ┌── play(wav_bytes) ──→ aplay subprocess ┐  │
    │         │   (cancellable via SIGTERM)             │  │
    │         └────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘

Key design:
  - Producer-consumer with asyncio.Queue
  - Synthesize sentence N+1 while playing sentence N
  - stop() sends SIGTERM to active aplay → immediate silence
  - Graceful fallback: if Piper/aplay missing, logs warning and returns
  - English-first: en_US-lessac-medium Piper model
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Sentence splitter
# ═══════════════════════════════════════════════════════════════════════════

# Split on sentence-ending punctuation followed by whitespace or end.
# Keeps emoji headers (📅, 🌤️ etc.) attached to following text.
_SENTENCE_RE = re.compile(
    r'(?<=[.!?;:])\s+|(?<=\n)\s*'
)


def split_sentences(text: str) -> list[str]:
    """Split briefing text into speakable chunks.

    Strategy: split on sentence boundaries (. ! ? ; :) and newlines.
    Merge very short fragments (< 10 chars) with their predecessor.
    Strip emoji and markdown formatting for cleaner speech.
    """
    if not text:
        return []

    # Normalize whitespace
    text = text.strip()

    # Split on sentence endings + newlines
    raw = _SENTENCE_RE.split(text)
    raw = [s.strip() for s in raw if s.strip()]

    if not raw:
        return []

    # Merge very short fragments with previous
    merged: list[str] = [raw[0]]
    for chunk in raw[1:]:
        if len(merged[-1]) < 10:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    # Clean for speech: strip emoji, markdown bold/italic
    cleaned = []
    for s in merged:
        # Remove emoji (rough range)
        s = re.sub(
            r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0'
            r'\U0000FE00-\U0000FE0F\U0000200D]+', '', s
        )
        # Remove markdown bold/italic
        s = re.sub(r'\*+', '', s)
        s = re.sub(r'_+', '', s)
        # Remove bullet markers
        s = re.sub(r'^[\s•\-–—]+', '', s)
        s = s.strip()
        if s:
            cleaned.append(s)

    return cleaned


# ═══════════════════════════════════════════════════════════════════════════
# Markdown sanitizer for TTS (#247)
# ═══════════════════════════════════════════════════════════════════════════

# Compiled regexes — strict ordering is critical (see issue #247 commentary).
# 1. Code blocks FIRST (before we strip backticks individually)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.DOTALL)
# 2. Inline code backticks
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
# 3. Markdown links [text](url) → keep "text"
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# 4. <thinking>…</thinking> blocks (import-free, self-contained)
_THINKING_RE = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)
# 5. Bare URLs
_URL_RE = re.compile(r"https?://[^\s\"'>]+")
# 6. Markdown heading hashes, bold/italic markers, blockquote
_MD_SYMBOLS_RE = re.compile(r"[#*_>~`]+")


def strip_markdown_for_tts(text: str) -> str:
    """Remove code, URLs, and Markdown syntax so TTS reads clean prose.

    Processing order matters — code blocks must be removed **before**
    individual backtick stripping, otherwise partial code leaks into
    the speech output and Piper reads it character-by-character.

    Steps (in strict order):
      1. Remove fenced code blocks (```…```) and all their content.
      2. Remove inline code (`…`).
      3. Convert Markdown links [text](url) → text.
      4. Remove <thinking>…</thinking> internal monologue blocks.
      5. Remove bare URLs (https://…).
      6. Strip remaining Markdown symbols (#, *, _, >, ~).
      7. Collapse whitespace.
    """
    if not text:
        return ""
    text = _CODE_BLOCK_RE.sub("", text)       # 1
    text = _INLINE_CODE_RE.sub("", text)      # 2
    text = _MD_LINK_RE.sub(r"\1", text)       # 3
    text = _THINKING_RE.sub("", text)         # 4
    text = _URL_RE.sub("", text)              # 5
    text = _MD_SYMBOLS_RE.sub("", text)       # 6
    text = re.sub(r"\s{2,}", " ", text)       # 7 — collapse whitespace
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# TTS Engine
# ═══════════════════════════════════════════════════════════════════════════


class TTSEngine:
    """Streaming Piper TTS with aplay playback and interrupt support."""

    def __init__(self) -> None:
        self._piper_path: str | None = None
        self._aplay_path: str | None = None
        self._sox_path: str | None = None
        self._model_path: str | None = None
        self._sample_rate: int = 22050  # read from model .onnx.json
        self._speaker: int = 0
        self._rate: float = 1.0
        self._playing: asyncio.subprocess.Process | None = None
        self._sox_proc: asyncio.subprocess.Process | None = None
        self._speaking: bool = False
        self._stop_requested: bool = False
        self._speak_task: asyncio.Task | None = None

    # ── Init (lazy) ─────────────────────────────────────────────────────

    def _ensure_init(self) -> bool:
        """Discover Piper and aplay binaries. Returns True if ready."""
        if self._piper_path is not None:
            return bool(self._piper_path and self._aplay_path)

        from bantz.config import config

        # Find piper
        piper = shutil.which("piper")
        if not piper:
            # Check common locations
            for p in [
                Path.home() / ".local" / "bin" / "piper",
                Path("/usr/local/bin/piper"),
                Path("/usr/bin/piper"),
            ]:
                if p.exists():
                    piper = str(p)
                    break

        # Find aplay
        aplay = shutil.which("aplay")
        if not aplay:
            for p in [Path("/usr/bin/aplay"), Path("/usr/local/bin/aplay")]:
                if p.exists():
                    aplay = str(p)
                    break

        if not piper:
            log.warning("TTS: piper binary not found — audio disabled")
            self._piper_path = ""
            self._aplay_path = ""
            return False

        if not aplay:
            log.warning("TTS: aplay binary not found — audio disabled")
            self._piper_path = ""
            self._aplay_path = ""
            return False

        self._piper_path = piper
        self._aplay_path = aplay

        # Resolve model path
        model = config.tts_model_path
        if not model:
            # Auto-discover: check ~/.local/share/piper-voices/
            model_name = config.tts_model
            search_dirs = [
                Path.home() / ".local" / "share" / "piper-voices",
                Path.home() / ".local" / "share" / "piper" / "voices",
                Path.home() / ".local" / "share" / "piper",
                Path("/usr/share/piper-voices"),
            ]
            for d in search_dirs:
                candidate = d / f"{model_name}.onnx"
                if candidate.exists():
                    model = str(candidate)
                    break
                # Try nested: en_US-lessac-medium/en_US-lessac-medium.onnx
                candidate2 = d / model_name / f"{model_name}.onnx"
                if candidate2.exists():
                    model = str(candidate2)
                    break

        if not model or not Path(model).exists():
            log.warning(
                "TTS: voice model not found (expected %s) — run: "
                "piper --download-dir ~/.local/share/piper-voices "
                "--model %s --update-voices",
                config.tts_model, config.tts_model,
            )
            self._piper_path = ""
            return False

        self._model_path = model
        self._speaker = config.tts_speaker
        self._rate = config.tts_rate

        # Read sample rate from model config (.onnx.json)
        import json as _json
        model_json = Path(model).with_suffix(".onnx.json")
        if not model_json.exists():
            model_json = Path(str(model) + ".json")
        if model_json.exists():
            try:
                meta = _json.loads(model_json.read_text())
                sr = meta.get("audio", {}).get("sample_rate", 22050)
                self._sample_rate = int(sr)
                log.info("TTS: model sample_rate=%d", self._sample_rate)
            except Exception as exc:
                log.warning("TTS: could not read model config — %s", exc)
        else:
            log.debug("TTS: no .onnx.json found, assuming 22050 Hz")

        # Discover sox for animatronic filter (#248)
        sox = shutil.which("sox")
        if sox:
            self._sox_path = sox
            log.info("TTS: sox found at %s (animatronic filter available)", sox)
        else:
            self._sox_path = ""
            log.debug("TTS: sox not found — animatronic filter unavailable")

        log.info("TTS: ready — piper=%s model=%s", piper, model)
        return True

    # ── Public API ──────────────────────────────────────────────────────

    def available(self) -> bool:
        """Check if TTS is available (Piper + aplay + model found)."""
        from bantz.config import config
        if not config.tts_enabled:
            return False
        return self._ensure_init()

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def speak(self, text: str) -> None:
        """Speak text with streaming sentence-by-sentence pipeline.

        Synthesizes the next sentence while playing the current one.
        Can be interrupted at any time via stop().
        """
        if not self.available():
            return

        self._stop_requested = False
        self._speaking = True

        # Audio ducking: lower other apps' volume while we speak
        ducked = False
        try:
            from bantz.config import config
            if config.audio_duck_enabled:
                from bantz.agent.audio_ducker import audio_ducker
                ducked = audio_ducker.duck()
        except Exception as exc:
            log.debug("TTS: audio duck failed — %s", exc)

        try:
            sentences = split_sentences(text)
            if not sentences:
                return

            log.info("TTS: speaking %d sentences", len(sentences))

            # Pre-synthesize first sentence
            next_wav: bytes | None = await self._synthesize(sentences[0])

            for i, sentence in enumerate(sentences):
                if self._stop_requested:
                    log.info("TTS: stop requested, aborting at sentence %d/%d", i + 1, len(sentences))
                    break

                # Use pre-synthesized wav for current sentence
                current_wav = next_wav

                # Start synthesizing next sentence in background
                synth_task: asyncio.Task | None = None
                if i + 1 < len(sentences):
                    synth_task = asyncio.create_task(
                        self._synthesize(sentences[i + 1])
                    )

                # Play current sentence
                if current_wav:
                    await self._play(current_wav)

                # Collect next sentence's wav
                if synth_task:
                    next_wav = await synth_task
                else:
                    next_wav = None

        except asyncio.CancelledError:
            log.info("TTS: speak task cancelled")
            self._kill_playback()
        except Exception as exc:
            log.error("TTS: speak error — %s", exc)
        finally:
            # Restore ducked audio
            if ducked:
                try:
                    from bantz.agent.audio_ducker import audio_ducker
                    audio_ducker.restore()
                except Exception as exc:
                    log.debug("TTS: audio restore failed — %s", exc)
            self._speaking = False
            self._playing = None

    async def speak_background(self, text: str) -> None:
        """Speak in background (fire-and-forget, interruptible)."""
        if self._speak_task and not self._speak_task.done():
            self.stop()
            # Give previous task a moment to clean up
            await asyncio.sleep(0.1)
        self._speak_task = asyncio.create_task(self.speak(text))

    def stop(self) -> None:
        """Immediately stop all TTS playback ("Sessiz ol Bantz!")."""
        self._stop_requested = True
        self._kill_playback()
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
        log.info("TTS: stopped by user")

    # ── Internal: Synthesis ─────────────────────────────────────────────

    async def _synthesize(self, sentence: str) -> bytes | None:
        """Synthesize a single sentence via Piper → raw WAV bytes."""
        if not sentence or not self._piper_path or not self._model_path:
            return None

        cmd = [
            self._piper_path,
            "--model", self._model_path,
            "--output-raw",
        ]
        if self._speaker > 0:
            cmd.extend(["--speaker", str(self._speaker)])
        if self._rate != 1.0:
            cmd.extend(["--length-scale", str(1.0 / self._rate)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=sentence.encode("utf-8")),
                timeout=30.0,
            )
            if proc.returncode != 0:
                log.warning("TTS: piper failed — %s", stderr.decode(errors="replace")[:200])
                return None
            return stdout
        except asyncio.TimeoutError:
            log.warning("TTS: piper synthesis timed out for: %s", sentence[:50])
            return None
        except Exception as exc:
            log.warning("TTS: synthesis error — %s", exc)
            return None

    # ── Internal: Playback ──────────────────────────────────────────────

    async def _play(self, wav_data: bytes) -> None:
        """Play raw WAV data via aplay, optionally through SoX (#248).

        When ``config.tts_animatronic_filter`` is enabled **and** ``sox`` is
        available, the pipeline becomes:
            stdin → sox (pitch/reverb/overdrive) → aplay
        Otherwise falls back to direct aplay playback.

        Sets PULSE_PROP so PulseAudio/PipeWire labels this stream as
        'BantzTTS' — the audio ducker uses this to skip Bantz's own audio.
        """
        if not wav_data or not self._aplay_path:
            return

        import os
        env = os.environ.copy()
        env["PULSE_PROP"] = "application.name='BantzTTS'"

        # Determine if animatronic SoX filter should be active
        use_sox = False
        try:
            from bantz.config import config
            use_sox = config.tts_animatronic_filter and bool(self._sox_path)
        except Exception:
            pass

        try:
            if use_sox:
                # Two-stage: stdin → sox (effects) → memory → aplay
                # NOTE: asyncio.StreamReader can't be passed as stdin
                # to another subprocess, so we collect sox output first.
                sox_proc = await asyncio.create_subprocess_exec(
                    self._sox_path,
                    "-t", "raw", "-r", str(self._sample_rate), "-e", "signed",
                    "-b", "16", "-c", "1", "-",   # input spec
                    "-t", "raw", "-r", str(self._sample_rate), "-e", "signed",
                    "-b", "16", "-c", "1", "-",   # output spec (explicit)
                    "pitch", "-300",
                    "reverb", "50",
                    "overdrive", "10",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=env,
                )
                self._sox_proc = sox_proc
                processed, _ = await asyncio.wait_for(
                    sox_proc.communicate(input=wav_data),
                    timeout=30.0,
                )
                self._sox_proc = None

                if not processed:
                    log.warning("TTS: sox produced no output")
                    return

                # Play processed audio via aplay
                proc = await asyncio.create_subprocess_exec(
                    self._aplay_path,
                    "-r", str(self._sample_rate),
                    "-f", "S16_LE",
                    "-t", "raw",
                    "-c", "1",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=env,
                )
                self._playing = proc
                await proc.communicate(input=processed)
                self._playing = None
            else:
                # Direct: stdin → aplay
                proc = await asyncio.create_subprocess_exec(
                    self._aplay_path,
                    "-r", str(self._sample_rate),
                    "-f", "S16_LE",
                    "-t", "raw",
                    "-c", "1",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=env,
                )
                self._playing = proc
                await proc.communicate(input=wav_data)
                self._playing = None
        except asyncio.CancelledError:
            self._kill_playback()
            raise
        except Exception as exc:
            log.warning("TTS: playback error — %s", exc)
            self._playing = None
            self._sox_proc = None

    def _kill_playback(self) -> None:
        """Send SIGTERM to active playback processes (aplay + sox)."""
        for proc in (self._playing, self._sox_proc):
            if proc and proc.returncode is None:
                try:
                    proc.send_signal(signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
        self._playing = None
        self._sox_proc = None

    # ── Diagnostics ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "available": self.available(),
            "speaking": self._speaking,
            "piper": self._piper_path or "not found",
            "aplay": self._aplay_path or "not found",
            "sox": self._sox_path or "not found",
            "model": self._model_path or "not found",
        }

    def status_line(self) -> str:
        if not self._ensure_init():
            return "tts=unavailable"
        state = "speaking" if self._speaking else "idle"
        return f"tts={state} model={Path(self._model_path or '').stem}"


# ═══════════════════════════════════════════════════════════════════════════
# Module singleton
# ═══════════════════════════════════════════════════════════════════════════

tts_engine = TTSEngine()
