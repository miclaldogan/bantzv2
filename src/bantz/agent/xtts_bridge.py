"""
XTTS bridge — high-quality Turkish TTS via a persistent subprocess.

Coqui XTTS v2 lives in a separate conda env (its transformers pin
conflicts with the bantz venv), so synthesis goes through
scripts/xtts_server.py over a JSON-lines pipe. The model loads once
(~17s, lazily on the first Turkish sentence) and stays resident; an
idle timer shuts the subprocess down to reclaim the ~2 GB VRAM.

synth() returns RAW 16-bit mono PCM resampled (via sox) to the sample
rate the TTS engine's player pipeline expects, so agent/tts.py can play
it exactly like Piper output.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

log = logging.getLogger("bantz.xtts")

_SERVER_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "xtts_server.py"


class XTTSBridge:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._last_used = 0.0
        self._unload_timer: threading.Timer | None = None
        self._failed = False

    def available(self) -> bool:
        from bantz.config import config
        if self._failed:
            return False
        python = Path(os.path.expanduser(config.xtts_python))
        return python.exists() and _SERVER_SCRIPT.exists()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def _ensure_started(self) -> bool:
        """Start the server subprocess and wait for its ready line."""
        if self._proc is not None and self._proc.poll() is None:
            return True
        from bantz.config import config
        python = os.path.expanduser(config.xtts_python)
        try:
            log.info("XTTS: starting server (first Turkish sentence pays the ~17s model load)")
            self._proc = subprocess.Popen(
                [python, str(_SERVER_SCRIPT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("ready"):
                    log.info("XTTS: ready on %s (load %.1fs)",
                             msg.get("device"), msg.get("load_s", 0))
                    return True
            log.error("XTTS: server did not become ready")
            self.shutdown()
            self._failed = True
            return False
        except Exception as exc:
            log.error("XTTS: failed to start server — %s", exc)
            self.shutdown()
            self._failed = True
            return False

    def shutdown(self) -> None:
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
            log.info("XTTS: server stopped")

    def _schedule_idle_unload(self) -> None:
        from bantz.config import config
        minutes = int(getattr(config, "xtts_idle_unload_min", 10))
        if minutes <= 0:
            return
        self._last_used = time.monotonic()
        if self._unload_timer is not None:
            self._unload_timer.cancel()

        def _unload() -> None:
            if time.monotonic() - self._last_used >= minutes * 60 - 1:
                log.info("XTTS: idle %d min — unloading", minutes)
                self.shutdown()

        self._unload_timer = threading.Timer(minutes * 60, _unload)
        self._unload_timer.daemon = True
        self._unload_timer.start()

    # ── synthesis ─────────────────────────────────────────────────────────

    def synth(self, text: str, sample_rate: int, language: str = "tr") -> bytes | None:
        """Synthesize *text* → raw 16-bit mono PCM at *sample_rate*.

        Thread-safe (one request at a time over the pipe). Returns None
        on any failure so the caller can fall back to Piper."""
        from bantz.config import config
        sox = shutil.which("sox")
        if not sox:
            return None
        with self._lock:
            if not self._ensure_started():
                return None
            wav_path = ""
            try:
                fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="bantz_xtts_")
                os.close(fd)
                req = {
                    "text": text,
                    "language": language,
                    "speaker": config.xtts_speaker,
                    "speed": config.xtts_speed,
                    "out": wav_path,
                }
                assert self._proc and self._proc.stdin and self._proc.stdout
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
                resp: dict = {}
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    line = self._proc.stdout.readline()
                    if not line:
                        break
                    try:
                        resp = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue  # stray library output — skip
                if not resp.get("ok"):
                    log.warning("XTTS: synth failed — %s", resp.get("error", "no response"))
                    return None
                r = subprocess.run(
                    [sox, wav_path, "-t", "raw", "-r", str(sample_rate),
                     "-e", "signed", "-b", "16", "-c", "1", "-"],
                    capture_output=True, timeout=15,
                )
                if r.returncode != 0:
                    return None
                self._schedule_idle_unload()
                return r.stdout
            except Exception as exc:
                log.warning("XTTS: synth error — %s (falling back to piper)", exc)
                self.shutdown()
                return None
            finally:
                if wav_path:
                    try:
                        os.unlink(wav_path)
                    except OSError:
                        pass


xtts_bridge = XTTSBridge()
