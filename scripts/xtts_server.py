#!/usr/bin/env python
"""Persistent XTTS v2 synthesis server for Bantz (Turkish voice).

Runs under the conda env that has Coqui TTS (butler_tts, from the
bantzAnimatronic project) — NOT the bantz venv, whose transformers
version conflicts with XTTS's <4.45 pin. Loads the 1.8 GB model once
(~17s) and then serves sentences in 0.3-1.3s each on CUDA.

Protocol: JSON lines on stdin/stdout.
  → {"text": "...", "language": "tr", "speaker": "Damien Black",
     "speed": 1.15, "out": "/tmp/x.wav"}
  ← {"ok": true, "wav": "/tmp/x.wav", "ms": 512}
A {"ready": true, "device": "cuda"} line is printed once after load.
"""
from __future__ import annotations

import contextlib
import json
import sys
import time


def main() -> int:
    # Coqui prints progress text to stdout, which would corrupt the JSON
    # protocol — everything the library says goes to stderr; only our
    # JSON lines touch the real stdout.
    proto = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from TTS.api import TTS

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t0 = time.time()
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print(json.dumps({"ready": True, "device": device,
                      "load_s": round(time.time() - t0, 1)}),
          file=proto, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            t1 = time.time()
            with contextlib.redirect_stdout(sys.stderr):
                tts.tts_to_file(
                    text=req["text"],
                    speaker=req.get("speaker", "Damien Black"),
                    language=req.get("language", "tr"),
                    speed=float(req.get("speed", 1.0)),
                    file_path=req["out"],
                )
            print(json.dumps({"ok": True, "wav": req["out"],
                              "ms": int((time.time() - t1) * 1000)}),
                  file=proto, flush=True)
        except Exception as exc:  # keep serving
            print(json.dumps({"ok": False, "error": str(exc)[:300]}),
                  file=proto, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
