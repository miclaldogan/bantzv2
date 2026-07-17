#!/usr/bin/env python
"""Train the local "Bantz" wake-word classifier (#551).

Fully offline pipeline:
  1. Synthesize positive clips ("bantz" and phonetic variants) and
     adversarial negatives with the locally installed Piper voices —
     the Turkish voice anchors the intended pronunciation (/bants/,
     open Turkish 'a', NOT English "bænts").
  2. Augment with sox: speed (piper length-scale), pitch, gain, noise.
  3. Embed every clip with openWakeWord's bundled melspectrogram +
     embedding ONNX models (the same features the runtime engine uses).
  4. Train a small scikit-learn classifier head on the final 16-frame
     embedding window and report held-out FA/FR.
  5. Save to ~/.local/share/bantz/wakewords/bantz_clf.pkl — picked up by
     BANTZ_WAKE_ENGINE=openwakeword.

Drop real recordings (16 kHz mono wav, word near the end) into
  ~/.local/share/bantz/wakewords/real_positives/   and/or
  ~/.local/share/bantz/wakewords/real_negatives/
and re-run to fine-tune on your actual voice/mic.

Usage:  python scripts/train_wakeword.py [--fresh]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path.home() / ".local/share/bantz/wakewords"
CORPUS = OUT_DIR / "corpus"
SR = 16000
CLIP_S = 2.0  # word sits at the END of a 2s clip, mirroring streaming detection

VOICES = {
    "tr": Path.home() / ".local/share/bantz/tr_TR-dfki-medium.onnx",
    "en": Path.home() / ".local/share/piper-voices/en_US-danny-low.onnx",
}

# Positive texts per voice. The Turkish voice reads "bantz" exactly as the
# user says it; the English voice gets phonetic spellings that pull the
# vowel from /æ/ toward the Turkish open /a/.
POSITIVE_TEXTS = {
    "tr": ["bantz", "bants", "Bantz.", "bantz!"],
    "en": ["bahnts", "bunts", "baants", "bonts."],
}

# Adversarial negatives — near-miss words + normal speech in both languages.
NEGATIVE_TEXTS = {
    "tr": ["banka", "banyo", "bando", "bant", "bana bak", "bence", "benzin",
           "battaniye", "pantolon", "merhaba", "saat kaç", "bugün hava nasıl",
           "banliyö treni geldi", "bantlı kutu masada"],
    "en": ["pants", "bands", "bank", "bounce", "benz", "thanks", "hands",
           "what time is it", "open the door please", "hello there",
           "turn on the lights", "bounce the ball once"],
}

LENGTH_SCALES = [0.85, 1.0, 1.2]
PITCHES = [-200, 0, 200]      # cents (sox)
NOISE_AMPS = [0.0, 0.004]     # additive white noise amplitude


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[:200]}")


def synth(voice: Path, text: str, length_scale: float, out: Path) -> None:
    r = subprocess.run(
        ["piper", "-m", str(voice), "-f", str(out), "--length-scale", str(length_scale)],
        input=text, capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"piper failed: {r.stderr[:200]}")


def augment(src: Path, dst: Path, pitch: int, noise_amp: float) -> None:
    """sox: resample to 16k mono, pitch shift, optional noise bed."""
    _run(["sox", str(src), "-r", str(SR), "-c", "1", "-b", "16", str(dst),
          "pitch", str(pitch)])
    if noise_amp > 0:
        dur = subprocess.run(["soxi", "-D", str(dst)], capture_output=True,
                             text=True).stdout.strip() or "2.0"
        noise = dst.with_suffix(".noise.wav")
        noisy = dst.with_suffix(".noisy.wav")
        _run(["sox", "-n", "-r", str(SR), "-c", "1", "-b", "16", str(noise),
              "synth", dur, "whitenoise", "vol", str(noise_amp)])
        _run(["sox", "-m", str(dst), str(noise), str(noisy)])
        noise.unlink(missing_ok=True)
        noisy.replace(dst)


def load_clip(path: Path) -> np.ndarray | None:
    """Load wav → int16 mono 16k, pad/trim so speech ends the clip."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getframerate() != SR or w.getnchannels() != 1:
                return None
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except Exception:
        return None
    target = int(CLIP_S * SR)
    if len(pcm) >= target:
        return pcm[-target:]
    return np.concatenate([np.zeros(target - len(pcm), dtype=np.int16), pcm])


def build_corpus(fresh: bool) -> tuple[list[Path], list[Path]]:
    pos_dir, neg_dir = CORPUS / "positive", CORPUS / "negative"
    if fresh:
        for d in (pos_dir, neg_dir):
            if d.is_dir():
                for f in d.glob("*.wav"):
                    f.unlink()
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    def generate(texts: dict[str, list[str]], dest: Path, tag: str) -> None:
        raw = CORPUS / "raw"
        raw.mkdir(exist_ok=True)
        n = 0
        for vkey, vpath in VOICES.items():
            if not vpath.exists():
                print(f"  ! voice {vkey} missing at {vpath} — skipped")
                continue
            for ti, text in enumerate(texts[vkey]):
                for ls in LENGTH_SCALES:
                    base = raw / f"{tag}_{vkey}_{ti}_{ls}.wav"
                    if not base.exists():
                        synth(vpath, text, ls, base)
                    for pitch in PITCHES:
                        for na in NOISE_AMPS:
                            out = dest / f"{tag}_{vkey}_{ti}_{ls}_{pitch}_{na}.wav"
                            if not out.exists():
                                augment(base, out, pitch, na)
                            n += 1
        print(f"  {tag}: {n} clips")

    print("Generating corpus (cached — re-runs are incremental)...")
    generate(POSITIVE_TEXTS, pos_dir, "pos")
    generate(NEGATIVE_TEXTS, neg_dir, "neg")

    # Pure noise/silence negatives
    for i, vol in enumerate([0.0, 0.02, 0.1]):
        out = neg_dir / f"neg_noise_{i}.wav"
        if not out.exists():
            if vol == 0.0:
                _run(["sox", "-n", "-r", str(SR), "-c", "1", "-b", "16",
                      str(out), "trim", "0", str(CLIP_S)])
            else:
                _run(["sox", "-n", "-r", str(SR), "-c", "1", "-b", "16",
                      str(out), "synth", str(CLIP_S), "whitenoise", "vol", str(vol)])

    # Optional real recordings
    pos = sorted(pos_dir.glob("*.wav")) + sorted((OUT_DIR / "real_positives").glob("*.wav"))
    neg = sorted(neg_dir.glob("*.wav")) + sorted((OUT_DIR / "real_negatives").glob("*.wav"))
    return pos, neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="regenerate augmented clips")
    args = ap.parse_args()

    pos_files, neg_files = build_corpus(args.fresh)
    print(f"Corpus: {len(pos_files)} positive / {len(neg_files)} negative clips")

    X_list, y = [], []
    for files, label in ((pos_files, 1), (neg_files, 0)):
        for f in files:
            clip = load_clip(f)
            if clip is not None:
                X_list.append(clip)
                y.append(label)
    X = np.stack(X_list)
    y = np.array(y)

    print("Embedding clips with openWakeWord feature models (CPU)...")
    from openwakeword.utils import AudioFeatures
    feats = AudioFeatures(ncpu=2)
    emb = feats.embed_clips(X, batch_size=64)          # (N, frames, 96)
    win = emb[:, -16:, :].reshape(len(emb), -1)        # final 16-frame window

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        win, y, test_size=0.25, stratify=y, random_state=7,
    )
    clf = LogisticRegression(max_iter=3000, C=0.5)
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    for thr in (0.5, 0.7, 0.9):
        fr = float(np.mean(proba[yte == 1] < thr))
        fa = float(np.mean(proba[yte == 0] >= thr))
        print(f"  threshold {thr}: false-reject {fr:.1%}  false-accept {fa:.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "bantz_clf.pkl"
    import joblib
    joblib.dump({"clf": clf, "n_frames": 16, "trained_on": len(y)}, out)
    print(f"Saved {out}")
    print("Enable with: BANTZ_WAKE_ENGINE=openwakeword + restart the daemon.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
