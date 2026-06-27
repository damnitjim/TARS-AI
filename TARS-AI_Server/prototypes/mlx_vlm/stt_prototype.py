#!/usr/bin/env python3
"""MLX-Whisper STT prototype — transcribe a known sample, measure latency.

Decodes the WAV with soundfile (no ffmpeg needed) and passes a float32 mono
16 kHz array straight to mlx_whisper.transcribe.
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx_whisper

HERE = Path(__file__).parent
MODEL = "mlx-community/whisper-large-v3-turbo"


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:                       # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != 16000:                          # resample to whisper's 16 kHz
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(
            np.linspace(0, len(audio), n, endpoint=False),
            np.arange(len(audio)), audio,
        ).astype(np.float32)
    return audio


def main() -> None:
    wav = HERE / (sys.argv[1] if len(sys.argv) > 1 else "stt_sample.wav")
    ref = (HERE / "stt_reference.txt").read_text().strip()
    audio = load_audio(wav)
    dur = len(audio) / 16000

    print(f"[stt] model={MODEL}")
    print(f"[stt] clip={wav.name} ({dur:.1f}s)")

    t0 = time.time()
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL)
    elapsed = time.time() - t0

    text = result["text"].strip()
    print(f"\nREF: {ref}")
    print(f"HYP: {text}")
    print(f"\n=== METRICS ===")
    print(f"audio:    {dur:.1f}s")
    print(f"transcribe: {elapsed:.2f}s   (RTF {elapsed/dur:.2f}x)")

    # crude word-level accuracy
    import re
    norm = lambda s: re.findall(r"[a-z]+", s.lower())
    r, h = norm(ref), norm(text)
    hits = sum(1 for a, b in zip(r, h) if a == b)
    print(f"word match: {hits}/{len(r)}")


if __name__ == "__main__":
    main()
