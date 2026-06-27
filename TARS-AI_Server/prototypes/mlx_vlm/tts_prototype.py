#!/usr/bin/env python3
"""mlx-audio + Kokoro TTS prototype.

Synthesizes speech on Apple Silicon via MLX, then round-trips it through
mlx-whisper to confirm the audio is intelligible (TTS -> STT -> compare).
"""
import glob
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import mlx_whisper
from mlx_audio.tts.generate import generate_audio

HERE = Path(__file__).parent
TTS_MODEL = "prince-canuma/Kokoro-82M"
VOICE = "af_heart"
STT_MODEL = "mlx-community/whisper-large-v3-turbo"
TEXT = "The mission requires absolute precision and a healthy sense of humor."
PREFIX = "kokoro_out"


def main() -> None:
    for f in glob.glob(str(HERE / f"{PREFIX}*")):
        Path(f).unlink()

    print(f"[tts] model={TTS_MODEL} voice={VOICE}")
    t0 = time.time()
    generate_audio(
        text=TEXT, model=TTS_MODEL, voice=VOICE,
        output_path=str(HERE), file_prefix=PREFIX, audio_format="wav",
        join_audio=True, save=True, verbose=False,
    )
    t_synth = time.time() - t0

    outs = sorted(glob.glob(str(HERE / f"{PREFIX}*.wav")))
    assert outs, "no wav produced"
    wav = outs[0]
    audio, sr = sf.read(wav, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    print(f"[tts] wrote {Path(wav).name}  {dur:.1f}s @ {sr} Hz  in {t_synth:.1f}s "
          f"(RTF {t_synth/dur:.2f}x)")

    # round-trip: resample to 16k and transcribe
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio16 = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                            np.arange(len(audio)), audio).astype(np.float32)
    else:
        audio16 = audio
    hyp = mlx_whisper.transcribe(audio16, path_or_hf_repo=STT_MODEL)["text"].strip()

    norm = lambda s: re.findall(r"[a-z]+", s.lower())
    r, h = norm(TEXT), norm(hyp)
    hits = sum(1 for a, b in zip(r, h) if a == b)
    print(f"\nTTS in : {TEXT}")
    print(f"STT out: {hyp}")
    print(f"round-trip word match: {hits}/{len(r)}")


if __name__ == "__main__":
    main()
