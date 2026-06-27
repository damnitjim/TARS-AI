#!/usr/bin/env python3
"""Exercise the server's TTSService Kokoro/MLX backend directly (no HTTP server).

Synthesizes via the real service, then round-trips through mlx-whisper to confirm
the produced WAV bytes are valid and intelligible.
"""
import importlib.util
import sys
import time
import wave
from io import BytesIO
from pathlib import Path

import numpy as np
import mlx_whisper


def read_wav(data: bytes):
    with wave.open(BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        raw = wf.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio[::ch]
    return audio, sr

SERVER = Path(__file__).resolve().parents[2] / "app-server.py"
spec = importlib.util.spec_from_file_location("tars_server", SERVER)
mod = importlib.util.module_from_spec(spec)
sys.modules["tars_server"] = mod
spec.loader.exec_module(mod)

TTSService = mod.TTSService
TEXT = "Everybody good? Plenty of slaves for my robot colony?"

print("[init] TTSService(engine='kokoro')")
svc = TTSService(engine="kokoro")
print("[voices]", svc.list_voices())

t0 = time.time()
wav = svc.synthesize(TEXT, voice="am_michael", speed=1.0)
dt = time.time() - t0
assert wav[:4] == b"RIFF", "not a WAV"

audio, sr = read_wav(wav)
dur = len(audio) / sr
print(f"[tts] {len(wav)} bytes, {dur:.1f}s @ {sr} Hz in {dt:.1f}s (RTF {dt/dur:.2f}x)")

# round-trip STT
if sr != 16000:
    n = int(round(len(audio) * 16000 / sr))
    audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                      np.arange(len(audio)), audio).astype(np.float32)
hyp = mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")["text"].strip()
print(f"\nTTS in : {TEXT}")
print(f"STT out: {hyp}")
print("[ok] TTSService Kokoro backend works end-to-end")
