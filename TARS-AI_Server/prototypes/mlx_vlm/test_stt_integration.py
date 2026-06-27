#!/usr/bin/env python3
"""Exercise the server's STTService MLX backend directly (no HTTP server)."""
import importlib.util
import sys
import time
from io import BytesIO
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "app-server.py"
spec = importlib.util.spec_from_file_location("tars_server", SERVER)
mod = importlib.util.module_from_spec(spec)
sys.modules["tars_server"] = mod
spec.loader.exec_module(mod)

STTService = mod.STTService
MODEL = "mlx-community/whisper-large-v3-turbo"

print(f"[init] STTService({MODEL})  vad_filter=False")
svc = STTService(model_size=MODEL, vad_filter=False)
assert svc.use_mlx, "expected MLX backend"

wav = (Path(__file__).parent / "stt_sample.wav").read_bytes()
ref = (Path(__file__).parent / "stt_reference.txt").read_text().strip()

t0 = time.time()
results, info = svc.transcribe(BytesIO(wav))
elapsed = time.time() - t0
text = " ".join(r["text"] for r in results).strip()

print(f"REF: {ref}")
print(f"HYP: {text}")
print(f"lang={info.language} prob={info.language_probability} segments={len(results)} time={elapsed:.2f}s")
assert text, "empty transcription"
print("[ok] STTService MLX backend works end-to-end")
