#!/usr/bin/env python3
"""Exercise the server's VisionService MLX backend directly (no HTTP server)."""
import importlib.util
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "app-server.py"

spec = importlib.util.spec_from_file_location("tars_server", SERVER)
mod = importlib.util.module_from_spec(spec)
sys.modules["tars_server"] = mod
spec.loader.exec_module(mod)

VisionService = mod.VisionService

MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
print(f"[init] VisionService({MODEL})")
t0 = time.time()
svc = VisionService(model_name=MODEL, device="auto")
print(f"[init] backend={svc.backend!r} loaded in {time.time()-t0:.1f}s")
assert svc.backend == "mlx", f"expected mlx backend, got {svc.backend}"

img = (Path(__file__).parent / "test_cats.jpg").read_bytes()

for prompt in ["Describe this image in one sentence.",
               "How many cats are there? Answer with just the number."]:
    t1 = time.time()
    out = svc.caption(img, prompt=prompt)
    print(f"\nQ: {prompt}\nA: {out}\n   ({time.time()-t1:.1f}s)")

print("\n[ok] VisionService MLX backend works end-to-end")
