#!/usr/bin/env python3
"""
MLX-VLM prototype: caption/VQA an image with Qwen2.5-VL-7B (4-bit) on Apple Silicon.

Goal: validate that a quantized VLM runs on this M1 Pro / 32 GB via MLX, and
measure load time, generation latency, and peak memory — the numbers that decide
whether to swap the TARS server's transformers vision path for MLX.

Usage:
    .venv/bin/python caption_prototype.py [image] [prompt]
"""
import sys
import time

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


def main() -> None:
    image = sys.argv[1] if len(sys.argv) > 1 else "test_cats.jpg"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in one sentence."

    print(f"[load] {MODEL}")
    t0 = time.time()
    model, processor = load(MODEL)
    config = load_config(MODEL)
    t_load = time.time() - t0
    print(f"[load] done in {t_load:.1f}s")

    formatted = apply_chat_template(processor, config, prompt, num_images=1)

    print(f"[gen] image={image!r}  prompt={prompt!r}")
    t1 = time.time()
    result = generate(
        model, processor, formatted, image=[image],
        max_tokens=128, temperature=0.0, verbose=False,
    )
    t_gen = time.time() - t1

    text = getattr(result, "text", str(result))
    peak_gb = mx.get_peak_memory() / 1e9

    print("\n=== RESULT ===")
    print(text.strip())
    print("\n=== METRICS ===")
    print(f"load:        {t_load:.1f}s")
    print(f"generation:  {t_gen:.1f}s")
    for attr in ("prompt_tokens", "generation_tokens", "generation_tps", "peak_memory"):
        if hasattr(result, attr):
            print(f"{attr}: {getattr(result, attr)}")
    print(f"peak_memory(mx): {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
