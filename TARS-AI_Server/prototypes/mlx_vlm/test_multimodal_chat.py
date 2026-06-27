#!/usr/bin/env python3
"""Exercise the server's multimodal chat path: extract image -> VLM -> SSE."""
import base64
import importlib.util
import json
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[2] / "app-server.py"
spec = importlib.util.spec_from_file_location("tars_server", SERVER)
mod = importlib.util.module_from_spec(spec)
sys.modules["tars_server"] = mod
spec.loader.exec_module(mod)

img_b64 = base64.b64encode((Path(__file__).parent / "test_cats.jpg").read_bytes()).decode()
messages = [
    {"role": "system", "content": "You are TARS. Be concise and dry."},
    {"role": "user", "content": [
        {"type": "text", "text": "What animals are in this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]},
]

image_bytes, prompt = mod._extract_vision_request(messages)
assert image_bytes is not None, "image not extracted"
assert "What animals" in prompt and "TARS" in prompt, f"prompt missing parts: {prompt!r}"
print(f"[extract] image={len(image_bytes)} bytes; prompt={prompt!r}")

# text-only chat must NOT trigger the vision branch
none_img, _ = mod._extract_vision_request([{"role": "user", "content": "hello"}])
assert none_img is None, "text-only chat wrongly flagged as vision"
print("[extract] text-only correctly ignored")

# run through the real VLM service (what the endpoint calls)
svc = mod.VisionService(model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit", device="auto")
answer = svc.caption(image_bytes, prompt=prompt)
print(f"[vlm] {answer}")
assert answer.strip(), "empty VLM answer"

# persona temperature must flow through to the VLM (directLLM humor setting)
answer_t = svc.caption(image_bytes, prompt=prompt, temperature=0.9)
print(f"[vlm temp=0.9] {answer_t}")
assert answer_t.strip(), "empty VLM answer with temperature"

# verify the SSE framing the endpoint streams back
chunks = list(mod._vlm_sse(answer, svc.model_name))
assert chunks[-1] == "data: [DONE]\n\n", "missing DONE"
first = json.loads(chunks[0].removeprefix("data: ").strip())
assert first["choices"][0]["delta"]["content"] == answer
assert first["object"] == "chat.completion.chunk"
print(f"[sse] {len(chunks)} chunks, framing OK")
print("[ok] multimodal chat path works end-to-end (directLLM)")
