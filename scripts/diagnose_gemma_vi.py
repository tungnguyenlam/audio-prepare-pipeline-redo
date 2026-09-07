#!/usr/bin/env python3
import sys, json, base64, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.test_vietnamese_prompt import VI_PROMPT

audio_path = Path(".data/experiment_khanhvy/cuts/turn_001_spk_00_0.10-1.85.wav")
with open(audio_path, "rb") as f:
    audio_bytes = f.read()

payload = {
    "model": "unsloth/gemma-4-12B-it-qat-GGUF",
    "messages": [{"role": "user", "content": VI_PROMPT}],
    "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    "temperature": 0.1,
    "max_tokens": 1024,
}
req = urllib.request.Request(
    "http://localhost:8889/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": "Bearer sk-unsloth-d5e3632a095b7e04619fd36a650a9162",
        "Content-Type": "application/json",
    },
)
with urllib.request.urlopen(req, timeout=120.0) as resp:
    res = json.loads(resp.read().decode())

choice = res["choices"][0]
print("finish_reason:", choice.get("finish_reason"))
msg = choice["message"]
print("content:", repr(msg.get("content")))
print("reasoning_content:", repr(msg.get("reasoning_content")[:300] if msg.get("reasoning_content") else None))
