#!/usr/bin/env python3
import sys, json, base64, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.diarization.OverlapVerifier import OVERLAP_PROMPT, _normalize_result

audio_path = Path(".data/experiment_khanhvy/cuts/turn_001_spk_00_0.10-1.85.wav")
with open(audio_path, "rb") as f:
    audio_bytes = f.read()

payload = {
    "model": "unsloth/gemma-4-12b-it-GGUF",
    "messages": [{"role": "user", "content": OVERLAP_PROMPT}],
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
t0 = time.time()
with urllib.request.urlopen(req, timeout=120.0) as resp:
    res = json.loads(resp.read().decode())
elapsed = time.time() - t0
msg = res["choices"][0]["message"]
content = msg.get("content", "").strip()
print(f"Elapsed: {elapsed:.2f}s")
print("Content:", content)
parsed = _normalize_result(content, backend="Unsloth-12B-Q6")
print("Parsed:", parsed)
