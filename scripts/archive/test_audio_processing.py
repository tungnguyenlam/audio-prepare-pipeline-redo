import os
import json
import librosa
import torch
from dotenv import load_dotenv

load_dotenv()

from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=os.getenv("HF_TOKEN"))

# Load first sample
with open(".data/distillation/train.jsonl") as f:
    sample = json.loads(f.readline())

audio_path = sample["audio_path"]
prompt = sample["prompt"]
target_str = json.dumps(sample["target_json"], ensure_ascii=False)

# Load audio at 16kHz
audio_data, sr = librosa.load(audio_path, sr=16000)
print(f"Loaded audio: {audio_path}, shape: {audio_data.shape}, sr: {sr}")

# Construct messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_data},
            {"type": "text", "text": prompt},
        ],
    },
    {
        "role": "model",
        "content": [
            {"type": "text", "text": target_str}
        ],
    },
]

# Apply chat template
text = processor.apply_chat_template(messages, add_generation_prompt=False)
print("Rendered chat template (first 300 chars):", repr(text[:300]))
print("Rendered chat template (last 300 chars):", repr(text[-300:]))

# Process with processor
inputs = processor(
    text=text,
    audio=audio_data,
    return_tensors="pt",
    sampling_rate=16000,
)
print("Input keys:", inputs.keys())
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")
