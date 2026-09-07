#!/usr/bin/env python3
import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
import soundfile as sf
from transformers import Gemma4Processor

processor = Gemma4Processor.from_pretrained("google/gemma-4-E4B-it")
print("Audio feature extractor:", processor.feature_extractor)
print("Chat template:", processor.chat_template[:200] if processor.chat_template else "None")

audio_path = ".data/experiment_khanhvy/cuts/turn_001_spk_00_0.10-1.85.wav"
wav, sr = sf.read(audio_path)
print(f"Loaded audio: shape={wav.shape}, sr={sr}")

# Test message format
messages = [
    {"role": "user", "content": [{"type": "audio", "audio": wav}, {"type": "text", "text": "Is this audio pure?"}]}
]
try:
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print("Formatted text:", repr(text))
    inputs = processor(text=text, audio=wav, sampling_rate=sr, return_tensors="pt")
    print("Inputs keys:", list(inputs.keys()))
    print("input_ids shape:", inputs.input_ids.shape)
    if "input_features" in inputs:
        print("input_features shape:", inputs.input_features.shape)
except Exception as e:
    print("Processor apply error:", e)
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
