#!/usr/bin/env python3
import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
import unsloth
from unsloth import FastLanguageModel
import inspect

print("Checking unsloth methods and attributes:")
methods = [m for m in dir(unsloth) if not m.startswith('_')]
print("Unsloth exports:", methods)

# Check FastLanguageModel supported models or multimodal
flm_attrs = [a for a in dir(FastLanguageModel) if not a.startswith('_')]
print("FastLanguageModel exports:", flm_attrs)

try:
    from unsloth import FastVisionModel
    fvm_attrs = [a for a in dir(FastVisionModel) if not a.startswith('_')]
    print("FastVisionModel exports:", fvm_attrs)
except Exception as e:
    print("FastVisionModel error:", e)

# Check if there is FastAudioModel or FastMultimodalModel
for name in ["FastAudioModel", "FastMultimodalModel", "FastSpeechModel"]:
    print(f"{name} in unsloth:", hasattr(unsloth, name))
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
