#!/usr/bin/env python3
import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
import os, torch
from dotenv import load_dotenv
load_dotenv()

print("ROCm available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

import unsloth
from unsloth import FastLanguageModel

try:
    print("Testing FastLanguageModel.from_pretrained on unsloth/gemma-4-E4B-it with load_in_4bit=True...")
    # Just inspect if unsloth architecture mapping supports it
    from unsloth.models.loader import FastModel
    print("FastModel available.")
except Exception as e:
    print("Error:", e)
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
