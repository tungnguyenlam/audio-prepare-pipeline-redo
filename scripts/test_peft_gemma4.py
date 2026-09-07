#!/usr/bin/env python3
import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
from transformers import Gemma4ForConditionalGeneration, Gemma4Processor
from peft import LoraConfig, get_peft_model
print("Successfully imported Gemma4ForConditionalGeneration and Gemma4Processor!")
print("Peft LoraConfig available.")
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
