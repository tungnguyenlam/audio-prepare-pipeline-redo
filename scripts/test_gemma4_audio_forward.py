#!/usr/bin/env python3
import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
from transformers import Gemma4Processor, Gemma4ForConditionalGeneration
import inspect

print("Processor call signature:", inspect.signature(Gemma4Processor.__call__))
print("Forward signature:", inspect.signature(Gemma4ForConditionalGeneration.forward))
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
