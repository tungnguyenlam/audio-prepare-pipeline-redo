#!/usr/bin/env python3
import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
import unsloth
import unsloth_zoo
import inspect

zoo_items = [x for x in dir(unsloth_zoo) if any(k in x.lower() for k in ['audio', 'gemma', 'multimodal', 'speech', 'voice'])]
print("Matching unsloth_zoo items:", zoo_items)

import transformers
models_with_gemma = [m for m in dir(transformers) if 'gemma' in m.lower()]
print("Gemma in transformers:", models_with_gemma)
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
