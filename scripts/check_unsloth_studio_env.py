#!/usr/bin/env python3
"""Check packages and capabilities inside ~/.unsloth virtual environment."""

import subprocess

cmd = [
    "/home/nguyenlt/.unsloth/studio/unsloth_studio/bin/python",
    "-c",
    """
import pkg_resources
installed = {p.key: p.version for p in pkg_resources.working_set}
print("Installed key packages:")
for k in ["unsloth", "unsloth-zoo", "torch", "torchaudio", "transformers", "peft", "trl"]:
    print(f"  {k}: {installed.get(k, 'NOT INSTALLED')}")

try:
    import unsloth
    print("Unsloth module found at:", unsloth.__file__)
except Exception as e:
    print("Unsloth import error:", e)
"""
]

res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
