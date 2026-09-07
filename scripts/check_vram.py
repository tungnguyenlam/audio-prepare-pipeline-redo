#!/usr/bin/env python3
import subprocess

out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True)
for line in out.stdout.splitlines():
    if "VRAM Total" in line:
        print(line.strip())
