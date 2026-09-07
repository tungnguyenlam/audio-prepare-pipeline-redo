#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv(override=True)
for key in ["WANDB_API_KEY", "HF_TOKEN", "GEMINI_API_KEY"]:
    val = os.getenv(key)
    if val:
        print(f"{key}: configured (length={len(val)}, prefix={val[:6]}...)")
    else:
        print(f"{key}: NOT FOUND")
