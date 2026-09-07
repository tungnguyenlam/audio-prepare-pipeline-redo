#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)
api = HfApi(token=os.getenv("HF_TOKEN"))
info = api.model_info("google/gemma-4-E4B-it", files_metadata=True)
for s in info.siblings:
    if s.rfilename.endswith(".safetensors"):
        print(f"{s.rfilename}: {s.size / 1e9:.2f} GB")
