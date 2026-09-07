#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)
api = HfApi(token=os.getenv("HF_TOKEN"))
for r in ["google/gemma-4-E4B-it", "unsloth/gemma-4-E4B-it"]:
    info = api.model_info(r)
    print(f"Files in {r}:")
    for s in info.siblings:
        print(" ", s.rfilename)
