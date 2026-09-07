#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)
api = HfApi(token=os.getenv("HF_TOKEN"))

candidates = [
    "google/gemma-4-E4B-it",
    "unsloth/gemma-4-E4B-it",
    "google/gemma-4-e4b-it",
    "google/gemma-4-4b-it",
]

for repo in candidates:
    try:
        info = api.model_info(repo)
        print(f"FOUND: {repo} (id={info.id}, siblings={[s.rfilename for s in info.siblings[:5]]})")
    except Exception as e:
        print(f"Not found: {repo}")
