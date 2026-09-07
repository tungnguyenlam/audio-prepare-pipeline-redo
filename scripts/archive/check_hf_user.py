#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(override=True)
api = HfApi(token=os.getenv("HF_TOKEN"))
user_info = api.whoami()
print("HF Username:", user_info["name"])
print("HF Email:", [e.get("email") for e in user_info.get("emails", [])])
