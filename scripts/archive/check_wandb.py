#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv(override=True)
import wandb

key = os.getenv("WANDB_API_KEY")
res = wandb.login(key=key)
print("Wandb login success:", res)
