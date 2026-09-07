import os
from dotenv import load_dotenv

load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import snapshot_download

token = os.getenv("HF_TOKEN")
model_id = "google/gemma-4-E4B-it"

print(f"Starting snapshot download with hf_transfer for {model_id}...")
path = snapshot_download(
    repo_id=model_id,
    token=token,
    resume_download=True,
)
print(f"Downloaded {model_id} to: {path}")
