import torch
from transformers import Gemma4ForConditionalGeneration

device_map = {
    "model.vision_tower": "cpu",
    "model.embed_vision": "cpu",
    "model.audio_tower": "cuda:0",
    "model.embed_audio": "cuda:0",
    "model.language_model": "cuda:0",
    "lm_head": "cuda:0",
}

print("Loading with vision on CPU and audio/language on cuda:0...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    "google/gemma-4-E4B-it",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    low_cpu_mem_usage=True,
)
print("Model loaded successfully!")
print("CUDA memory allocated (GB):", torch.cuda.memory_allocated() / 1e9)
print("CUDA memory reserved (GB):", torch.cuda.memory_reserved() / 1e9)
