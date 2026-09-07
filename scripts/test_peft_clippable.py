import torch
from transformers import AutoConfig, AutoModelForCausalLM, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model
import re

print("Testing PEFT target modules...")

# In Gemma4, model structure:
# model.language_model.model.layers[i].self_attn.q_proj
# audio_tower.layers[i].self_attn.q_proj is Gemma4ClippableLinear

# Let's test with regex target_modules:
target_modules = r".*language_model.*(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)$"

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load a tiny dummy model or check with model
model = Gemma4ForConditionalGeneration.from_pretrained(
    "google/gemma-4-E4B-it",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
print("PEFT attached successfully!")
