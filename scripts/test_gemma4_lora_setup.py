import os
import torch
from dotenv import load_dotenv
from transformers import AutoConfig, Gemma4ForConditionalGeneration
from peft import LoraConfig, get_peft_model

load_dotenv()

print("CUDA available:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/gemma-4-E4B-it"
print(f"Loading config for {model_id}...")
config = AutoConfig.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
print("Model architecture:", config.architectures)

# Let's inspect linear layers in config or load with device_map="cpu" first to inspect
print("Loading model on CPU with bfloat16 to inspect module names...")
model = Gemma4ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    token=os.getenv("HF_TOKEN"),
)
print("Model loaded successfully!")

target_modules = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        # get the last submodule name
        sub_name = name.split(".")[-1]
        if sub_name not in target_modules:
            target_modules.append(sub_name)

print("Linear layer module names:", target_modules)

# Setup LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
