import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import librosa
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

print("Starting test with vision on CPU and gradient checkpointing...")

device_map = {
    "model.vision_tower": "cpu",
    "model.embed_vision": "cpu",
    "model.audio_tower": "cuda:0",
    "model.embed_audio": "cuda:0",
    "model.language_model": "cuda:0",
    "lm_head": "cuda:0",
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_skip_modules=["audio_tower", "vision_tower", "embed_audio", "embed_vision", "lm_head"],
)

model = Gemma4ForConditionalGeneration.from_pretrained(
    "google/gemma-4-E4B-it",
    quantization_config=bnb_config,
    device_map=device_map,
    low_cpu_mem_usage=True,
)

print("Model loaded. VRAM allocated (GB):", torch.cuda.memory_allocated() / 1e9)

for param in model.parameters():
    param.requires_grad = False

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=r".*language_model.*(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)$",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

orig_get_audio_features = model.model.get_audio_features
def get_audio_features_detached(*args, **kwargs):
    with torch.no_grad():
        out = orig_get_audio_features(*args, **kwargs)
        out.pooler_output = out.pooler_output.detach()
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            out.last_hidden_state = out.last_hidden_state.detach()
        return out
model.model.get_audio_features = get_audio_features_detached

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")

with open(".data/distillation/train.jsonl") as f:
    lines = f.readlines()
sample = json.loads(lines[28])
print(f"Testing on long sample 28: {sample['audio_path']}")

audio_data, _ = librosa.load(sample["audio_path"], sr=16000)
messages = [
    {"role": "user", "content": [{"type": "audio", "audio": audio_data}, {"type": "text", "text": sample["prompt"]}]},
    {"role": "model", "content": [{"type": "text", "text": json.dumps(sample["target_json"])}]},
]
full_text = processor.apply_chat_template(messages, add_generation_prompt=False)
inputs = processor(text=full_text, audio=audio_data, return_tensors="pt", sampling_rate=16000).to("cuda:0")

prompt_part = full_text.split("<|turn>model\n")[0] + "<|turn>model\n"
p_inputs = processor(text=prompt_part, audio=audio_data, return_tensors="pt", sampling_rate=16000)
labels = inputs["input_ids"].clone()
labels[0, :p_inputs["input_ids"].shape[1]] = -100

peft_model.train()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, peft_model.parameters()), lr=2e-4)

print("Starting forward pass...")
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = peft_model(**inputs, labels=labels)
    loss = outputs.loss
    del outputs
    print("VRAM before backward (GB):", torch.cuda.memory_allocated() / 1e9)

print(f"Forward pass successful! Loss: {loss.item():.4f}")

print("Starting backward pass...")
loss.backward()
del inputs, labels
torch.cuda.empty_cache()
optimizer.step()
print("Backward pass successful! VRAM after step (GB):", torch.cuda.memory_allocated() / 1e9)
print("TRAINING STEP FULLY OPERATIONAL!")
