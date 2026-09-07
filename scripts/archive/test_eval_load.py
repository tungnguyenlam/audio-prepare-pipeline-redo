from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from peft.utils import set_peft_model_state_dict
import safetensors.torch
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration

ADAPTER_PATH = Path(".data/distillation/checkpoints/best_adapter")
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
base_model = Gemma4ForConditionalGeneration.from_pretrained(
    "google/gemma-4-E4B-it",
    quantization_config=bnb_config,
    device_map=device_map,
    low_cpu_mem_usage=True,
)
peft_config = LoraConfig.from_pretrained(str(ADAPTER_PATH))
model = get_peft_model(base_model, peft_config)
weights = safetensors.torch.load_file(str(ADAPTER_PATH / "adapter_model.safetensors"))
set_peft_model_state_dict(model, weights)
model.eval()
print("ADAPTER LOADED SUCCESSFULLY WITH GET_PEFT_MODEL!")
