import sys
import torch
import transformers
import peft

print("Python:", sys.version)
print("Torch:", torch.__version__, "ROCm:", torch.version.hip)
print("CUDA available:", torch.cuda.is_available(), "Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

print("Transformers:", transformers.__version__)
print("PEFT:", peft.__version__)

# Check for Gemma 4 classes in transformers
gemma_classes = [attr for attr in dir(transformers) if "gemma" in attr.lower() or "gemma4" in attr.lower()]
print("Gemma classes in transformers:", gemma_classes)

# Check auto classes
try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
    print("AutoProcessor / AutoModel available")
except Exception as e:
    print("Error importing auto classes:", e)
