import inspect
from transformers import AutoProcessor, AutoConfig, Gemma4ForConditionalGeneration, Gemma4Processor

print("Gemma4Processor init args:")
print(inspect.signature(Gemma4Processor.__init__))

try:
    processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
    print("Successfully loaded AutoProcessor for google/gemma-4-E4B-it")
    print("Processor components:", processor)
except Exception as e:
    print("Error loading AutoProcessor:", e)
