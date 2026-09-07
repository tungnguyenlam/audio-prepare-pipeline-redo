import os
import json
import librosa
import torch
from dotenv import load_dotenv

load_dotenv()
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-E4B-it", token=os.getenv("HF_TOKEN"))

with open(".data/distillation/train.jsonl") as f:
    sample = json.loads(f.readline())

audio_data, _ = librosa.load(sample["audio_path"], sr=16000)
prompt = sample["prompt"]
target_str = json.dumps(sample["target_json"], ensure_ascii=False)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_data},
            {"type": "text", "text": prompt},
        ],
    },
    {
        "role": "model",
        "content": [
            {"type": "text", "text": target_str}
        ],
    },
]

full_text = processor.apply_chat_template(messages, add_generation_prompt=False)
inputs = processor(text=full_text, audio=audio_data, return_tensors="pt", sampling_rate=16000)

input_ids = inputs["input_ids"][0]
labels = input_ids.clone()

# Find model turn start: token string "<|turn>model\n"
# In Gemma 4 tokenizer, special token 105 is <|turn>
# Let's inspect where '<|turn>model' occurs
model_turn_marker = "<|turn>model\n"
split_text = full_text.split(model_turn_marker)
assert len(split_text) == 2, "Expected exactly one model turn marker"
prompt_part = split_text[0] + model_turn_marker

# The prompt part processed with the same audio:
prompt_inputs = processor(text=prompt_part, audio=audio_data, return_tensors="pt", sampling_rate=16000)
prompt_len = prompt_inputs["input_ids"].shape[1]

print(f"Total tokens: {len(input_ids)}, Prompt tokens: {prompt_len}, Target tokens: {len(input_ids) - prompt_len}")
labels[:prompt_len] = -100

decoded_target = processor.tokenizer.decode(input_ids[prompt_len:])
print(f"Decoded target from label tokens:\n{decoded_target}")
assert target_str in decoded_target, "Target string must be preserved in labels!"
print("Label masking verified perfectly!")
