from transformers import Gemma4ForConditionalGeneration

model = Gemma4ForConditionalGeneration.from_pretrained(
    "google/gemma-4-E4B-it",
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    device_map="meta",
)

print("Top level modules:")
for name, child in model.named_children():
    print(f"  {name}: {type(child)}")
    if hasattr(child, "named_children"):
        for sub_name, sub_child in child.named_children():
            print(f"    {sub_name}: {type(sub_child)}")
