#!/usr/bin/env python3
"""Fine-tune Gemma 4 E2B via LoRA knowledge distillation from Gemini 3.8 Flash.

Trains google/gemma-4-E2B-it to evaluate:
1. Speaker Purity (pure, secondary_speaker, overlapping_speech)
2. Word Completeness (complete, clipped_word_start, clipped_word_end)
3. Audio Quality (studio_clean, music_bleed, noisy_reverberant, distorted)
Tracks metrics on Weights & Biases (W&B) and saves adapters locally.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import librosa
import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration, get_cosine_schedule_with_warmup
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("distill_trainer_e2b")

# Ensure repository root in sys.path and load environment
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# Hyperparameters & Constants
MODEL_ID = os.getenv("BASE_MODEL", "google/gemma-4-E2B-it")
HF_HUB_REPO = os.getenv("HF_HUB_REPO", "tungnguyenlam/gemma-4-e2b-acoustic-verifier")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "tungnguyenlam/gemma-4-e2b-acoustic-verifier-data")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "gemma-4-e2b-distill-verifier")
CHECKPOINT_DIR = Path(".data/distillation/checkpoints_e2b/best_adapter")
TRAIN_JSONL = Path(".data/distillation/train_e2b.jsonl")
VAL_JSONL = Path(".data/distillation/val_e2b.jsonl")

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
ACCUM_STEPS = int(os.getenv("ACCUM_STEPS", "4"))  # Effective batch size = 4
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def ensure_dataset(hf_token: str | None = None) -> None:
    """Ensure audio dataset is present locally, downloading and unpacking from HF Hub if needed."""
    audio_dir = REPO_ROOT / ".data" / "distillation_e2b" / "audio"
    if audio_dir.is_dir() and len(list(audio_dir.glob("*.wav"))) >= 50:
        logger.info("Found existing audio dataset at %s (%d files).", audio_dir, len(list(audio_dir.glob("*.wav"))))
        return

    logger.info("Audio dataset not found locally. Downloading from Hugging Face Hub (%s)...", HF_DATASET_REPO)
    from huggingface_hub import hf_hub_download
    import tarfile

    tar_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="e2b_audio_dataset.tar.gz",
        repo_type="dataset",
        token=hf_token,
    )
    logger.info("Extracting %s into %s...", tar_path, REPO_ROOT)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(REPO_ROOT))
    logger.info("Extracted %d audio files successfully.", len(list(audio_dir.glob("*.wav"))))


def build_sample(
    processor: AutoProcessor,
    audio_path: str,
    prompt: str,
    target_json: dict,
) -> dict[str, torch.Tensor]:
    """Pre-process a single multimodal audio+text sample with masked labels."""
    p = Path(audio_path)
    if not p.is_file():
        p = REPO_ROOT / audio_path
    if not p.is_file() and ".data" in str(audio_path):
        rel_data = str(audio_path).split(".data/")[-1]
        p = REPO_ROOT / ".data" / rel_data
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path} (resolved to {p})")
    audio_data, _ = librosa.load(str(p), sr=16000)
    target_str = json.dumps(target_json, ensure_ascii=False)

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
            "content": [{"type": "text", "text": target_str}],
        },
    ]

    full_text = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(text=full_text, audio=audio_data, return_tensors="pt", sampling_rate=16000)

    # Compute prompt length by processing prompt part alone with same audio
    model_turn_marker = "<|turn>model\n"
    prompt_part = full_text.split(model_turn_marker)[0] + model_turn_marker
    prompt_inputs = processor(text=prompt_part, audio=audio_data, return_tensors="pt", sampling_rate=16000)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Create labels: mask prompt tokens with -100
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    sample = {k: v.squeeze(0) for k, v in inputs.items()}
    sample["labels"] = labels.squeeze(0)

    return sample


def load_and_preprocess_dataset(
    processor: AutoProcessor,
    file_path: Path,
    desc: str,
) -> list[dict[str, torch.Tensor]]:
    """Load and pre-encode dataset items into memory."""
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    logger.info("Pre-processing %d %s samples...", len(items), desc)
    dataset = []
    for idx, item in enumerate(items):
        try:
            t_sample = build_sample(
                processor,
                audio_path=item["audio_path"],
                prompt=item["prompt"],
                target_json=item["target_json"],
            )
            dataset.append(t_sample)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(items):
                logger.info("  -> Encoded %d/%d %s samples", idx + 1, len(items), desc)
        except Exception as exc:
            logger.error("Failed encoding sample %s: %s", item.get("audio_path"), exc)

    logger.info("Successfully encoded %d %s samples into memory.", len(dataset), desc)
    return dataset


def evaluate(
    model: torch.nn.Module,
    val_dataset: list[dict[str, torch.Tensor]],
) -> float:
    """Compute average evaluation loss across validation set."""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for sample in val_dataset:
            batch = {
                k: v.unsqueeze(0).to(DEVICE)
                for k, v in sample.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                total_val_loss += outputs.loss.item()
                del outputs
            del batch
    torch.cuda.empty_cache()
    return total_val_loss / max(1, len(val_dataset))


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    wandb_key = os.getenv("WANDB_API_KEY")

    if not hf_token:
        logger.warning("HF_TOKEN not found in environment!")
    if wandb_key:
        wandb.login(key=wandb_key)
        logger.info("W&B login successful.")
    else:
        logger.warning("WANDB_API_KEY not found in environment; running W&B offline.")
        os.environ["WANDB_MODE"] = "offline"

    # 1. Initialize W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"gemma4-e2b-lora-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "base_model": MODEL_ID,
            "target_repo": HF_HUB_REPO,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_epochs": NUM_EPOCHS,
            "accum_steps": ACCUM_STEPS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "device": DEVICE,
        },
    )
    logger.info("W&B run initialized: %s", wandb.run.url)

    # 2. Load Processor
    logger.info("Loading AutoProcessor for %s...", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

    # 3. Pre-process datasets into memory
    ensure_dataset(hf_token)
    train_data = load_and_preprocess_dataset(processor, TRAIN_JSONL, "train")
    val_data = load_and_preprocess_dataset(processor, VAL_JSONL, "val")

    # 4. Load Base Model with 4-bit QLoRA
    logger.info("Loading base model %s with 4-bit QLoRA...", MODEL_ID)
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
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Optimization: Freeze and detach audio tower to prevent activation memory accumulation during backward
    orig_get_audio_features = model.model.get_audio_features
    def get_audio_features_detached(*args, **kwargs):
        with torch.no_grad():
            out = orig_get_audio_features(*args, **kwargs)
            out.pooler_output = out.pooler_output.detach()
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                out.last_hidden_state = out.last_hidden_state.detach()
            return out
    model.model.get_audio_features = get_audio_features_detached

    # 5. Attach PEFT LoRA
    logger.info("Configuring PEFT LoRA (r=%d, alpha=%d)...", LORA_R, LORA_ALPHA)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=r".*language_model.*(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)$",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_training_steps = (len(train_data) // ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = max(2, int(total_training_steps * 0.10))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    logger.info("Training schedule: %d total steps, %d warmup steps.", total_training_steps, warmup_steps)

    # Initial validation loss
    initial_val_loss = evaluate(model, val_data)
    logger.info("Initial Validation Loss (before fine-tuning): %.4f", initial_val_loss)
    wandb.log({"eval/loss": initial_val_loss, "epoch": 0})

    best_val_loss = initial_val_loss
    global_step = 0
    start_time = time.time()

    # 7. Training Loop
    logger.info("Starting training across %d epochs...", NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0
        random.shuffle(train_data)

        optimizer.zero_grad()
        accum_loss = 0.0

        for step, sample in enumerate(train_data):
            batch = {
                k: v.unsqueeze(0).to(DEVICE)
                for k, v in sample.items()
            }

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / ACCUM_STEPS
                item_loss = outputs.loss.item()
                del outputs

            loss.backward()
            del batch, loss
            torch.cuda.empty_cache()

            accum_loss += item_loss
            epoch_train_loss += item_loss

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

                global_step += 1
                avg_step_loss = accum_loss / ACCUM_STEPS
                accum_loss = 0.0

                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": avg_step_loss,
                    "train/learning_rate": current_lr,
                    "train/step": global_step,
                    "epoch": epoch - 1 + ((step + 1) / len(train_data)),
                })

                if global_step % 5 == 0:
                    logger.info(
                        "Epoch %d/%d | Step %d/%d | Train Loss: %.4f | LR: %.2e",
                        epoch,
                        NUM_EPOCHS,
                        step + 1,
                        len(train_data),
                        avg_step_loss,
                        current_lr,
                    )

        avg_epoch_loss = epoch_train_loss / len(train_data)
        val_loss = evaluate(model, val_data)
        logger.info(
            "=== Epoch %d Complete | Avg Train Loss: %.4f | Val Loss: %.4f ===",
            epoch,
            avg_epoch_loss,
            val_loss,
        )
        wandb.log({
            "eval/loss": val_loss,
            "train/epoch_loss": avg_epoch_loss,
            "epoch": epoch,
        })

        # Save checkpoint if improved or final
        if val_loss <= best_val_loss or epoch == NUM_EPOCHS:
            best_val_loss = min(val_loss, best_val_loss)
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Saving best adapter checkpoint to %s (Val Loss: %.4f)...", CHECKPOINT_DIR, val_loss)
            model.save_pretrained(str(CHECKPOINT_DIR))
            processor.save_pretrained(str(CHECKPOINT_DIR))

    elapsed_mins = (time.time() - start_time) / 60.0
    logger.info("Training complete in %.2f minutes! Best Val Loss: %.4f", elapsed_mins, best_val_loss)

    # 8. Push to Hugging Face Hub if token is available
    if hf_token:
        try:
            logger.info("Pushing adapter checkpoint to Hugging Face Hub (%s)...", HF_HUB_REPO)
            model.push_to_hub(HF_HUB_REPO, token=hf_token, private=True)
            processor.push_to_hub(HF_HUB_REPO, token=hf_token, private=True)
            logger.info("Successfully pushed adapter to https://huggingface.co/%s", HF_HUB_REPO)
        except Exception as exc:
            logger.error("Failed to push adapter to HF Hub: %s", exc)

    wandb.finish()


if __name__ == "__main__":
    main()
