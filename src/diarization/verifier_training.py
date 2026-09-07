"""Reusable training components for multimodal speech verifiers (Gemma 4 LoRA distillation)."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import librosa
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Gemma4ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

logger = logging.getLogger("verifier_training")


def resolve_audio_path(audio_path: str | Path, repo_root: Path) -> Path:
    """Resolve audio path across absolute, relative, or shared workspace locations."""
    p = Path(audio_path)
    if p.is_file():
        return p
    candidate = repo_root / audio_path
    if candidate.is_file():
        return candidate
    if ".data" in str(audio_path):
        rel_data = str(audio_path).split(".data/")[-1]
        candidate = repo_root / ".data" / rel_data
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio file not found: {audio_path} (resolved from {repo_root})")


def ensure_hf_dataset(
    repo_id: str,
    archive_name: str,
    target_dir: Path,
    repo_root: Path,
    min_files: int = 50,
    hf_token: str | None = None,
) -> None:
    """Ensure audio dataset is present locally, downloading and unpacking from HF Hub if needed."""
    if target_dir.is_dir() and len(list(target_dir.glob("*.wav"))) >= min_files:
        logger.info("Found existing dataset at %s (%d files).", target_dir, len(list(target_dir.glob("*.wav"))))
        return

    logger.info("Dataset not found locally. Downloading '%s' from HF Hub '%s'...", archive_name, repo_id)
    from huggingface_hub import hf_hub_download
    import tarfile

    tar_path = hf_hub_download(
        repo_id=repo_id,
        filename=archive_name,
        repo_type="dataset",
        token=hf_token,
    )
    logger.info("Extracting %s into %s...", tar_path, repo_root)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=str(repo_root))
    logger.info("Extracted %d audio files successfully.", len(list(target_dir.glob("*.wav"))))


def build_multimodal_sample(
    processor: AutoProcessor,
    audio_path: str | Path,
    prompt: str,
    target_json: dict[str, Any],
    repo_root: Path,
    sampling_rate: int = 16000,
) -> dict[str, torch.Tensor]:
    """Pre-process a single audio+text sample and mask prompt tokens in labels."""
    resolved_path = resolve_audio_path(audio_path, repo_root)
    audio_data, _ = librosa.load(str(resolved_path), sr=sampling_rate)
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
    inputs = processor(text=full_text, audio=audio_data, return_tensors="pt", sampling_rate=sampling_rate)

    # Compute prompt length by encoding prompt part alone
    model_turn_marker = "<|turn>model\n"
    prompt_part = full_text.split(model_turn_marker)[0] + model_turn_marker
    prompt_inputs = processor(text=prompt_part, audio=audio_data, return_tensors="pt", sampling_rate=sampling_rate)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # Create labels: mask prompt tokens with -100
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    sample = {k: v.squeeze(0) for k, v in inputs.items()}
    sample["labels"] = labels.squeeze(0)
    return sample


def load_dataset_samples(
    processor: AutoProcessor,
    file_path: Path,
    repo_root: Path,
    desc: str = "dataset",
) -> list[dict[str, torch.Tensor]]:
    """Load JSONL lines and encode into list of tensor samples."""
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
            sample = build_multimodal_sample(
                processor=processor,
                audio_path=item["audio_path"],
                prompt=item["prompt"],
                target_json=item["target_json"],
                repo_root=repo_root,
            )
            dataset.append(sample)
            if (idx + 1) % 15 == 0 or (idx + 1) == len(items):
                logger.info("  -> Encoded %d/%d %s samples", idx + 1, len(items), desc)
        except Exception as exc:
            logger.error("Failed encoding sample %s: %s", item.get("audio_path"), exc)

    logger.info("Successfully encoded %d %s samples into memory.", len(dataset), desc)
    return dataset


def load_trainable_verifier_model(
    model_id: str,
    device: str = "auto",
    quantization: str = "auto",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    hf_token: str | None = None,
) -> tuple[PeftModel, AutoProcessor, str]:
    """Load base Gemma 4 model, configure quantization/device, attach LoRA adapters."""
    # 1. Device resolution
    if device == "auto":
        actual_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        actual_device = device

    use_cuda = actual_device.startswith("cuda")
    logger.info("Configuring model on device: %s (CUDA: %s)", actual_device, use_cuda)

    # 2. Processor
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    # 3. Quantization configuration
    if quantization == "auto":
        quantization = "4bit" if use_cuda else "none"

    bnb_config = None
    device_map = None
    if use_cuda and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif not use_cuda:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    if use_cuda and quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["audio_tower", "vision_tower", "embed_audio", "embed_vision", "lm_head"],
        )
        device_map = {
            "model.vision_tower": "cpu",
            "model.embed_vision": "cpu",
            "model.audio_tower": actual_device,
            "model.embed_audio": actual_device,
            "model.language_model": actual_device,
            "lm_head": actual_device,
        }
    elif use_cuda and quantization == "none":
        device_map = "auto"
    else:
        # CPU loading
        device_map = "cpu"
        torch_dtype = torch.bfloat16

    logger.info("Loading base model %s (quantization=%s, dtype=%s)...", model_id, quantization, torch_dtype)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        token=hf_token,
    )

    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Optimization: Freeze and detach audio tower to prevent activation memory accumulation during backward pass
    orig_get_audio_features = model.model.get_audio_features
    def get_audio_features_detached(*args, **kwargs):
        with torch.no_grad():
            out = orig_get_audio_features(*args, **kwargs)
            out.pooler_output = out.pooler_output.detach()
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                out.last_hidden_state = out.last_hidden_state.detach()
            return out
    model.model.get_audio_features = get_audio_features_detached

    # 4. Attach PEFT LoRA
    logger.info("Attaching LoRA adapter (r=%d, alpha=%d)...", lora_r, lora_alpha)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=r".*language_model.*(q_proj|v_proj|k_proj|o_proj|gate_proj|up_proj|down_proj)$",
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, processor, actual_device


class VerifierTrainer:
    """End-to-end trainer for multimodal Gemma 4 acoustic verifiers."""

    def __init__(
        self,
        model: PeftModel,
        processor: AutoProcessor,
        device: str,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Path | str = Path(".data/distillation/checkpoints/best_adapter"),
        hf_repo: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.accum_steps = accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.hf_repo = hf_repo
        self.hf_token = hf_token

    def evaluate(self, val_dataset: list[dict[str, torch.Tensor]]) -> float:
        """Compute average validation loss across validation samples."""
        self.model.eval()
        total_val_loss = 0.0
        use_cuda = self.device.startswith("cuda")
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_cuda
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        )

        with torch.no_grad():
            for sample in val_dataset:
                batch = {k: v.unsqueeze(0).to(self.device) for k, v in sample.items()}
                with autocast_context:
                    outputs = self.model(**batch)
                    total_val_loss += outputs.loss.item()
                    del outputs
                del batch
        if use_cuda:
            torch.cuda.empty_cache()
        return total_val_loss / max(1, len(val_dataset))

    def train(
        self,
        train_dataset: list[dict[str, torch.Tensor]],
        val_dataset: list[dict[str, torch.Tensor]],
        wandb_run: Any = None,
    ) -> float:
        """Run training across specified epochs with cosine decay and validation checks."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)

        total_training_steps = (len(train_dataset) // self.accum_steps) * self.num_epochs
        warmup_steps = max(2, int(total_training_steps * 0.10))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        use_cuda = self.device.startswith("cuda")
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_cuda
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        )

        initial_val_loss = self.evaluate(val_dataset)
        logger.info("Initial Validation Loss (before fine-tuning): %.4f", initial_val_loss)
        if wandb_run:
            wandb_run.log({"eval/loss": initial_val_loss, "epoch": 0})

        best_val_loss = initial_val_loss
        global_step = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            epoch_train_loss = 0.0
            random.shuffle(train_dataset)
            optimizer.zero_grad()
            accum_loss = 0.0

            for step, sample in enumerate(train_dataset):
                batch = {k: v.unsqueeze(0).to(self.device) for k, v in sample.items()}
                with autocast_context:
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.accum_steps
                    item_loss = outputs.loss.item()
                    del outputs

                loss.backward()
                del batch, loss
                if use_cuda:
                    torch.cuda.empty_cache()

                accum_loss += item_loss
                epoch_train_loss += item_loss

                if (step + 1) % self.accum_steps == 0 or (step + 1) == len(train_dataset):
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if use_cuda:
                        torch.cuda.empty_cache()

                    global_step += 1
                    avg_step_loss = accum_loss / self.accum_steps
                    accum_loss = 0.0
                    current_lr = scheduler.get_last_lr()[0]

                    if wandb_run:
                        wandb_run.log({
                            "train/loss": avg_step_loss,
                            "train/learning_rate": current_lr,
                            "train/step": global_step,
                            "epoch": epoch - 1 + ((step + 1) / len(train_dataset)),
                        })

                    if global_step % 5 == 0:
                        logger.info(
                            "Epoch %d/%d | Step %d/%d | Train Loss: %.4f | LR: %.2e",
                            epoch,
                            self.num_epochs,
                            step + 1,
                            len(train_dataset),
                            avg_step_loss,
                            current_lr,
                        )

            avg_epoch_loss = epoch_train_loss / len(train_dataset)
            val_loss = self.evaluate(val_dataset)
            logger.info(
                "=== Epoch %d Complete | Avg Train Loss: %.4f | Val Loss: %.4f ===",
                epoch,
                avg_epoch_loss,
                val_loss,
            )
            if wandb_run:
                wandb_run.log({
                    "eval/loss": val_loss,
                    "train/epoch_loss": avg_epoch_loss,
                    "epoch": epoch,
                })

            if val_loss <= best_val_loss or epoch == self.num_epochs:
                best_val_loss = min(val_loss, best_val_loss)
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Saving best adapter checkpoint to %s (Val Loss: %.4f)...", self.checkpoint_dir, val_loss)
                self.model.save_pretrained(str(self.checkpoint_dir))
                self.processor.save_pretrained(str(self.checkpoint_dir))

        elapsed_mins = (time.time() - start_time) / 60.0
        logger.info("Training complete in %.2f minutes! Best Val Loss: %.4f", elapsed_mins, best_val_loss)

        if self.hf_repo and self.hf_token:
            try:
                logger.info("Pushing adapter checkpoint to Hugging Face Hub (%s)...", self.hf_repo)
                self.model.push_to_hub(self.hf_repo, token=self.hf_token, private=True)
                self.processor.push_to_hub(self.hf_repo, token=self.hf_token, private=True)
                logger.info("Successfully pushed adapter to https://huggingface.co/%s", self.hf_repo)
            except Exception as exc:
                logger.error("Failed to push adapter to HF Hub: %s", exc)

        return best_val_loss
