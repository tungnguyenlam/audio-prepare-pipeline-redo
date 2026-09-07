#!/usr/bin/env python3
"""Unified CLI for training Gemma 4 LoRA speech verifiers via knowledge distillation.

Supports both CUDA (with 4-bit NF4 QLoRA) and CPU (bfloat16) execution, automated
dataset retrieval from Hugging Face Hub, W&B tracking, and Hugging Face Hub checkpoint pushing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import torch
import wandb

from src.diarization.verifier_training import (
    VerifierTrainer,
    ensure_hf_dataset,
    load_dataset_samples,
    load_trainable_verifier_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_verifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma 4 speech verifiers via LoRA distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model & LoRA options
    parser.add_argument("--model-id", type=str, default="google/gemma-4-E2B-it", help="Base Gemma 4 model on HF Hub")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--quantization", type=str, default="auto", choices=["auto", "4bit", "none"], help="Quantization mode")

    # Dataset paths & options
    parser.add_argument("--train-data", type=str, default=".data/distillation/train_e2b.jsonl", help="Train dataset JSONL")
    parser.add_argument("--val-data", type=str, default=".data/distillation/val_e2b.jsonl", help="Val dataset JSONL")
    parser.add_argument("--hf-dataset-repo", type=str, default="tungnguyenlam/gemma-4-e2b-acoustic-verifier-data", help="HF dataset repository")
    parser.add_argument("--dataset-archive", type=str, default="e2b_audio_dataset.tar.gz", help="Dataset tar.gz archive on Hub")
    parser.add_argument("--audio-target-dir", type=str, default=".data/distillation_e2b/audio", help="Local extracted audio directory")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")

    # Hardware & Performance
    parser.add_argument("--device", type=str, default="auto", help="Device to use ('auto', 'cuda:0', 'cpu')")
    parser.add_argument("--cpu-threads", type=int, default=16, help="Number of PyTorch CPU threads if running on CPU")

    # Output & Hub Integration
    parser.add_argument("--output-dir", type=str, default=".data/distillation/checkpoints_e2b/best_adapter", help="Local adapter checkpoint directory")
    parser.add_argument("--hf-repo", type=str, default="tungnguyenlam/gemma-4-e2b-acoustic-verifier", help="HF Hub repo to push adapter (optional)")
    parser.add_argument("--no-push-hub", action="store_true", help="Disable pushing to Hugging Face Hub")

    # Logging & Weights & Biases
    parser.add_argument("--wandb-project", type=str, default="gemma-4-distill-verifier", help="W&B project name")
    parser.add_argument("--wandb-mode", type=str, default="auto", choices=["auto", "online", "offline", "disabled"], help="W&B logging mode")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Hardware setup
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        torch.set_num_threads(args.cpu_threads)
        logger.info("Configured PyTorch CPU threads: %d", torch.get_num_threads())

    # HF Token setup
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment or .env! Base model download might fail if gated.")

    # 1. Ensure dataset exists
    audio_dir = REPO_ROOT / args.audio_target_dir
    ensure_hf_dataset(
        repo_id=args.hf_dataset_repo,
        archive_name=args.dataset_archive,
        target_dir=audio_dir,
        repo_root=REPO_ROOT,
        hf_token=hf_token,
    )

    # 2. Weights & Biases setup
    wandb_key = os.getenv("WANDB_API_KEY")
    if args.wandb_mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        wandb_run = None
    else:
        if args.wandb_mode == "online" or (args.wandb_mode == "auto" and wandb_key):
            if wandb_key:
                wandb.login(key=wandb_key)
            os.environ["WANDB_MODE"] = "online"
        else:
            os.environ["WANDB_MODE"] = "offline"

        run_name = f"{Path(args.model_id).name}-lora-r{args.lora_r}-{time.strftime('%Y%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        logger.info("W&B initialized in mode '%s'. Run: %s", os.environ["WANDB_MODE"], getattr(wandb_run, "url", "offline"))

    # 3. Load model and processor
    model, processor, actual_device = load_trainable_verifier_model(
        model_id=args.model_id,
        device=args.device,
        quantization=args.quantization,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        hf_token=hf_token,
    )

    # 4. Load datasets
    train_path = REPO_ROOT / args.train_data if not Path(args.train_data).is_file() else Path(args.train_data)
    val_path = REPO_ROOT / args.val_data if not Path(args.val_data).is_file() else Path(args.val_data)

    train_data = load_dataset_samples(processor, train_path, repo_root=REPO_ROOT, desc="train")
    val_data = load_dataset_samples(processor, val_path, repo_root=REPO_ROOT, desc="val")

    # 5. Initialize Trainer and run
    trainer = VerifierTrainer(
        model=model,
        processor=processor,
        device=actual_device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        accum_steps=args.accum_steps,
        max_grad_norm=args.max_grad_norm,
        checkpoint_dir=REPO_ROOT / args.output_dir,
        hf_repo=args.hf_repo if not args.no_push_hub else None,
        hf_token=hf_token,
    )

    best_loss = trainer.train(train_data, val_data, wandb_run=wandb_run)
    logger.info("Training completed successfully! Best validation loss: %.4f", best_loss)

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
