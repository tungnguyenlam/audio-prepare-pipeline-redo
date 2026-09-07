# Scripts Directory Index

This directory contains production-ready runner scripts, general CLIs, and utilities for the audio preparation pipeline.

All tools follow the repository engineering ideology: **reusable components**, **parameterization via command-line flags (`--flags`)**, and minimal one-off duplication.

---

## 1. Web Applications & Services

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| [`start_web.py`](start_web.py) / [`start_web.sh`](start_web.sh) | Primary entrypoint for the shared aiohttp server serving SonicStudio (`/studio/`) and SonicPipeline (`/pipeline/`) on port 8765. | `./scripts/start_web.sh [port] [host]` |
| [`start_studio.py`](start_studio.py) / [`start_studio.sh`](start_studio.sh) | Compatibility alias launching the backend and pointing users to SonicStudio. | `./scripts/start_studio.sh` |
| [`start_pipeline.py`](start_pipeline.py) / [`start_pipeline.sh`](start_pipeline.sh) | Compatibility alias launching the backend and pointing users to SonicPipeline. | `./scripts/start_pipeline.sh` |

---

## 2. Environment Setup & Synchronization

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| [`setup_worker_envs.sh`](setup_worker_envs.sh) | Installs and shims isolated worker environments: `.venv-diarizen`, `.venv-3dspeaker`, `.venv-sortformer`, and `.venv-vibevoice`. | `./scripts/setup_worker_envs.sh` |
| [`sync/`](sync/) | Shell utilities to rsync code and data between local development machine (`VF-TUNGNL5-L`) and model GPU server (`vsf-242`). | `./scripts/sync/code_to_server.sh` |

---

## 3. Data Ingestion & Clean Benchmarks

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| [`crawl_channels.py`](crawl_channels.py) | YouTube crawler ingesting long-form videos from channels (`@TRANTHANHTOWN`, `@KhánhVyOFFICIAL`), normalizing to 16kHz mono WAV. | `python scripts/crawl_channels.py --max-videos 3` |
| [`run_clean_pipeline_benchmark.py`](run_clean_pipeline_benchmark.py) | **Core Benchmark:** Runs Mel-Band RoFormer vocal separation $\to$ Diarizers (Pyannote Comm-1, DiariZen Large, 3D-Speaker) $\to$ Intelligent valley splitting $[2.0\text{s}, 15.0\text{s}]$ $\to$ Zero-contamination boundary mitigation $\to$ Gemini 3.8 Flash (`thinkingLevel="MEDIUM"`) multi-factor acoustic audit. | `python scripts/run_clean_pipeline_benchmark.py` |
| [`target_speaker.py`](target_speaker.py) | Enrolls target speaker voiceprints (ResNet34 / 3D-Speaker) and filters candidate segments by cosine similarity. | `python scripts/target_speaker.py score --audio input.wav` |

---

## 4. Generalized Distillation, Training & Evaluation Tools

These general tools replace legacy single-purpose scripts. All behaviors are configured via CLI flags.

### 4.1. Student Model Fine-Tuning: [`train_verifier.py`](train_verifier.py)
Unified trainer for multimodal speech verifiers using LoRA distillation from Gemini teacher annotations. Backed by modular components in [`src/diarization/verifier_training.py`](../src/diarization/verifier_training.py).

- **Hardware Agnostic:** Automatically handles CUDA GPU (4-bit NF4 QLoRA via `bitsandbytes`) or CPU fallback (`bfloat16` with configurable `--cpu-threads`).
- **Hub Integration:** Automatically pulls dataset tarball from Hugging Face Hub if missing locally, logs to Weights & Biases, and pushes checkpoints to Hugging Face Hub.
- **Key Flags:**
  ```bash
  # Fine-tune Gemma 4 E2B on GPU (default)
  python scripts/train_verifier.py --model-id google/gemma-4-E2B-it --epochs 3 --lr 2e-4

  # Fine-tune Gemma 4 E4B on CPU
  python scripts/train_verifier.py --model-id google/gemma-4-E4B-it --device cpu --cpu-threads 16 --quantization none
  ```

### 4.2. Acoustic Verification & Model Benchmark: [`evaluate_verifier.py`](evaluate_verifier.py)
Unified evaluator supporting Gemini API models and local Hugging Face / LoRA models on JSONL datasets, directories of WAVs, or experiment turn files.

- **Key Flags:**
  ```bash
  # Evaluate Gemini 3.8 Flash on Khanh Vy cuts
  python scripts/evaluate_verifier.py --backend gemini --model gemini-3.8-flash --reasoning-effort medium --input .data/experiment_khanhvy/cuts/ --output-report report.md --export-csv results.csv

  # Evaluate fine-tuned local LoRA adapter on validation split
  python scripts/evaluate_verifier.py --backend hf_local --model google/gemma-4-E2B-it --adapter-path .data/distillation/checkpoints_e2b/best_adapter --input .data/distillation/val_e2b.jsonl
  ```

### 4.3. Distillation Dataset Pipeline: [`build_distillation_dataset.py`](build_distillation_dataset.py)
Unified dataset builder with subcommands for the entire distillation lifecycle:
- **`annotate`:** Query Gemini 3.8 Flash teacher across directories of audio turns.
- **`balance`:** Stratify pass/reject ratios (e.g., 60% pass / 40% reject) and split into `train.jsonl` / `val.jsonl`.
- **`package`:** Compress audio files into `tar.gz` and optionally upload to Hugging Face Hub dataset repository.
- **Example Usage:**
  ```bash
  # 1. Annotate raw cuts
  python scripts/build_distillation_dataset.py annotate --audio-dirs .data/clean_benchmark_khanhvy/cuts/ --output-file .data/distillation/annotated.jsonl

  # 2. Balance dataset (60/40 ratio)
  python scripts/build_distillation_dataset.py balance --input-files .data/distillation/annotated.jsonl --pass-ratio 0.60

  # 3. Package and push to Hub
  python scripts/build_distillation_dataset.py package --audio-dir .data/distillation_e2b/audio --hf-repo tungnguyenlam/gemma-4-e2b-acoustic-verifier-data
  ```

### 4.4. One-Click External Runner: [`run_train_4090.sh`](../run_train_4090.sh)
Root wrapper script for self-contained execution on remote/standalone GPU machines (e.g., RTX 4090). Passes all trailing arguments directly to `scripts/train_verifier.py`.

---

## 5. Archive (`scripts/archive/`)

One-off exploratory checks, ad-hoc model probes, and legacy single-model scripts are archived under [`scripts/archive/`](archive/):
- **Archived Single-Purpose Scripts:** `train_gemma4_e2b_lora.py`, `train_gemma4_e4b_lora.py`, `evaluate_gemini_35_flash_lite.py`, `evaluate_gemma4_12b_khanhvy.py`, `evaluate_gemma4_12b_q6_khanhvy.py`, `evaluate_gemma4_e4b_q8_khanhvy.py`, `evaluate_finetuned_verifier.py`, `compare_verifiers_khanhvy.py`, `compare_gemini_reasoning_levels.py`, `export_results_csv.py`, `generate_distillation_dataset.py`, `generate_e2b_distillation_dataset.py`, `generate_extended_distillation_data.py`, `generate_haveasip_distillation_data.py`, `prepare_balanced_e2b_dataset.py`, `prepare_combined_dataset.py`, `push_dataset_to_hf.py`.
- **Diagnostic Probes & Tests:** Hardware tests, vllm/unsloth loaders, and intermediate smoke tests.
