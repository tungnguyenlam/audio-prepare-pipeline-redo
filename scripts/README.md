# Scripts Directory Index

This directory contains executable runner scripts, training pipelines, ingestion utilities, and benchmark tools for the audio preparation pipeline.

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
| [`export_results_csv.py`](export_results_csv.py) | Exports verifier benchmark results and latency stats to CSV format. | `python scripts/export_results_csv.py` |

---

## 4. Distillation & Model Fine-Tuning

| Script | Purpose | Usage |
| [`generate_e2b_distillation_dataset.py`](generate_e2b_distillation_dataset.py) | **Active Pipeline:** Ingests crawled challenging channels, separates vocals via Mel-Band RoFormer on GPU, segments with DiariZen & Pyannote, applies valley splitting & boundary mitigation, audits with Gemini 3.8 Flash (MEDIUM), and builds `train_e2b.jsonl`. | `python scripts/generate_e2b_distillation_dataset.py` |
| [`train_gemma4_e2b_lora.py`](train_gemma4_e2b_lora.py) | **Active Student Trainer:** 4-bit QLoRA fine-tuning for `google/gemma-4-E2B-it` with frozen audio tower, training on multi-factor schema (`speaker_purity`, `word_completeness`, `audio_quality`). Logs to W&B. | `python scripts/train_gemma4_e2b_lora.py` |
| [`train_gemma4_e4b_lora.py`](train_gemma4_e4b_lora.py) | 4-bit QLoRA training pipeline for `google/gemma-4-E4B-it`. | `python scripts/train_gemma4_e4b_lora.py` |
| [`prepare_combined_dataset.py`](prepare_combined_dataset.py) | Merges multi-source distillation jsonl files, deduplicates, and splits into train/val sets. | `python scripts/prepare_combined_dataset.py` |
| [`generate_distillation_dataset.py`](generate_distillation_dataset.py) | Generates teacher labels from Gemini 3.8 Flash for candidate audio slices. | `python scripts/generate_distillation_dataset.py` |
| [`generate_haveasip_distillation_data.py`](generate_haveasip_distillation_data.py) | Generates synthetic boundary-clipped and tail-intrusion test pairs. | `python scripts/generate_haveasip_distillation_data.py` |
| [`generate_extended_distillation_data.py`](generate_extended_distillation_data.py) | Generates extended hard negative/positive pairs for verifier training. | `python scripts/generate_extended_distillation_data.py` |
| [`push_dataset_to_hf.py`](push_dataset_to_hf.py) | Packages and pushes labeled audio verification datasets to Hugging Face Hub. | `python scripts/push_dataset_to_hf.py` |
| [`evaluate_finetuned_verifier.py`](evaluate_finetuned_verifier.py) | Evaluates fine-tuned LoRA student against Gemini 3.8 Flash teacher predictions. | `python scripts/evaluate_finetuned_verifier.py` |
| [`run_distillation_pipeline.sh`](run_distillation_pipeline.sh) | End-to-end bash orchestrator for distillation dataset building and training. | `./scripts/run_distillation_pipeline.sh` |

---

## 5. Verifier Comparisons & Teacher Benchmarks

| Script | Purpose |
| :--- | :--- |
| [`compare_gemini_reasoning_levels.py`](compare_gemini_reasoning_levels.py) | Evaluates Gemini 3.8 Flash across reasoning budgets (`NONE`, `LOW`, `MEDIUM`, `HIGH`) to verify boundary detection sensitivity. |
| [`compare_verifiers_khanhvy.py`](compare_verifiers_khanhvy.py) | Multi-verifier benchmark on Khanh Vy vlog cuts (VibeVoice, Gemma 4, Gemini). |
| [`evaluate_gemma4_12b_khanhvy.py`](evaluate_gemma4_12b_khanhvy.py) | Evaluates unquantized `google/gemma-4-12B-it` direct-audio reasoning. |
| [`evaluate_gemma4_12b_q6_khanhvy.py`](evaluate_gemma4_12b_q6_khanhvy.py) | Evaluates llama.cpp GGUF Q6 quantized Gemma 4 12B. |
| [`evaluate_gemma4_e4b_q8_khanhvy.py`](evaluate_gemma4_e4b_q8_khanhvy.py) | Evaluates llama.cpp GGUF Q8 quantized Gemma 4 E4B. |

---

## 6. Archive (`scripts/archive/`)

Exploratory checks, hardware compatibility tests, and diagnostic scripts used during environment setup and debugging are archived under [`scripts/archive/`](archive/):
- **Hardware & Environment Checks:** `check_vram.py`, `check_unsloth_studio_env.py`, `check_unsloth_audio_support.py`, `check_keys.py`, `check_wandb.py`, `check_status.py`.
- **Model Architecture Checks:** `check_gemma4_arch.py`, `check_gemma4_support.py`, `check_e4b_size.py`, `check_hf_e4b_repo.py`, `check_hf_user.py`, `inspect_e4b_files.py`.
- **Smoke Tests & Verification:** `test_audio_processing.py`, `test_bnb_qlora.py`, `test_device_map.py`, `test_eval_load.py`, `test_gemma4_audio_forward.py`, `test_gemma4_lora_setup.py`, `test_gemma4_processor.py`, `test_gemma_vi_2048.py`, `test_label_masking.py`, `test_mel_roformer.py`, `test_module_map.py`, `test_peft_clippable.py`, `test_peft_gemma4.py`, `test_processor_audio.py`, `test_turn1_12b_q6.py`, `test_turn1_e4b_q8.py`, `test_unsloth_e4b_loader.py`, `test_vietnamese_prompt.py`, `diagnose_gemma_vi.py`, `download_base_model.py`, `download_less_quant_models.py`, `load_model_unsloth.py`, `unload_unsloth.py`.
