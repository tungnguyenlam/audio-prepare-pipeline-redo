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
| [`sync/code_to_server.sh`](sync/code_to_server.sh) / `from` | Shell utilities to rsync code between local development machine (`VF-TUNGNL5-L`) and model GPU server (`vsf-242` / `10.148.21.12`). | `./scripts/sync/code_to_server.sh` |
| [`sync/data_to_server.sh`](sync/data_to_server.sh) / `from` | Shell utilities to rsync runtime `.data/` artifacts with the model server. | `./scripts/sync/data_to_server.sh` |
| [`sync/code_to_loi.sh`](sync/code_to_loi.sh) / `from` | Shell utilities to rsync code with `Host loi` (`loinh8@10.148.1.176`). | `./scripts/sync/code_to_loi.sh` |
| [`sync/data_to_loi.sh`](sync/data_to_loi.sh) / `from` | Shell utilities to rsync `.data/` artifacts with `Host loi` (`loinh8@10.148.1.176`). | `./scripts/sync/data_to_loi.sh` |
| [`sync/code_to_anhnct.sh`](sync/code_to_anhnct.sh) / `from` | Shell utilities to rsync code with `anhnct` (`10.148.21.113`). | `./scripts/sync/code_to_anhnct.sh` |

---

## 3. Data Ingestion & Clean Benchmarks

| Script | Purpose | Usage |
| :--- | :--- | :--- |
| [`crawl_channels.py`](crawl_channels.py) | YouTube crawler ingesting long-form videos from channels (`@TRANTHANHTOWN`, `@KhánhVyOFFICIAL`) or explicit `--urls`, skipping IDs already in the crawled manifest, normalizing to 16kHz mono WAV. | `python scripts/crawl_channels.py --urls 'https://www.youtube.com/watch?v=VIDEO_ID'` |
| [`run_clean_pipeline_benchmark.py`](run_clean_pipeline_benchmark.py) | **Core Benchmark:** Runs Mel-Band RoFormer vocal separation $\to$ Diarizers (Pyannote Comm-1, DiariZen Large, 3D-Speaker) $\to$ Intelligent valley splitting $[2.0\text{s}, 15.0\text{s}]$ $\to$ Zero-contamination boundary mitigation $\to$ Gemini 3.8 Flash (`thinkingLevel="MEDIUM"`) multi-factor acoustic audit. | `python scripts/run_clean_pipeline_benchmark.py` |
| [`target_speaker.py`](target_speaker.py) | Enrolls target speaker voiceprints (ResNet34 / 3D-Speaker) and filters candidate segments by cosine similarity. | `python scripts/target_speaker.py score --audio input.wav` |

---

## 4. Generalized Distillation, Training & Evaluation Tools

These general tools replace legacy single-purpose scripts. All behaviors are configured via CLI flags.

### Production-strategy audit: [`audit_tts_data.py`](audit_tts_data.py)

Inventories existing manifests without inference; records exact duplicate files,
missing audio, duration eligibility, unknown source lineage, and saved-reference
acceptance error. Creates an opaque-name review packet under `.data/` without
overwriting existing reviews. The user accepts Gemini 3.8 Flash MEDIUM as ground
truth; the `teacher` subcommand makes bounded, resumable API calls with all nine
quality dimensions, input/configuration hashes, responses, and token usage saved.
The legacy challenge set cannot establish a representative production risk rate.

```bash
uv run --no-sync python scripts/audit_tts_data.py prepare --output .data/tts_strategy/phase1_20260908
uv run --no-sync python scripts/audit_tts_data.py teacher --packet .data/tts_strategy/phase1_20260908 --limit 31
uv run --no-sync python scripts/audit_tts_data.py report --packet .data/tts_strategy/phase1_20260908 --labels .data/tts_strategy/phase1_20260908/gemini_medium/labels.jsonl --eligible-only
uv run --no-sync python scripts/audit_tts_data.py lineage --output .data/tts_strategy/lineage_20260908_v2
uv run --no-sync python scripts/audit_tts_data.py export --packet .data/tts_strategy/pool_20260908 --output .data/tts_strategy/pool_20260908/labeled_pool.jsonl
uv run --no-sync python scripts/audit_tts_data.py boundaries --packet .data/tts_strategy/phase1_20260908 --source .data/experiment_khanhvy/khanhvy_180s_slice.wav --output .data/tts_strategy/boundaries_20260908 --device cuda:0
uv run --no-sync python scripts/crawl_channels.py --urls 'https://www.youtube.com/watch?v=VIDEO_ID'
.venv-sortformer/bin/python scripts/audit_tts_data.py extract \
  --manifest .data/crawled/crawled_manifest.json --only-ids VIDEO_ID \
  --output .data/tts_strategy/extract_20260908 --device cuda:0
uv run --no-sync python scripts/audit_tts_data.py teacher --packet .data/tts_strategy/extract_20260908/review_packet --limit 75
uv run --no-sync python scripts/audit_tts_data.py export --packet .data/tts_strategy/extract_20260908/review_packet --output .data/tts_strategy/extract_20260908/labeled.jsonl
uv run --no-sync python scripts/audit_tts_data.py combine \
  --inputs .data/tts_strategy/pool_20260908/labeled_pool.jsonl .data/tts_strategy/extract_20260908/labeled.jsonl \
  --output .data/tts_strategy/pool_20260908_v2/labeled_pool.jsonl
```

`prepare` requires a new output directory. `teacher` caches each completed label,
requires `GEMINI_API_KEY` from the root `.env`, and sends audio to Google only when
explicitly invoked. No API calls occur in `prepare`, `report`, `lineage`, `export`,
`extract`, `combine`, or the locate stage of `boundaries`. `boundaries` locates each
packet cut as an exact PCM crop of `--source`, then calls
`align_and_lock_syllable_boundaries` and `smart_segment_speaker_turns` (2–15 s TTS
duration policy). Changed children are copied into `review_packet/` for a later
`teacher` run. `extract` composes `run_zero_contamination_pipeline` on already
ingested sources with the measured no-gap-expansion lock and 2–15 s segmentation,
then writes a teacher packet. Use `.venv-sortformer/bin/python` so PhoWhisper sees
the ROCm GPU; the project `.venv` currently ships CUDA wheels that report no GPU.
`--consensus` is off by default because `.venv-diarizen` is CPU torch on this host.
`crawl_channels.py --urls` skips IDs already in `.data/crawled/crawled_manifest.json`.
`combine` refuses reserved challenge rows and duplicate bytes across recordings.
`report` supports adjudicated human labels or the user-approved teacher labels and
never treats missing labels as clean. Its confidence bound assumes independent
sampling and is diagnostic only for this recording-dependent challenge set.

`export` joins completed MEDIUM labels to rows that already have verified
`recording_id` values. `balance` in `build_distillation_dataset.py` then splits
whole recordings and refuses to overwrite existing manifests.

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

  # Evaluate MiniCPM-o with custom prompt text or file
  python scripts/evaluate_verifier.py --backend hf_local --model openbmb/MiniCPM-o-4_5 --prompt-file prompts/strict_acoustic.txt --output-report report.md

  # Evaluate Kimi-Audio 7B Instruct (.venv-kimi setup via scripts/setup_kimi_env.sh)
  .venv-kimi/bin/python scripts/evaluate_verifier.py --backend hf_local --model moonshotai/Kimi-Audio-7B-Instruct --input .data/experiment_khanhvy/results.json --output-report report_kimi.md
  ```

### 4.3. Distillation Dataset Pipeline: [`build_distillation_dataset.py`](build_distillation_dataset.py)
Unified dataset builder with subcommands for the entire distillation lifecycle:
- **`annotate`:** Query Gemini 3.8 Flash teacher across directories of audio turns.
- **`balance`:** Split by verified `recording_id` first, then optionally class-balance only the training split. Validation recordings stay source-disjoint and are not class-balanced. Duplicate audio bytes and missing recording IDs are rejected. Existing output paths are never overwritten.
- **`package`:** Compress audio files into `tar.gz` and optionally upload to Hugging Face Hub dataset repository.
- **Example Usage:**
  ```bash
  # 1. Annotate raw cuts
  python scripts/build_distillation_dataset.py annotate --audio-dirs .data/clean_benchmark_khanhvy/cuts/ --output-file .data/distillation/annotated.jsonl

  # 2. Recording-disjoint split; class-balance training only
  python scripts/build_distillation_dataset.py balance --input-files .data/tts_strategy/pool_20260908/labeled_pool.jsonl --pass-ratio 0.50 --validation-recordings youtube:fwN5VT_QxkY youtube:Oa-mVxGS4cw --train-out .data/tts_strategy/pool_20260908/train_v2.jsonl --val-out .data/tts_strategy/pool_20260908/calibration_v2.jsonl

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
