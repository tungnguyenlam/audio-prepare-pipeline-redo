# Speech Verifier Distillation & Evaluation — Session Handoff Context

> Historical session record, superseded for next actions by
> [TTS production strategy](TTS_PRODUCTION_STRATEGY.md) and its
> [execution log](TTS_STRATEGY_EXECUTION.md). Do not automatically resume the
> downloads/model runs below. Architecture explanations are hypotheses; the
> existing Khanh Vy labels omit audio quality and their generator configures LOW.
> A fresh explicit MEDIUM full-rubric audit is required. The user accepts Gemini
> 3.8 Flash MEDIUM as ground truth; separate human review is not required.

**Last Updated:** 2026-09-07 22:33 (Local Time)  
**Host Environment:** `tungnl5@VF-TUNGNL5-L` | AMD Radeon RX 9060 XT (16 GB VRAM, ROCm 10.0 / HIP)  
**Active Python Runtime:** `.venv/bin/python` (Python 3.13, PyTorch 2.13.0+rocm10.0.0, Transformers 5.16.1)

---

## 1. Executive Summary & Objective

The objective of this pipeline component is to distill commercial multimodal audio foundation models (**Gemini 3.8 Flash** with `medium` reasoning) into compact, open-weight on-premise acoustic verifiers (**Gemma 4 E2B**, **MiniCPM-o 4.5 INT4**, **Kimi-Audio-7B-Instruct**) to automate zero-contamination TTS dataset filtering without cloud API costs or latency.

The verifier evaluates 3 strict acoustic dimensions:
1. **Speaker Purity:** `pure` vs. `secondary_speaker` / `overlapping_speech`.
2. **Word Completeness:** `complete` vs. `clipped_word_start` / `clipped_word_end` (không lẹm âm/cắt từ).
3. **Audio Quality:** `studio_clean` vs. `music_bleed` / `noisy_reverberant` / `distorted`.

---

## 2. Completed Milestones & Current State

### A. Data Mining, Augmentation & Hub Datasets
- **Total Unique Annotated Turns:** **949 samples** (148 `pass`, 801 `reject` across 7 distinct failure codes) extracted from 8 full podcast/interview episodes (>2 hours of audio) and synthetic boundary shavings.
- **Master Balanced V3 Dataset:**
  - `train_v3.jsonl` (263 samples: 119 pass, 144 reject — 45% pass ratio)
  - `val_v3.jsonl` (65 samples: 29 pass, 36 reject)
  - `v3_audio_dataset.tar.gz` (71.8 MB containing all 328 audio cut WAV files)
- **Hugging Face Dataset Repo:** [`tungnguyenlam/gemma-4-e2b-acoustic-verifier-data`](https://huggingface.co/datasets/tungnguyenlam/gemma-4-e2b-acoustic-verifier-data)
  - Successfully synced and authenticated via `HF_TOKEN`.

### B. Gemma 4 E2B LoRA Distillation Runs
All fine-tuning was executed on the local AMD GPU (`cuda:0`) in native `bfloat16` with LoRA ($r=16, \alpha=32$):

| Run | Dataset | Epochs | Initial Val Loss | Best Val Loss | Loss Reduction | Checkpoint / Hub Adapter |
|---|---|---|---|---|---|---|
| **V1** | 120 basic cuts | 3 | 1.6548 | 0.3437 | 79.2% | `.data/distillation/checkpoints_e2b/best_adapter` |
| **V2** | + Synthetic boundary hard-negatives (312 samples) | 3 | 1.6164 | **0.2756** | 83.0% | `.data/distillation/checkpoints_e2b_v2/best_adapter` |
| **V3** | Master balanced crawl + boundary cuts (328 samples) | 3 | 1.6924 | **0.3285** | 80.6% | [tungnguyenlam/gemma-4-e2b-acoustic-verifier](https://huggingface.co/tungnguyenlam/gemma-4-e2b-acoustic-verifier) (W&B: `sm0ky3on`) |

### C. Benchmark on 31 Khanh Vy Test Cuts & Architectural Verdict
- **Results:** 31/31 cuts predicted as `pass` (Agreement with Gemini 3.8 Flash: **38.7%**, 12 True Passes, 19 False Passes, 0 False Rejects).
- **Technical Diagnosis:** The 31 test cuts are isolated vocal stems from Mel-Band RoFormer. A 2B causal audio LLM conformer treats 80-150ms trailing coda decay or quiet secondary bleed as acceptable natural speech, exhibiting a high inductive bias toward predicting `pass`.
- **Architectural Law (Hybrid Pipeline):**
  - **Stage 3 (Deterministic DSP):** Micro-Energy Valley Snapping ($\le -32\text{ dBFS}$) and PhoWhisper Forced Alignment Syllable Lock are strictly required to physically prevent boundary truncations.
  - **Stage 5 (Student Multimodal LLM):** Used specifically as the gatekeeper for co-host cross-talk, secondary vocal leakage, reverb, and music bleed.

---

## 3. Hardware & Runtime Constraints (AMD ROCm / RDNA 4)

1. **Memory Budget:**
   - Full fine-tuning of 2.3B params in bfloat16 requires $\ge 27.6\text{ GB}$ VRAM (Immediate OOM).
   - LoRA fine-tuning ($r=16$) consumes ~11.5 – 13.5 GB VRAM and fits cleanly within the 16 GB hardware limit.
2. **ROCm Attention Stability Rule:**
   - Do **NOT** export `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for Gemma 4 or multimodal LLM inference/training on RDNA 4 (`gfx1200`). It triggers `torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)`.
   - Default PyTorch SDPA / rocBLAS attention runs at 100% GPU core utilization without crashes.

---

## 4. Current State of MiniCPM-o 4.5 & Kimi-Audio-7B

### MiniCPM-o 4.5 INT4 (~11 GB)
- Target checkpoint: `ericleigh007/MiniCPM-o-4_5-BNB-Int4` (Pre-quantized BitsAndBytes NF4, 8.44 GB total weight size).
- Environment packages prepared: `minicpmo`, `onnx`, `hyperpyyaml`, `diffusers`, `bitsandbytes`, `hf_xet`.
- Background download launched: Background task `task-3903` is actively pulling shards to Hugging Face cache (`~/.cache/huggingface/hub/models--ericleigh007--MiniCPM-o-4_5-BNB-Int4/`).
- Network note: During international peak hours (21:00-23:30), Vietnam ISP upstream routes to CloudFront CDN run at ~200-500 kB/s. If left overnight or resumed during off-peak morning hours, the download will complete smoothly.

### Kimi-Audio-7B-Instruct (~18.2 GB)
- Target repository: `moonshotai/Kimi-Audio-7B-Instruct`.
- Architecture: Moonshot Kimia CausalLM (Qwen2 backbone) + embedded Whisper-large-v3 audio encoder.
- Inference can be executed via `evaluate_verifier.py --model moonshotai/Kimi-Audio-7B-Instruct --torch-dtype bfloat16` or with 4-bit quantization / `device_map="auto"`.

---

## 5. Resuming Tomorrow: Step-by-Step Instructions

When resuming this session, execute the following steps:

### Step 1: Check Download Status of MiniCPM-o 4.5 INT4
```bash
# Verify if all shards exist in HF cache
.venv/bin/python -c "
from huggingface_hub import try_to_load_from_cache
f1 = try_to_load_from_cache('ericleigh007/MiniCPM-o-4_5-BNB-Int4', 'model-00001-of-00002.safetensors')
f2 = try_to_load_from_cache('ericleigh007/MiniCPM-o-4_5-BNB-Int4', 'model-00002-of-00002.safetensors')
print('Shard 1 cached:', bool(f1))
print('Shard 2 cached:', bool(f2))
"
```
If not yet finished downloading, resume with native automatic retry:
```bash
.venv/bin/huggingface-cli download ericleigh007/MiniCPM-o-4_5-BNB-Int4
```

### Step 2: Evaluate MiniCPM-o 4.5 INT4 on 31 Khanh Vy Benchmark Cuts
```bash
.venv/bin/python scripts/evaluate_verifier.py \
  --backend hf_local \
  --model ericleigh007/MiniCPM-o-4_5-BNB-Int4 \
  --device cuda:0 \
  --input .data/experiment_khanhvy/results.json \
  --output-json .data/distillation/reports/minicpmo_int4_eval.json \
  --output-report .data/distillation/reports/minicpmo_int4_vs_gemini38.md \
  --export-csv .data/distillation/reports/minicpmo_int4_vs_gemini38.csv
```

### Step 3: Evaluate Kimi-Audio-7B-Instruct
```bash
.venv/bin/python scripts/evaluate_verifier.py \
  --backend hf_local \
  --model moonshotai/Kimi-Audio-7B-Instruct \
  --device auto \
  --input .data/experiment_khanhvy/results.json \
  --output-json .data/distillation/reports/kimi_audio_7b_eval.json \
  --output-report .data/distillation/reports/kimi_audio_7b_vs_gemini38.md \
  --export-csv .data/distillation/reports/kimi_audio_7b_vs_gemini38.csv
```

### Step 4: Update Documentation
Update `docs/04_zero_contamination_diarization.md` and comparison tables with the side-by-side agreement rates, failure detection sensitivities, and latency tradeoffs across all models:
- Gemini 3.8 Flash (Teacher Baseline)
- Gemini 3.5 Flash-Lite
- Gemma 4 E2B LoRA (Student V1/V2/V3)
- MiniCPM-o 4.5 INT4
- Kimi-Audio-7B-Instruct
