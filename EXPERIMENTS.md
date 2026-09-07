# Pipeline & Acoustic Verifier Experiment Log

This log records the complete trajectory of experiments, architectural decisions, model comparisons, and benchmark results in this repository.

---

## 1. Pipeline Evolution Overview

```text
Raw YouTube / Audio Input
  │
  ▼
[1. Vocal Separation: Mel-Band RoFormer]  ──> Strips beats, synths, and heavy accompaniment
  │
  ▼
[2. Multi-Diarizer Segmentation]          ──> Pyannote Comm-1 / DiariZen Large / 3D-Speaker
  │
  ▼
[3. Intelligent Valley Splitting]         ──> Clamps all turns strictly into [2.0s, 15.0s]
  │                                           Splits turns > 15s at silence valleys (<= -32 dBFS, >= 150ms)
  ▼
[4. Zero-Contamination Boundary Mitigation]──> Snaps boundaries to local silence floors
  │                                           (50ms lead-in, 60ms lead-out padding)
  ▼
[5. Multi-Factor Acoustic Teacher Audit]  ──> Gemini 3.8 Flash (thinkingLevel="MEDIUM")
  │                                           Evaluates: Speaker Purity, Word Completeness, Audio Quality
  ▼
[6. Student Model Distillation]           ──> Gemma 4 E2B LoRA (google/gemma-4-E2B-it)
```

---

## 2. Phase 1: 3-Way Diarization & Word Completeness Benchmark

**Test Slice:** 180.0-second real-world vlog audio (`source_slice_180s.wav`).  
**Acoustic Stem:** Vocal stem isolated via Mel-Band RoFormer (Kimberley Jensen checkpoint on AMD ROCm GPU).  
**Teacher Verifier:** Google Gemini 3.8 Flash (`thinkingLevel="MEDIUM"`).  
**Artifact Report:** [`.data/benchmark_v2/BENCHMARK_REPORT.md`](.data/benchmark_v2/BENCHMARK_REPORT.md)

### Benchmark Summary Table

| Diarizer Model | Cut Strategy | Duration Clamping | Word Incomplete (%) | Speaker Intrusion (%) | Audio Quality Clean (%) | Overall Pass Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiariZen Large** (`wavlm-large-s80-md-v2`) | `raw` | $[2.0\text{s}, 15.0\text{s}]$ | 37.5% | 50.0% | 0.0% | **0.0%** |
| **DiariZen Large** (`wavlm-large-s80-md-v2`) | `mitigated` | $[2.0\text{s}, 15.0\text{s}]$ | **25.0%** *(−12.5%)* | 50.0% | 0.0% | **0.0%** |
| **Pyannote Comm-1** (`speaker-diarization-community-1`) | `raw` | $[2.0\text{s}, 15.0\text{s}]$ | 75.0% | 25.0% | 37.5% | **0.0%** |
| **Pyannote Comm-1** (`speaker-diarization-community-1`) | `mitigated` | $[2.0\text{s}, 15.0\text{s}]$ | **62.5%** *(−12.5%)* | 50.0% | 25.0% | **0.0%** |
| **3D-Speaker** (`speech_campplus_sv_zh_en_16k`) | `raw` | $[2.0\text{s}, 15.0\text{s}]$ | 37.5% | 50.0% | 12.5% | **0.0%** |
| **3D-Speaker** (`speech_campplus_sv_zh_en_16k`) | `mitigated` | $[2.0\text{s}, 15.0\text{s}]$ | 50.0% | 62.5% | 37.5% | **0.0%** |

### Key Experimental Discoveries
1. **Effectiveness of Zero-Contamination Boundary Snapping:**
   - On **DiariZen Large**, boundary snapping to local acoustic energy valleys reduced word incompleteness from **37.5% down to 25.0%**.
   - On **Pyannote**, word incompleteness dropped from **75.0% down to 62.5%**.
   - Snapping boundaries to local silence floors ($\le -32\text{dBFS}$) preserves natural vowel decay and coda consonant closures (Vietnamese *-p, -t, -k, -m, -n, -ng*).
2. **Why Audio Quality Had 0% Pass Rate:**
   - Gemini 3.8 Flash strictly enforced the multi-factor gate (`pass` requires `pure` + `complete` + `studio_clean`).
   - **Residual Music Bleed:** In vlog sections ($0\text{s} - 25\text{s}$ and $130\text{s} - 140\text{s}$), acoustic guitar tracks had faint harmonic bleed or vocal chops remaining even after vocal separation.
   - **Reverberant Hall Environment:** In the middle section ($59\text{s} - 125\text{s}$), the audio shifted to an echoey auditorium with crowd cheering and PA reverberation, correctly rejected as `noisy_reverberant`.
3. **Model Characteristics:**
   - **DiariZen Large:** Best boundary precision and lowest word truncation.
   - **Pyannote Comm-1:** Lowest speaker intrusion on raw cuts (25.0%), but higher word clipping (62.5%–75.0%).
   - **3D-Speaker:** Rapid inference, but broader boundary expansions led to higher intrusion on mitigated cuts.

---

## 3. Phase 2: Teacher Verifier Reasoning & Metric Decomposition

### Why Deeper Reasoning (`MEDIUM`) Was Essential
In earlier experiments on 12 Khanh Vy vlog cuts:
- Gemini 3.8 Flash with `thinkingLevel="MEDIUM"` produced **6 decision flips** compared to unreasoned / low-reasoning verifiers.
- Shallow models accepted cuts where the final consonant closure had already begun to be chopped off, mistaking trailing silence for completion.
- `MEDIUM` reasoning verified full acoustic resonance decay into silence, ensuring high-fidelity TTS training data.

### Decomposition of `boundary_issue`
`boundary_issue` was originally used as an ambiguous catch-all field. It was replaced by three distinct orthogonal dimensions:
1. **`speaker_purity`**: `pure` | `secondary_speaker` | `overlapping_speech`
2. **`word_completeness`**: `complete` | `clipped_word_start` | `clipped_word_end`
3. **`audio_quality`**: `studio_clean` | `music_bleed` | `noisy_reverberant` | `distorted`

---

## 4. Phase 3: Crawling Challenging Real-World Audio

To stress-test separation, diarization, and boundary mitigation, audio was ingested from two demanding YouTube channels:
1. **`https://www.youtube.com/@TRANTHANHTOWN`**:
   - Characterized by rapid conversational banter, dramatic volume swings, film set acoustics, and laughing crowds.
   - Ingested: House Tour (`QBml8L3wS3Q`), Behind the Scenes #5 Pháo (`i0zYcXBjytE`), Behind the Scenes #5 Góc Nhược Suy (`j83rzAzRDAI`).
2. **`https://www.youtube.com/@KhánhVyOFFICIAL`**:
   - Characterized by rapid bilingual Vietnamese/English code-switching, vlog background music, and event hall PA sound.
   - Ingested: English Self-Study (`Oa-mVxGS4cw`), PGS Sister Award Vlog (`H0VpjeULCck`), Gen Z 20s Advice (`lfIbjICmfW0`).
- **Normalized Artifacts:** 6 tracks (4,817.3s / ~1.34 hours) saved to `.data/crawled/` at 16,000 Hz mono WAV, cataloged in `.data/crawled/crawled_manifest.json`.

---

## 5. Phase 4: Student Model Distillation (Gemma 4 E2B)

### Why Gemma 4 E2B?
- Previously tested Gemma 4 12B and E4B. E4B and 12B require heavy VRAM allocation, causing inference bottlenecks during real-time data ingestion.
- **Gemma 4 E2B** (`google/gemma-4-E2B-it`) offers ~2B parameters with the same native audio conformer architecture, fitting comfortably in local VRAM with lower inference latency.

### Training Architecture (`scripts/train_gemma4_e2b_lora.py`)
- **Base Model:** `google/gemma-4-E2B-it`.
- **Quantization:** 4-bit NormalFloat4 (NF4) with BitsAndBytes.
- **LoRA Targets:** Attention projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`) with $r=16, \alpha=32$.
- **Acoustic Optimization:** Vision tower offloaded to CPU; audio tower frozen and detached during backward passes to eliminate activation caching overhead.
- **Target Schema:** Predicts `speaker_purity`, `word_completeness`, and `audio_quality` matching Gemini 3.8 Flash teacher annotations.
- **Dataset Built:** 96 paired clips (`raw` vs `mitigated`) generated from crawled tracks via `scripts/generate_e2b_distillation_dataset.py`, split into `train_e2b.jsonl` (76 samples) and `val_e2b.jsonl` (20 samples).
- **Telemetry:** Live tracking on Weights & Biases (`WANDB_PROJECT=gemma-4-e2b-distill-verifier`).
- **Hub Repository:** `tungnguyenlam/gemma-4-e2b-acoustic-verifier`.

---

## 6. Phase 5: Gemini 3.5 Flash-Lite Benchmark on 31 Turns

**Test Suite:** 31 benchmark turns from Khanh Vy vlog.  
**Script:** [`scripts/evaluate_gemini_35_flash_lite.py`](scripts/evaluate_gemini_35_flash_lite.py).  
**Report:** [`.data/experiment_khanhvy/GEMINI_35_FLASH_LITE_REPORT.md`](.data/experiment_khanhvy/GEMINI_35_FLASH_LITE_REPORT.md).

### Results:
- **Total Evaluated:** 31 turns
- **Pass Rate:** 23 / 31 (74.2%)
- **Reject Rate:** 8 / 31 (25.8%)
- **Agreement with Gemini 3.8 Flash (MEDIUM reasoning):** 16 / 31 (51.6%)
- **Agreement with Gemma 4 E4B:** 21 / 31 (67.7%)
- **Average Latency:** 1.79s per clip

### Acoustic Analysis:
`gemini-3.5-flash-lite` operates with shallow acoustic reasoning: it detects overt overlapping dialogue and loud secondary voices (e.g. turns 11, 17, 18, 19, 20, 21, 22, 30), but frequently fails to catch subtler boundary clipping (cut-off coda consonants or tone contours) where `gemini-3.8-flash` (MEDIUM reasoning) correctly rejected candidates. This confirms why `gemini-3.8-flash` with reasoning is essential as our primary teacher model for high-fidelity data supervision.

---

## 7. Phase 6: Gemma 4 E2B LoRA Distillation on AMD GPU (ROCm 10.0)

**Model:** `google/gemma-4-E2B-it` (Audio-Language Model).  
**Dataset:** 220 samples (175 train, 45 validation) mined from real-world Tran Thanh and Khanh Vy vlog recordings with Gemini 3.8 Flash teacher annotations.  
**Hardware:** AMD Radeon RX 9060 XT (16 GB VRAM, RDNA 4 / `gfx1200`).  
**Runtime:** ROCm 10.0 / HIP, PyTorch `2.13.0+rocm10.0.0`, `torchvision==0.28.0+rocm10.0.0`.  
**Script:** [`scripts/train_verifier.py`](scripts/train_verifier.py).  
**W&B Dashboard:** [chess/gemma-4-distill-verifier](https://wandb.ai/chess/gemma-4-distill-verifier/runs/u3liwf2n).  
**Hub Adapter:** [tungnguyenlam/gemma-4-e2b-acoustic-verifier](https://huggingface.co/tungnguyenlam/gemma-4-e2b-acoustic-verifier).

### Hardware & Convergence Metrics:
- **Precision Mode:** Native `bfloat16` (`--quantization none`). Fits within 16 GB VRAM without requiring CUDA-only `bitsandbytes` kernels.
- **VRAM Utilization:** 16.26 GB peak / 16.38 GB (98% capacity utilized).
- **GPU Engine Throughput:** 100% active GPU utilization during forward and backward passes (~16 seconds per 20 optimizer steps; ~0.8s per sample with gradient accumulation).
- **Initial Zero-Shot Validation Loss:** `1.6548`
- **Final Epoch 3 Validation Loss:** `0.3437` *(79.2% loss reduction)*
- **Final Train Loss:** `0.2299`
- **Artifacts:**
  - Local Checkpoint: `.data/distillation/checkpoints_e2b/best_adapter/` (123 MB bundle: `adapter_model.safetensors`, `adapter_config.json`, tokenizers, and processor configs).
  - Pushed to Hugging Face Model Hub: `https://huggingface.co/tungnguyenlam/gemma-4-e2b-acoustic-verifier`.

