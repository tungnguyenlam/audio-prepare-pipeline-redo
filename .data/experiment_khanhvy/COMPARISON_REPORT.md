# Vietnamese Acoustic Boundary & Speaker Purity Verification Benchmark

- **Primary Source Audio:** `VLOG_Tháp_tùng_chị_gái_nhận_hàm_PGS_Đằng_sau_các_thành_tựu_viral_là_gì_(ft.YouTube_Works_Awards)__H0VpjeULCck.wav` (Khánh Vy VLOG, 180s benchmark slice)
- **Total Candidate Turns:** 31 natural conversational speech cuts
- **Distillation Teacher:** Google Gemini 3.8 Flash (Direct Native Audio API)
- **Student Models:** Google Gemma 4 12B (Local ROCm / 4-bit) & Gemma 4 E4B (Local ROCm / 4-bit QLoRA)
- **Hugging Face Hub:**
  - **Adapter:** [tungnguyenlam/gemma-4-e4b-acoustic-verifier](https://huggingface.co/tungnguyenlam/gemma-4-e4b-acoustic-verifier)
  - **Dataset:** [tungnguyenlam/vietnamese-acoustic-boundary-verifier-data](https://huggingface.co/datasets/tungnguyenlam/vietnamese-acoustic-boundary-verifier-data)
- **Weights & Biases Run:** [chess/gemma-4-e4b-distill-verifier](https://wandb.ai/chess/gemma-4-e4b-distill-verifier) (Run ID: `4pi9e2i2`)

---

## 1. Executive Summary & Core Metrics

| Model / Configuration | Reasoning Level | Passed Turns | Rejected Turns | Pass Rate | Avg Latency | Deployment / Hardware |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Google Gemini 3.8 Flash** | `LOW` | 12 | 19 | 38.7% | 2.66s | Google Cloud API |
| **Google Gemini 3.8 Flash** | `MEDIUM` | 13 | 18 | 41.9% | 4.88s | Google Cloud API |
| **Gemma 4 12B** (Zero-Shot) | N/A | 24 | 7 | 77.4% | 14.92s | AMD Radeon RX 7900 GRE (12.87 GB VRAM) |
| **Gemma 4 E4B** (Zero-Shot) | N/A | 27 | 4 | 87.1% | 5.56s | AMD Radeon RX 7900 GRE (9.47 GB VRAM) |
| **Gemma 4 E4B** (Fine-Tuned LoRA) | N/A | 31 | 0 | 100.0% | 5.42s | AMD Radeon RX 7900 GRE (9.47 GB VRAM) |

### Key Consensus Numbers
- **Gemini Low vs Gemini Medium Agreement:** **25/31 (80.6%)** (6 critical flips)
- **Gemini Medium vs Gemma 4 12B Agreement:** **10/31 (32.3%)**
- **Gemini Medium vs Gemma 4 E4B (Base) Agreement:** **15/31 (48.4%)**
- **Full Consensus Across All Native Evaluators:** **8/31 (25.8%)**

---

## 2. Gemini 3.8 Flash Reasoning Level Audit (`LOW` vs `MEDIUM`)

Testing on identical 31 audio cuts with `thinkingLevel="MEDIUM"` (budget: ~1,024 to 2,048 reasoning tokens) revealed profound differences in acoustic comprehension compared to `LOW` reasoning. **6 out of 31 decisions (19.4%) flipped**.

### Summary of Flipped Decisions

| Turn ID | Duration | `LOW` Decision | `MEDIUM` Decision | Flipped Error Code | Phonetic Analysis & Justification |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `turn_003` | 0.92s | `pass` | **`reject`** | `+clipped_word_end` | **Micro-Coda Cut:** `LOW` overlooked the abrupt cut of the trailing tone contour. `MEDIUM` acoustic tracing detected the vowel ending prematurely without natural room acoustic reverb/decay. |
| `turn_009` | 4.42s | `reject` | **`pass`** | `-clipped_word_end` | **Tone Falloff vs Clipping:** `LOW` hallucinated a clipped coda on a natural Vietnamese glottal stop / tone drop (`nặng`/`sắc` closure). `MEDIUM` correctly distinguished natural phonetic closure into silence. (Gemma 4 12B and E4B both passed this turn). |
| `turn_013` | 1.23s | `reject` | **`pass`** | `-clipped_word_end` | **Short Utterance Coarticulation:** `LOW` falsely flagged the trailing word boundary as clipped. `MEDIUM` confirmed complete word boundaries and single-speaker purity. |
| `turn_015` | 3.99s | `reject` | **`pass`** | `-clipped_word_end` | **False Rejection Rectified:** `LOW` misclassified the natural acoustic pause. `MEDIUM` verified clean boundary decay without clipping or tail intrusions. |
| `turn_024` | 1.17s | `pass` | **`reject`** | `+secondary_speaker`, `+clipped_word_end` | **Subtle Speaker Intrusion Caught:** `LOW` missed a quiet background secondary voice and trailing syllabic cut. `MEDIUM` flagged both acoustic flaws. |
| `turn_030` | 1.68s | `pass` | **`reject`** | `+clipped_word_start` | **Initial Attack Truncation Caught:** `LOW` missed the clipped onset consonant. `MEDIUM` detected the missing initial plosive attack. |

> **Conclusion on Teacher Selection:** `MEDIUM` reasoning in Gemini 3.8 Flash is strictly necessary for distillation. `LOW` reasoning suffers from two symmetrical failure modes:
> 1. **False Positives (Hallucinated Clipping):** Mistaking Vietnamese glottal stops, falling tones (`dấu nặng`), and unreleased final stops (`-p`, `-t`, `-c`, `-ch`) for artificial cutoffs.
> 2. **False Negatives (Missed Defects):** Missing subtle initial consonant cutoffs (<50ms attack truncations) and low-amplitude secondary speaker intrusions.

---

## 3. Full 31-Turn Comparison Matrix

| Turn ID | Dur | Gemini `LOW` | Gemini `MEDIUM` | Gemma 4 12B | Gemma 4 E4B | Reason Summary (Gemini `MEDIUM`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `turn_001` | 1.75s | `reject` | `reject` | `pass` | `reject` | Audio cuts off abruptly mid-syllable at the very end. |
| `turn_002` | 3.08s | `pass` | `pass` | `error` | `pass` | Clean single speaker with natural onsets and decays. |
| `turn_003` | 0.92s | `pass` | **`reject`** | `pass` | `pass` | Final word cut off before natural tonal contour & decay. |
| `turn_004` | 1.87s | `pass` | `pass` | `pass` | `pass` | Single speaker, clear boundaries, natural silence decay. |
| `turn_005` | 9.01s | `reject` | `reject` | `pass` | `pass` | Audio cuts off abruptly at the end while mid-syllable. |
| `turn_006` | 1.05s | `pass` | `pass` | `error` | `pass` | Clean acoustic boundaries, no overlapping voices. |
| `turn_007` | 4.97s | `reject` | `reject` | `error` | `reject` | Secondary singing voice at start; abrupt cut on final word. |
| `turn_008` | 7.86s | `reject` | `reject` | `pass` | `pass` | First word truncated at onset, missing initial attack. |
| `turn_009` | 4.42s | `reject` | **`pass`** | `pass` | `pass` | Single speaker, clean onsets and offsets, free of clipping. |
| `turn_010` | 1.57s | `reject` | `reject` | `pass` | `pass` | Multiple voices; distinct secondary voice intruding at tail. |
| `turn_011` | 2.97s | `reject` | `reject` | `pass` | `pass` | Two distinct speakers: female voice followed by male voice. |
| `turn_012` | 2.93s | `pass` | `pass` | `pass` | `pass` | Natural onset, full acoustic coda and decay, single speaker. |
| `turn_013` | 1.23s | `reject` | **`pass`** | `pass` | `pass` | Complete words, natural boundaries, single speaker throughout. |
| `turn_014` | 15.07s | `pass` | `pass` | `pass` | `pass` | Clean single speaker, natural boundaries, no foreground overlap. |
| `turn_015` | 3.99s | `reject` | **`pass`** | `error` | `pass` | Defined word boundaries, no speech overlap or tail intrusion. |
| `turn_016` | 1.72s | `reject` | `reject` | `pass` | `pass` | Cuts off abruptly during vowel/coda before natural decay. |
| `turn_017` | 2.38s | `reject` | `reject` | `pass` | `pass` | Secondary voice audible in background following laughter. |
| `turn_018` | 1.73s | `reject` | `reject` | `pass` | `reject` | Starts with cut-off vocalization from secondary speaker. |
| `turn_019` | 1.63s | `reject` | `reject` | `pass` | `pass` | Multiple speakers audible with overlapping conversational speech. |
| `turn_020` | 2.51s | `reject` | `reject` | `pass` | `pass` | Two distinct speakers audible with overlapping speech. |
| `turn_021` | 1.12s | `pass` | `pass` | `error` | `pass` | Starts and ends cleanly with clear word boundaries. |
| `turn_022` | 3.31s | `reject` | `reject` | `error` | `pass` | Multiple background voices clearly audible and overlapping. |
| `turn_023` | 2.01s | `pass` | `pass` | `error` | `pass` | Clean acoustic boundaries, single speaker, no interference. |
| `turn_024` | 1.17s | `pass` | **`reject`** | `pass` | `pass` | Secondary background voice at start; final word cut abruptly. |
| `turn_025` | 1.23s | `reject` | `reject` | `pass` | `pass` | Background secondary voices; primary speaker cut mid-vowel. |
| `turn_026` | 9.48s | `pass` | `pass` | `pass` | `pass` | Single speaker throughout with natural acoustic boundaries. |
| `turn_027` | 3.27s | `reject` | `reject` | `pass` | `pass` | Final syllable cut off abruptly before vowel/tonal conclusion. |
| `turn_028` | 3.05s | `reject` | `reject` | `pass` | `reject` | Cut off abruptly while vocal fold vibration is active. |
| `turn_029` | 1.30s | `pass` | `pass` | `pass` | `pass` | Single clear speaker with intact word boundaries. |
| `turn_030` | 1.68s | `pass` | **`reject`** | `pass` | `pass` | Initial syllable abruptly cut off at onset without acoustic lead-in. |
| `turn_031` | 1.82s | `reject` | `reject` | `pass` | `pass` | Final syllable cut off before completion of tone and decay. |

---

## 4. Hardened Audio Duration Specification: `[2.0s, 15.0s]`

To ensure high-quality dataset construction and prevent student model degradation, the pipeline enforces strict duration bounds on all newly generated audio clips:

$$\mathbf{2.0\,\text{s}} \le \mathbf{\text{Duration}} \le \mathbf{15.0\,\text{s}}$$

### Rationale & Boundary Justification

1. **Lower Bound ($< 2.0\,\text{s}$ Rejected):**
   - Vietnamese is an isolating, tonal language where tonal contours (ngang, huyền, ngã, hỏi, sắc, nặng) require preceding and succeeding co-articulatory context.
   - Ultra-short segments (< 2.0s) often contain isolated syllables where natural glottal stops or pitch resets cannot be distinguished from artificial windowing cutoffs.
   - Preserves existing sub-2.0s clips in historical benchmark sets, but prohibits them for future dataset mining.

2. **Upper Bound ($> 15.0\,\text{s}$ Split / Bounded):**
   - Audio segments exceeding 15.0s introduce multiple compound clauses and conversational breathing pauses where sentence-splitting ambiguity arises.
   - Long clips cause quadratic memory growth in the multimodal audio conformer/transformer attention layers ($O(T^2)$), risking VRAM exhaustion on local hardware.
   - Sentences spanning $> 15.0$s are segmented at acoustic energy minima / silence intervals ($\le -35$ dBFS) before verification.

---

## 5. Local Student Distillation (Gemma 4 E4B QLoRA)

### Training Configuration
- **Base Model:** `google/gemma-4-e4b` (4-bit NF4 via BitsAndBytes)
- **Adapter Configuration:**
  - $r = 16$, $\alpha = 32$, $\text{dropout} = 0.05$
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **VRAM Optimizations on AMD ROCm (ROCm 6.2 / RX 7900 GRE):**
  - Vision tower offloaded to CPU: `model.vision_tower.to("cpu")`
  - Conformer audio features computed under `torch.no_grad()` to detach intermediate graphs.
  - Forward output logits explicitly deleted prior to `loss.backward()` to release the 256k-vocabulary tensor.
  - Total VRAM Consumption: **8.4 GB - 9.5 GB** (fits comfortably within 16GB VRAM limit).
- **W&B Loss Trajectory:**
  - Initial `train/loss`: $2.84$
  - Final `train/loss`: $0.0309$
  - Final `eval/loss`: $0.7987$ (3 epochs on 173 samples)

### Next-Phase Distillation Roadmap
1. **Teacher:** Annotate new corpus (Have A Sip #118, 220 samples) exclusively using **Gemini 3.8 Flash (`thinkingLevel="MEDIUM"`)**.
2. **Data Balancing:** Maintain 50/50 balance between clean utterances (`pass`) and synthetic boundary/speaker defects (`reject` with calibrated 80ms/180ms coda cuts, 90ms/220ms onset cuts, and secondary vocal intrusions).
3. **Hard Negative Mining:** Retrain Gemma 4 E4B LoRA on the combined dataset to eliminate greedy `pass` bias and impart precise glottal tone discrimination into the local student.
