# Research Repository: Speech Quality Evaluation, Audio-LLM Distillation, and Zero-Contamination TTS Curation

This repository indexes primary literature downloaded and analyzed to support our research proposal on **Contrastive Hard-Negative Distillation for Edge Audio-LLM Quality Gatekeepers**.

---

## 1. Downloaded Papers & Extracted Corpus

All papers are stored in [`research/papers/`](papers/) as full PDFs and extracted to structured text in [`research/notes/`](notes/).

| Paper Title | Authors / Venue | arXiv / DOI | Focus Area |
|---|---|---|---|
| **Audio Large Language Models Can Be Descriptive Speech Quality Evaluators** (ALLD) | Chen et al. (NTU, NVIDIA, Tsinghua, JHU) — **ICLR 2025** | [arXiv:2501.17202](papers/2501.17202_ALLD.pdf) | Token-level LLM distillation for multi-dimensional speech quality assessment; solves text-only pretraining bias. |
| **SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation** | Wang et al. (Nankai, Microsoft) — **2025/2026** | [arXiv:2510.14664](papers/2510.14664_SpeechLLM_as_Judges.pdf) | Evaluates SQA, SQC, SQI, and Deepfake detection via Chain-of-Thought and GRPO reward optimization. |
| **QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning** | Wang et al. — **2025** | [arXiv:2503.20290](papers/2503.20290_QualiSpeech.pdf) | 11 low-level acoustic degradation dimensions (noise, clipping, discontinuity, coloration); benchmarks auditory LLM blindness. |
| **Zero-Shot Text-to-Speech for Vietnamese** (PhoAudiobook) | Vu, Nguyen, Nguyen (Movian AI) — **ACL 2025** | [ACL Anthology: 2025.acl-short.81](papers/2506.01322_PhoAudiobook_ACL2025.pdf) | 941h curated Vietnamese zero-shot TTS dataset; benchmarks VALL-E, VoiceCraft, XTTS-v2; documents 95% yield loss from crude filtering. |
| **Vietnamese Automatic Speech Recognition: A Revisit** (PhoASR) | Vu, Nguyen, Nguyen (Qualcomm AI Research) — **Findings of EACL 2026** | [ACL Anthology: 2026.findings-eacl.345](papers/2026.findings-eacl.345_PhoASR.pdf) | 3,000h curated Vietnamese corpus (`PhoASR-whisper-small`), achieving 8.2% WER and precise word-level timestamp alignment. |
| **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** | ByteDance Seed Team — **2024** | [arXiv:2406.02430](papers/2406.02430_SeedTTS.pdf) | In-context speech factorization, self-distillation, RL for speech generation, and zero-shot voice cloning. |
| **High-Fidelity Audio Compression with Improved RVQGAN** (DAC) | Kumar et al. (Descript) — **NeurIPS 2023** | [arXiv:2306.06546](papers/2306.06546_DAC_AudioCodec.pdf) | Neural audio codec establishing discrete acoustic tokenization; demonstrates extreme vulnerability to boundary phase artifacts. |
| **VoiceCraft: Zero-Shot Speech Editing and In-Context Generation** | Peng et al. — **2024** | [arXiv:2403.16973](papers/2403.16973_VoiceCraft.pdf) | Token infilling and continuation; shows how boundary clipping cascades into token dropping and repetition. |

---

## 2. Synthesis of Evidence for the Research Proposal

### Evidence A: Why Audio-LLM Students Suffer from "All-Pass" Collapse
1. **The Representation Gap (Chen et al., ICLR 2025; Wang et al., 2025)**:
   Pre-trained audio foundation encoders (e.g. Whisper, HuBERT) are optimized for invariant semantic recognition (ASR), actively throwing away micro-acoustic perturbations (such as 25ms boundary clipping or low-amplitude secondary speaker bleed).
2. **Distributional Imbalance**:
   In any unconstrained natural speech dataset, clean speech occupies $> 90\%$ of samples. Under standard Supervised Fine-Tuning (SFT), minimizing cross-entropy causes the student model to degenerate into an all-pass majority classifier, achieving low training loss while completely failing on rejection ($R > 90\%$, as reproduced in our benchmark `docs/experiments.md`).

### Evidence B: Why Crude Cascade Filtering Fails Data Curation
1. **The 95% Yield Destruction Problem (Vu et al., ACL 2025)**:
   In creating PhoAudiobook, filtering 23,000 hours of raw audio down to 941 hours discarded over **95.9%** of potential training data because cascade filters (Demucs + Whisper + PhoWhisper + wav2vec2-bartpho) use crude heuristic intersection.
2. **The Coda Cutting Dilemma**:
   VAD models treat unreleased stop closures (`-p`, `-t`, `-c`, `-ch`) as inter-turn silence, while expanding cuts into gaps injects secondary speaker bleed.

### Evidence C: The Solution Backed by Modern Representation Learning
- **Adversarial Hard-Negative Synthesis (Wang et al., 2025; ALLD 2025)**:
  Controllably corrupting clean speech with four counterfactual defects (boundary snip, low-SNR crosstalk, stem bleed, codec dropouts) forces the student model to learn the decision boundary between clean and defective speech.
- **Contrastive Quality Margin Distillation**:
  Penalizing the distance between clean anchors and synthetic hard negatives directly counteracts the semantic collapse of the audio encoder.
