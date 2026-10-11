# Research Repository: Speech Quality Evaluation, Audio-LLM Distillation, and Zero-Contamination TTS Curation

This repository indexes primary literature downloaded and analyzed to support the research plan in [`GRAND_PLAN.md`](GRAND_PLAN.md): a controlled benchmark of pipeline-induced curation defects and an audit of audio-LLM judges for Vietnamese TTS data.

---

## 1. Downloaded Papers & Extracted Corpus

All papers are stored in [`research/papers/`](papers/) as full PDFs and extracted to structured text in [`research/notes/`](notes/).

| Paper Title | Authors / Venue | arXiv / DOI | Focus Area |
|---|---|---|---|
| **Audio Large Language Models Can Be Descriptive Speech Quality Evaluators** (ALLD) | Chen et al. (NTU, NVIDIA, Tsinghua, JHU) — **ICLR 2025** | [arXiv:2501.17202](papers/2501.17202_ALLD.pdf) | DPO-style token-level distillation from a text LLM (LLaMA-3.1-70B) given human-rated meta information; descriptive MOS/A-B quality evaluation. |
| **SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation** | Wang et al. (Nankai, Microsoft) — **2025/2026** | [arXiv:2510.14664](papers/2510.14664_SpeechLLM_as_Judges.pdf) | Speech-quality judge for assessment, comparison, improvement suggestions and deepfake detection via CoT + GRPO (~43 A100 GPU-hours). |
| **QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning** | Wang et al. — **2025** | [arXiv:2503.20290](papers/2503.20290_QualiSpeech.pdf) | 11 low-level quality aspects with natural-language descriptions; open auditory LLMs show strong numerical bias (e.g. 66–86 % of scores on one value). Training data includes simulated noise and NISQA distortions. |
| **Zero-Shot Text-to-Speech for Vietnamese** (PhoAudiobook) | Vu, Nguyen, Nguyen (Movian AI) — **ACL 2025** | [ACL Anthology: 2025.acl-short.81](papers/2506.01322_PhoAudiobook_ACL2025.pdf) | 941 h audiobook TTS dataset; 23K h → 1,400 h via Demucs, Whisper/PhoWhisper exact-match and multi-speaker filters, then a 4 h/speaker cap; benchmarks VALL-E, VoiceCraft, XTTS-v2. |
| **Vietnamese Automatic Speech Recognition: A Revisit** (PhoASR) | Vu, Nguyen, Nguyen (Qualcomm AI Research) — **Findings of EACL 2026** | [ACL Anthology: 2026.findings-eacl.345](papers/2026.findings-eacl.345_PhoASR.pdf) | Curated Vietnamese ASR corpus (PhoASR-3100h) with word-level timestamps; dual-ASR WER < 5 % filtering; fine-tuned whisper-small models used here as the ASR metric. |
| **Seed-TTS: A Family of High-Quality Versatile Speech Generation Models** | ByteDance Seed Team — **2024** | [arXiv:2406.02430](papers/2406.02430_SeedTTS.pdf) | Zero-shot TTS family; standard objective protocol (WER + speaker similarity). Does not study training-data defects. |
| **High-Fidelity Audio Compression with Improved RVQGAN** (DAC) | Kumar et al. (Descript) — **NeurIPS 2023** | [arXiv:2306.06546](papers/2306.06546_DAC_AudioCodec.pdf) | Neural audio codec (RVQGAN). Background only; makes no claim about training-data hygiene. |
| **VoiceCraft: Zero-Shot Speech Editing and In-Context Generation** | Peng et al. — **2024** | [arXiv:2403.16973](papers/2403.16973_VoiceCraft.pdf) | Token infilling and continuation; reports occasional long silences and dragged sounds in generation (not attributed to training-data clipping). |

---

## 2. Evidence status

Checked against the extracted text in `notes/` on 2026-10-11:

- **Supported:** open auditory LLMs collapse onto a few score values (QualiSpeech);
  LALM judges can be trained with CoT + GRPO at multi-GPU-day cost
  (SpeechLLM-as-Judges); PhoAudiobook's and PhoASR's curation relies on ASR
  agreement and speaker filters.
- **Not supported by these papers (removed from the plan):** a 95.9 % discard
  rate caused by filters (the last step is a speaker cap); codec or TTS corruption
  from sub-50 ms interruptions (DAC, Seed-TTS); boundary clipping causing VoiceCraft
  repetition; score compression into [3.8, 5.0] for SpeechLLM-as-Judges.
- **Own evidence** lives in [`../docs/experiments.md`](../docs/experiments.md): the
  Gemma 4 E2B LoRA student passed 288/288 gold clips despite balanced training data,
  and the best Flash-Lite prompt still missed 74 of 129 teacher rejects.
- **Missing papers to add:** Emilia / Emilia-Pipe, DNSMOS, pyannote overlapped-speech
  detection, Brouhaha; verify AHa-Bench and MISHAP-Bench before citing.
