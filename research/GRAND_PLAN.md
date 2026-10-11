# Research Plan: Can Audio LLMs Hear Pipeline Cuts?

**Working title:** *Can Audio LLMs Hear Pipeline Cuts? A Controlled Benchmark of
Curation Defects for Vietnamese TTS Data*
**Target venue:** Interspeech 2027 (4 pages + references). The 2027 deadline is not
announced yet; past deadlines fell 12–25 February, so plan for **early February
2027**. ICASSP 2027 (16 Sep 2026) and SLT 2026 have passed. Fallback: ICASSP 2028
or a journal extension.
**Revised:** 2026-10-11. Supersedes the "Logit-Adjusted Contrastive Distillation"
draft, whose citations and premises did not survive checking (see §7).

## 1. Constraints this plan is designed around

| Resource | What we actually have | Consequence |
|---|---|---|
| GPU | 1× RX 9060 XT, 16 GB, ROCm 10.0 (`docs/hardware.md`); NVIDIA server `vsf-242` for separation/diarization | Inference, feature probes and small fine-tunes only. Gemma 4 E2B bf16 LoRA is proven here; 7B models need 4-bit. |
| Gemini | AI Studio **free** key, tight daily quota | Gemini is a *judge under test* on a stratified subset, not the source of ground truth. Trickle requests over days with `--continue`. |
| OpenRouter | Optional key | One or two extra commercial audio judges through the existing `scripts/s4-agent/verifier/endpoint.sh` (OpenAI-compatible `input_audio`). |
| Human time | The authors | ~400 real clips double-annotated (≈ 2–3 h listening each). Required for any real-data claim. |
| Clean Vietnamese audio | `.data/s1-download/thu-am-studio/` (~4 h studio recordings); PhoAudiobook test splits (manually verified, non-commercial) | In-domain clean anchors. Check speaker inventory and redistribution rights before releasing derived audio. |
| Prior measurements | `docs/experiments.md`: 288-clip gold set, E2B LoRA V1–V3, Flash-Lite prompt tuning, ViYT-Diar DER | Real baselines and motivation; no invented numbers. |

## 2. Motivation from our own data (not from literature)

1. **The student verifier never learned to reject.** Gemma 4 E2B LoRA V3, trained on
   a *balanced* crawl, passed 288/288 gold clips (129 leaks: music 61, secondary 37,
   clipped end 29). V2 already included synthetic boundary hard negatives. Class
   imbalance is therefore not the explanation; the open question is whether the
   defect information reaches the model at all.
2. **Cheap API judges miss subtle defects.** Gemini 3.5 Flash-Lite with the best
   prompt (V2 defect-first) still let 74 of 129 teacher rejects pass, mostly quiet
   music and exact word-edge clipping.
3. **Real defects are common.** 45 % of gold clips are rejects; yield ranges from
   0.9 accepted min/source h (music-backed vlogs) to 6.7 (studio interviews).
4. **ASR timestamps can't localize the defect.** PhoWhisper-small word edges overlap
   almost every clip boundary, and aligned word gaps contain few of the real pauses.
   Boundary errors must be detected acoustically.

These points give the central question: **at what severity do audio-LLM judges
start to notice the defects our pipeline introduces, and is that information
already present in their audio encoders?**

## 3. Contributions (scoped to 4 pages)

- **C1 — ViCut-Bench.** Controlled, parameterised curation defects on clean
  Vietnamese anchors, with exact severities, per-defect perceptual thresholds from
  a small listening calibration, and a human-annotated real subset.
- **C2 — Judge audit.** Detection-vs-severity curves and paired clean/defect
  consistency for API judges, local audio LLMs, and non-LLM signal baselines.
- **C3 — Where the information is lost.** Layer-wise linear probes on frozen
  audio encoders, compared with the same model's generated verdict. A probe head is
  also a cheap local gatekeeper.
- **C4 — Does it matter for TTS?** A dose-response experiment: one TTS recipe,
  fixed clean data, controlled injection of each defect type.

Out of scope for the paper: new loss functions, a 100 h crawl, and the 50 h
dataset release. These can be follow-ups once C1–C4 exist.

## 4. Experiments

### Phase 0 — Resource check (week 1)

- Measure the free Gemini quota for the chosen model (requests/day, audio length
  limits). Size the API subset from this number.
- If an OpenRouter key is provided, list audio-input models and run 5 clips
  through `verifier/endpoint.sh --endpoint https://openrouter.ai/api/v1/chat/completions`.
  The endpoint reads `OPENAI_API_KEY` from the shell environment, not from `.env`.
- Inventory `thu-am-studio`: speakers, total single-speaker duration, rights.
- TTS smoke test on ROCm (1 h): choose a recipe that fine-tunes on one speaker
  in ≤ 3 h (F5-TTS from a Vietnamese checkpoint if available and licensed;
  otherwise a single-speaker VITS/Matcha-class model). If ROCm blocks it, run
  Phase 4 on `vsf-242`.
- One-day audit of the E2B all-pass failure: adapter actually loaded, audio tokens
  present in the forward pass, loss masked to the answer only. Record the result;
  it becomes a paragraph in §C3 either way.

### Phase 1 — ViCut-Bench (weeks 2–4)

**Anchors.** ~300 clean 3–12 s utterances from studio audio, trimmed with a
measured speech offset (energy/VAD), source-disjoint splits by recording.

**Defect operators** (applied relative to detected speech onset/offset, never file
edges; SNR is target-to-interferer, so larger = quieter interferer):

| Defect | Operator | Severities |
|---|---|---|
| Clipped word end | Cut Δt before speech offset; anchors whose final syllable ends in `-p/-t/-c/-ch` (from orthography) plus open-syllable controls | 10, 20, 40, 80 ms |
| Clipped word start | Cut Δt after speech onset | 10, 20, 40, 80 ms |
| Secondary speaker | Add a different speaker's fragment (0.2–1.0 s, edge or mid) | SIR 24, 18, 12, 6 dB |
| Separation residue | Mix anchor + music, run our separator, keep vocal stem (real Demucs/RoFormer artifacts, not additive music) | music SNR before separation 0, 5, 10, 15 dB |
| Background music | Additive music on the clean anchor | SNR 30, 24, 18, 12 dB |

300 anchors × 5 defects × 4 severities = 6,000 defective clips + 300 clean
originals. Every defective clip has its clean twin, so paired tests are free.

**Perceptual calibration.** Two listeners do an ABX/forced choice on 20 pairs per
(defect, severity). The lowest severity heard above chance defines the "should
reject" threshold. This separates operator labels from audible defects.

**Real subset.** ~400 clips: the 288 gold clips plus ~110 from fresh,
source-disjoint recordings. Two annotators label `speaker_purity`,
`word_completeness`, `audio_quality` blind to Gemini. Report Cohen's κ and
human–Gemini agreement. Humans are the ground truth; the gold set stays frozen.

### Phase 2 — Judge audit (weeks 4–7)

| Judge | Where | Clips |
|---|---|---|
| Signal baselines: final/initial frame energy and spectral flux at edges, pyannote overlapped-speech detection, speaker-embedding variance across windows, separated-accompaniment RMS, DNSMOS | local CPU/GPU | all |
| Local audio LLMs: Gemma 4 E2B/E4B, Qwen2-Audio-7B (4-bit), MiniCPM-o, Kimi-Audio (existing `verifier/*.sh`) | RX 9060 XT | all (~3.5 s/clip ⇒ ~6 h/model) |
| Gemini (free tier) | API | stratified subset (e.g. 15 clips per defect × severity + 60 clean ≈ 360) + real subset |
| 1–2 OpenRouter audio models | API | same stratified subset |

**Metrics:** detection rate per (defect, severity) with the 50 % detection
threshold per judge; false-reject rate on clean clips; paired consistency
(clean passes *and* twin rejects); on the real subset, R = bad accepted /
accepted, accepted-minute yield, and reject recall per defect. Keep all raw
responses via the production verifier commands.

### Phase 3 — Probes and probe-head gatekeeper (weeks 7–8)

- Extract mean- and edge-pooled features from every layer of frozen encoders
  (Gemma 4 audio encoder, Whisper-large-v3 / PhoWhisper encoder, WavLM).
- Train logistic-regression probes per defect on the synthetic train split; test on
  held-out anchors and on the real subset.
- Compare probe detection with the generated verdict of the LLM built on the same
  encoder. Expected outcomes are both publishable: information present but
  discarded by generation, or absent from the encoder.
- The best probe head (a few KB on top of a frozen encoder) is evaluated as a
  local gatekeeper on the real subset, using R and yield.

### Phase 4 — TTS dose-response (weeks 8–11)

- One speaker, fixed clean training set (1–2 h from studio audio), fixed text,
  steps and seed.
- Conditions: clean; each of clipped end, secondary speaker and separation
  residue injected at 30 % of clips; mixed defects at 10 % and 30 %. That is 6 runs,
  plus a second seed for clean and mixed-30 % (8 runs at ≤ 3 h each).
- Evaluation on 200 held-out sentences:
  - CER and WER with PhoASR-whisper-small and PhoWhisper.
  - Tone error rate, defined as errors that disappear when tone diacritics are stripped.
  - SECS (WavLM-TDNN or CAM++).
  - DNSMOS and UTMOS, reported as relative scores only because UTMOS is trained on English.
  - Truncated-ending rate, using an energy criterion at the end of the synthesised utterance.
  - Secondary-voice rate, using the overlap detector.
- Small listening test: 10 listeners × 20 pairs, clean vs mixed-30 %.

### Writing and buffer (weeks 12–17)

Draft from week 12; internal review week 14; buffer for reruns until the deadline.

## 5. Hypotheses and decision points

| ID | Hypothesis | If false |
|---|---|---|
| H1 | Judges' 50 % detection thresholds for clipped ends and secondary speech are far above the perceptual thresholds | Report as calibration of judges; still a benchmark paper |
| H2 | Paired consistency is low (judges pass both twins) | Leniency claim dropped |
| H3 | Encoder probes detect defects at lower severity than the LLM's verdict | Report encoder blindness; the gatekeeper claim falls back to signal baselines |
| H4 | Defects degrade TTS dose-dependently: clipped ends raise truncated endings and CER; crosstalk lowers SECS | Reframe downstream as "which defects are tolerable", which still informs curation |

Decision after Phase 2 (end of week 7): if no judge or baseline detects any defect
class at audible severities, consider shifting weight to C3/C4.

## 6. Budget

| Item | Cost |
|---|---|
| Synthetic generation, signal baselines | CPU, hours |
| Local judge inference | ~6 GPU-h per model × 4–5 models |
| Feature extraction + probes | ~5 GPU-h |
| TTS dose-response | ~24 GPU-h (local or `vsf-242`) |
| Gemini free tier | ~800 requests total, spread over days |
| OpenRouter | ~800 short audio requests per model, only if a key is provided |
| Human annotation | ~400 real clips × 2 annotators + ~800 calibration pairs |

## 7. Literature, checked against `research/notes/`

- **QualiSpeech** (Wang et al., 2025): open auditory LLMs show *numerical bias*.
  Qwen2-Audio-7B-Instruct gives 3 to 66 % of samples, and SALMONN-7B predicts every
  continuity score as 4. Its training data already adds simulated noise and NISQA
  distortions, so generic synthetic perturbation is prior art. Our novelty is
  **pipeline-induced boundary and contamination defects** in Vietnamese,
  including unreleased stop codas.
- **SpeechLLM-as-Judges** (Wang et al., 2025): CoT + GRPO speech-quality judge,
  about 43 A100 GPU-hours. That training budget is beyond ours, which motivates
  probes and baselines instead.
- **ALLD** (Chen et al., ICLR 2025): token-level distillation from a text LLM
  (LLaMA-3.1-70B) given human-rated meta information. If we later train a
  descriptive judge, our perturbation parameters can play the meta-information role.
- **PhoAudiobook** (Vu et al., ACL 2025): 23K h of audiobooks → 1,400 h after
  Demucs, a Whisper-large-v3/PhoWhisper-large exact-match filter, a wav2vec2-bartpho
  multi-speaker filter and silence filtering. The cut to 941 h is a deliberate
  4 h/speaker cap. This is a single-narrator domain; do not cite it as evidence
  that conversational filters discard valid speech.
- **PhoASR** (Vu et al., Findings EACL 2026): Vietnamese ASR corpus with word-level
  timestamps; its curation keeps samples where two ASR systems agree with WER < 5 %.
  We use it as the ASR metric model.
- **Seed-TTS** (2024): standard objective protocol for zero-shot TTS (WER + speaker
  similarity). Does **not** study training-data defects.
- **VoiceCraft** (2024): reports occasional long silences and dragged sounds in
  generation. These are failure modes worth measuring in Phase 4, but the paper does
  not attribute them to training-data clipping.
- **DAC** (2023): codec background only; makes no claim about data hygiene. Dropped
  from the argument.
- **To add before writing:** Emilia / Emilia-Pipe (in-the-wild TTS data processing;
  the expected curation baseline), DNSMOS, pyannote overlapped-speech detection,
  Brouhaha. AHa-Bench and MISHAP-Bench are not in the local corpus; verify their
  bibliographic details before citing.

## 8. References (local copies)

1. Wang et al. *SpeechLLM-as-Judges.* arXiv:2510.14664 — [pdf](papers/2510.14664_SpeechLLM_as_Judges.pdf)
2. Wang et al. *QualiSpeech.* arXiv:2503.20290 — [pdf](papers/2503.20290_QualiSpeech.pdf)
3. Chen et al. *Audio Large Language Models Can Be Descriptive Speech Quality Evaluators.* ICLR 2025, arXiv:2501.17202 — [pdf](papers/2501.17202_ALLD.pdf)
4. Vu, Nguyen, Nguyen. *Zero-Shot Text-to-Speech for Vietnamese.* ACL 2025 (short) — [pdf](papers/2506.01322_PhoAudiobook_ACL2025.pdf)
5. Vu, Nguyen, Nguyen. *Vietnamese Automatic Speech Recognition: A Revisit.* Findings of EACL 2026 — [pdf](papers/2026.findings-eacl.345_PhoASR.pdf)
6. Seed Team. *Seed-TTS.* arXiv:2406.02430 — [pdf](papers/2406.02430_SeedTTS.pdf)
7. Peng et al. *VoiceCraft.* arXiv:2403.16973 — [pdf](papers/2403.16973_VoiceCraft.pdf)
8. Kumar et al. *High-Fidelity Audio Compression with Improved RVQGAN.* NeurIPS 2023 — [pdf](papers/2306.06546_DAC_AudioCodec.pdf)
