# Local Vietnamese TTS data: production strategy

Created 2026-09-08. Status: execution started; no production quality claim established.

## Controlling user clarification — 2026-09-08

The user has tested Gemini 3.8 Flash Medium against human judgment and explicitly
accepts it as ground truth. The user authorizes API calls for assistant audio
hearing. **Use fresh, provenance-recorded Gemini 3.8 Flash MEDIUM evaluations to
execute the strategy; do not block on separate human annotation.** References to
required human review below describe the original strategy and are superseded by
this instruction. Human review is optional adjudication, not a phase gate.
Record evaluator model, MEDIUM configuration, full quality rubric, input hashes,
response and usage. A Gemini-ground-truth metric is valid for project decisions;
do not call those labels human-authored. Representative source sampling and
source-disjoint evaluation remain necessary. Production inference stays local.

## Resume here

Read this document, [execution log](TTS_STRATEGY_EXECUTION.md), and
[Phase 1 brief](TTS_PHASE1.md), then inspect `git status` and recent commits.
Continue the first unfinished, unblocked task. Update the execution log before
each checkpoint commit. Do not mistake proposed experiments for completed work.
Runtime manifests, audio, and reports live under `.data/` and are not pushed;
record reproducible commands and concise findings in tracked documentation.

The 2026-09-08 located-cut evaluation is done. Word-lock *expansion* into
inter-turn gaps raised accepted risk (15 reject / 1 pass among new children).
Rejecting every ASR-overlapping edge dropped all 31 clips. The measured policy
is: do not expand into gaps; reject only competitor/adjacent-turn word
completions. That emits 8 duration-eligible crops with inherited MEDIUM R = 50%.

The first locked growth increment is done. Three new recordings were ingested
and extracted with Sortformer + the measured lock (no DiariZen consensus on this
AMD host). Fresh MEDIUM: 20 pass / 16 reject. The labeled pool is now 133 clips
across 10 recordings (62 pass / 71 reject); calibration recordings are frozen.
Studio interview (`10.000 hours` EP5) yielded 16/19 passes; the music-backed FPT
vlog yielded 2/13. Next unblocked task is more first-tier narration/interview
sources, then a local acoustic baseline for leftover music/reverb/in-interval
secondary speech. ASR edge overlap is not clipping proof.

The user authorized saving, committing, pushing, and executing this strategy,
with regular detailed commits so another model can continue. This authorization
does not override AGENTS.md's prohibition on writing/running test cases without
an explicit request. Human acoustic adjudication must be performed by humans;
never fabricate it from teacher outputs. Do not assume access to the user's
4090 from the current AMD host. Do not publish datasets or audio merely because
source-code pushes are authorized. No new crawl→separate→diarize→mix framework.

## Objective and governing decision

Extract approximately 2–15-second Vietnamese speech clips with exactly one
speaker, complete first/final words, no overlap or short secondary intrusion,
no unacceptable music/effects/noise, and preserved voice quality. A tens-of-ms
audible intrusion can invalidate a clip. Precision takes priority over recall.

Optimize **bad accepted clips / all accepted clips**, counting a clip once if
it contains any important defect. Target below 10%, preferably below 5%.
Measure yield and runtime alongside quality so accepting nothing cannot win.

Primary investment: reliable evaluation and boundary-safe candidate generation,
then a small specialized temporal acoustic verifier. Keep the existing working
pipeline and diarizers. Gemini 3.8 Flash Medium is an offline teacher; production
Gemini is a last resort. Gemma 4 12B is a bounded challenger, not an assumed answer.

## Evidence at strategy creation

Verified from local artifacts/code, not human listening:

- `.data/distillation/reports/finetuned_e2b_v3_vs_gemini38.md`: E2B V3 accepts
  all 31 Khanh Vy clips. Gemini rejects 19: 61.3% accepted error **relative to
  the teacher**, not a measured human defect rate. This small challenge set is
  not representative production sampling and includes clips outside 2–15 s.
- V3 manifests have 263 train (119 pass / 144 reject) and 65 validation
  (29 pass / 36 reject) rows. Multiple recording filename prefixes occur in
  both splits, despite no identical filenames between splits.
- `scripts/build_distillation_dataset.py` balances and randomly splits rows,
  not recording groups. Training loss does not establish unseen-source quality.
- `src/diarization/verifier_training.py` predicts JSON plus reasons, selects by
  validation loss, detaches audio features, and adapts language-model projections.
- `src/diarization/zero_contamination.py` performs optional smart segmentation
  after word alignment; child cuts can subsequently move to acoustic valleys.
- Alignment errors retain incoming turns; inspected foundation-verifier request
  errors reject them. Documentation includes contradictory or excessive claims.

Hypotheses to investigate, not findings: semantic representations suppress short
defects; the E2B failure reflects representation limits; 12B waveform projection
will help; a small temporal model can reach the final target. The all-pass failure
also requires adapter, audio-input, prompt-mask, and distribution checks.

## Evaluation contract

Define R = bad accepted clips / accepted clips. Undefined when none are accepted.
Teacher agreement, DER, overall accuracy, F1, and false-positive rate cannot
replace R. Keep every defect label but count each bad clip only once in R.

Report R, accepted minutes/source hour, risk by defect/source/duration/route,
runtime/source hour, runtime/accepted hour, and missing/failed gate counts.
Missing labels are unknown, never clean. Failed inference is not an acceptance.

Use three separate collections:

1. Production audit: random sampling from the intended production population.
2. Challenge set: deliberately enriched subtle defects; diagnostic only.
3. Training pool: human, teacher, and synthetic examples with provenance.

Split by recording **before augmentation**. Alternate cuts, stems, and synthetic
derivatives share the parent's split. Resolve aliases using authoritative source
IDs. Unknown lineage is quarantined, not guessed. Include unseen-channel and,
where required, unseen-speaker evaluation. Tune thresholds on calibration data,
then freeze policy and evaluate untouched recordings. Never tune on release data.

Release gate: one-sided 95% upper risk bound below 10%, preferably below 5%,
within a declared source scope. Plan 500–1,000 audited accepted clips across many
recordings; analyze source dependence (recording-cluster resampling/source strata)
instead of claiming neighboring clips are independent. An iid binomial bound is
only an initial diagnostic. Zero defects in 59 independent clips yields an upper
bound just below 5%, but one interview is not a production qualification.

Provisional planning yield floor: 10 accepted minutes/source hour in the easiest
source tier; not a user-approved contractual target. Revisit against data needs.

## Candidate generation and boundary correctness

Use Sortformer/DiariZen agreement to identify credible speaker regions, not
finished word boundaries. Preserve competitor evidence from both engines even
when consensus removes it. Intersection and inward shaving can clip target
speech; outward expansion can restore a word while importing another speaker.

Find complete utterances inside trusted regions. Use ASR/alignment to propose
word bounds, pauses and local spectral/energy evidence to support them. Around
uncertain handoffs, skip the edge utterance and start at a later complete one.
Reject if no safe 2–15-second interval exists. Do not force a duration split.

Generate all children before final boundary checks; validate every child's start
and end. ASR timestamps, low energy, and zero crossings are not guarantees.
Avoid universal dBFS thresholds across mastering conditions. Fades do not repair
clipped phonemes. Natural Vietnamese unreleased stops or abrupt endings must not
automatically be labeled clipping: compare with the original continuation.

Required unavailable/failed/inconclusive evidence means reject/quarantine under
the strict production policy. Development fallback must not masquerade as a
passed production gate. Preserve informative audit records.

## Audio paths and verification inputs

Retain original audio and the exact export, with source/sample offsets and
processing provenance. Keep analysis-rate copies separate from high-quality
source/export audio. Account for resampling, delay, and channel information.

Route already clean speech directly from the original. Separate only where
useful; verify music residue and processing/voice damage on the stem. Initially
reject severe crowd speech, overlap, reverberation, and heavily produced sources.
Music in the original is not proof of contamination in a successfully cleaned
export; cleanliness of the original is not proof the stem is undamaged.

Verifier inputs: full candidate, approximately 0.5–1.5 s of outside context at
each edge, original/stem pairing where applicable, and a trusted target-speaker
reference when available. Outside-context events do not invalidate an inside
interval unless they establish a defect at its boundary. Validate exact exported
samples; later material processing requires renewed validation.

## Specialized local modeling sequence

1. Calibrate existing overlap/segmentation evidence and one music/effects model
   (PANNs is a baseline candidate). Retain only gates improving risk/yield.
2. Train WavLM Base+ with a shallow temporal convolutional multi-label head.
   Start frozen, then investigate limited encoder adaptation if justified.
3. Use full-clip plus contextual start/end views. Retain temporal evidence;
   compare short-window/peak aggregation to averaging. Calibrate false spikes.
4. Add a small 5–10 ms-hop log-Mel branch only as an ablation if speech embeddings
   miss short acoustic cues. Fine output spacing is not fine detection accuracy.
5. Condition on target references only when trusted; 20 ms may be insufficient
   for speaker identity even when contamination is audible. Reject uncertainty.

Labels are multi-label: secondary speaker, overlap, clipped start/end, music,
effects, noise, reverb, processing artifacts. Tail intrusion is secondary speech
with end location, optionally exposed as a convenient UI label. An additional
model must demonstrate incremental value on unseen sources, not only agreement.

## Labels and offline teacher

First-cycle budget: 800–1,200 human-audited real candidates across approximately
30–50 recordings; double-review ambiguous cases and a random routine subset.
Start training with 3,000–5,000 mixed-provenance examples, grow by learning curves.
These are planning numbers, not quality guarantees.

Capture defect, location, approximate interval and uncertainty, audibility,
severity, reviewer/teacher provenance, and accepted/rejected/unresolved state.
An unannotated dimension is not clean. Distinguish acoustic completeness from
grammatical completeness. Freeze the listening rubric before release evaluation.

Gemini proposes labels, transcripts, broad events, difficult regions and
disagreement priorities. Humans anchor the criterion. Store exact teacher model,
prompt, reasoning setting, response, and date. Teacher timestamps are approximate
unless verified; use weak clip supervision or uncertain regions, not invented
precise frame targets. Video-only evidence cannot justify audio-only defects.
Teacher-generated explanations are not evidence of an acoustic mechanism.

Synthetic controlled pairs: 20/40/80/150/300 ms intrusions across relative levels,
locations, voices; clean/clipped pairs; natural abrupt endings, same-speaker
breaths/fillers/laughter as controls. Shaving silence is not clipping. Avoid splice
click shortcuts and keep all derivatives in their recording's split. Human-check
realism; synthetic timing precision does not establish perceptual importance.

## Gemma 4 12B challenger and hardware

Google documents direct 16 kHz waveform projection in 640-sample (40 ms) frames.
This provides a plausible experiment, not proof of micro-defect sensitivity.
The frame size is neither a guaranteed detection limit nor a capability claim.

On the user's RTX 4090 24 GB: frozen NF4 base, BF16 LoRA rank 8/16 initially,
gradient checkpointing, microbatch one, short labels and bounded context. Measure
actual peak memory and throughput before scale-up. Q4 weight size is not training
memory. Current execution host is AMD; do not assume CUDA kernels or remote access.

Sequential ablations: base; LoRA only; LoRA plus audio projection. The last needs
removing relevant detachment and verifying gradients/checkpoint persistence.
Before expensive training investigate the E2B all-pass collapse with matched
audio pairs, audio substitution, adapter on/off, and prompt-mask inspection.
No test cases may be written/run without explicit authorization under AGENTS.md.

Select by held-out accepted risk/yield, not JSON loss. Limit challenger effort to
one bounded cycle. Keep it only for incremental production value. Rent larger
hardware only if a promising result is demonstrably constrained by memory/context.

## Ordered execution and exit gates

| Phase | Deliverable | Exit gate |
|---|---|---|
| 1 | Human rubric, existing-artifact audit, source lineage/splits, baseline measurement | Human-grounded risk and ranked defects; pending human work cannot be marked done |
| 2 | Final-boundary ownership, safe splitting, strict gate status, extraction-route comparison | Lower boundary risk at useful yield |
| 3 | Calibrated existing local overlap and event baselines | Best pre-training local risk/yield curve |
| 4 | Small temporal verifier and targeted labeling | Gains on unseen recordings and brief events |
| 5 | Bounded 12B QLoRA challenge | Incremental quality/yield justifies cost |
| 6 | Frozen policy, release audit, throughput and drift monitoring | Human risk bound meets target within declared scope |

One developer plus annotation support: roughly 6–8 weeks, contingent on data and
hardware. Begin with narration/well-recorded interviews, add edited dialogue,
then qualify harder tiers separately. Infrastructure work independent of human
labels may proceed while review is pending; never skip evidential phase gates.

## Production and fallback

Compose current public APIs; no new orchestration framework. Persist source ID,
output hash, sample offsets, route, model revisions, thresholds, and gate status.
Do not multiply dependent model confidences or assume another AND gate improves
survivor risk. Calibrate and measure the entire cascade. Randomly audit accepted
outputs continuously, including confident ones, and track source/model drift.

If local risk misses target: tighten acceptance, restrict source tiers, label
dominant residual failures. Only then evaluate Gemini on local survivors against
human labels. It cannot override a known hard defect. A required production
Gemini gate means cloud-assisted operation, not local-only production.

## References and limits

- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm): speech representation baseline, not a pretrained Vietnamese clipping detector.
- [Pyannote segmentation](https://huggingface.co/pyannote/segmentation-3.0): local speaker/overlap outputs.
- [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn): sound-event baseline.
- [whisper_timestamped](https://github.com/linto-ai/whisper-timestamped): alignment capabilities and limitations.
- [Gemma 12B guide](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/): waveform-frame architecture.
- [Gemma overview](https://ai.google.dev/gemma/docs/core): weight memory is not training memory.
- [QLoRA](https://arxiv.org/abs/2305.14314): quantized-base adapter training.
- [Gemini audio](https://ai.google.dev/gemini-api/docs/audio): no demonstrated ms-level guarantee.
- [NIST proportions](https://itl.nist.gov/div898/handbook/prc/section2/prc24.htm): independent-binomial baseline, not a remedy for clustered sampling.

References were checked during the strategy conversation on 2026-09-08. Pin actual
software/model revisions in experiment manifests; recheck compatibility before use.
