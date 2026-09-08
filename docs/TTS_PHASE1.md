# Phase 1: establish trustworthy acceptance evidence

Parent: [production strategy](TTS_PRODUCTION_STRATEGY.md).

## Updated ground-truth authority

The user explicitly accepts Gemini 3.8 Flash MEDIUM as human-quality ground
truth and authorizes API audio assessment. Execute this brief with that evaluator;
the original human-review requirements below are now optional. Preserve exact
evaluator/input provenance. Do not reuse legacy labels as MEDIUM merely because
a historical report says so: the archived Khanh Vy generator configures LOW,
and all 31 saved references omit audio_quality. Re-audit the complete rubric.

## Work order

1. Inventory existing V3 manifests and saved verifier predictions without model
   inference. Count missing audio, durations, duplicate identities, source lineage
   coverage, split leakage, labels, and accepted teacher disagreements.
2. Export a blind human-review queue for existing challenge clips. Keep teacher
   answers separate so reviewers are not primed. This queue is diagnostic and
   must never be described as a random production audit.
3. Define explicit recording IDs and parent lineage before replacement splits.
   Filename prefixes may identify leakage risks but cannot prove disjointness.
4. Report accepted-sample risk only from valid human decisions with coverage,
   unknown-label counts, accepted denominator, and sampling caveats. Teacher
   reference error remains a separate metric. No accepted items means undefined.
5. Inspect training audio path, masking, adapter targets, and checkpoint selection.
   Defer inference/gradient experiments until data/protocol and hardware are ready.

## Human listening protocol

Listen to the exact candidate at normal speed with consistent headphones and
playback gain; inspect contextual original/stem audio for uncertain edges. Slow
playback and spectrograms help diagnosis but do not redefine normal-speed audibility.
Do not independently normalize tiny edge windows into misleading loud intrusions.

Assess all of: speaker purity, overlap, initial/final word integrity, music,
important effects, unacceptable noise/reverb, and processing/voice damage.
Accept grammatical fragments if acoustically complete. Natural Vietnamese final
stop closures, breaths and same-speaker fillers are not automatically defects.

Use `pass`, `reject`, or `unresolved`; no default answer. A reject needs one or
more defects. A pass needs assessment of every required dimension. Store reviewer
ID, rubric version, evidence coverage (candidate/context), notes, and approximate
intervals with uncertainty. Human review never edits teacher labels in place.
Use two reviewers for ambiguous cases and a random routine subset; adjudicate
disagreement. Unknown/no context is explicit, not fabricated.

## Sampling and release limits

Existing Khanh Vy examples are a challenge set with teacher judgments, not a
production baseline. A representative baseline requires random samples from the
actual intended source population and recorded inclusion probabilities if stratified.
Reviewing only model disagreements biases risk estimates. Include confident
accepts and ordinary clean examples. Keep recording families disjoint across
training/calibration/final evaluation; all derived audio follows its parent.

Initial human effort: 800–1,200 candidates from 30–50 recordings, expanding as
needed. Release evaluation separately targets 500–1,000 accepted clips spanning
recordings. Report source clustering and confidence-bound assumptions explicitly.

## Exit criteria

- Reproducible, provenance-aware inventory and baseline reporting exist.
- A human rubric and actionable blind queue exist; actual human decisions are
  required before claiming human error rates or ranking perceptual failure causes.
- Source-disjoint manifests use verified recording IDs, with unresolved lineage
  quarantined; no random clip split presented as unseen-source validation.
- Current pipeline/configuration and model versions are recorded when known;
  unknown versions remain unknown.
- Phase 2 infrastructure may proceed independently, but its improvement claim
  requires the human evaluation above.
