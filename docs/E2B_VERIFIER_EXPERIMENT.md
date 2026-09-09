# E2B acoustic verifier experiment

Created 2026-09-08. Status: plan and workspace scaffold only; no new training,
generation, annotation, or benchmark evaluation has been performed.

## Controlling decisions

- Preserve `.data/tts_strategy/gold_benchmark_20260908/` and its 288 clips as
  the fixed benchmark. Do not relabel, regenerate, augment, rebalance, or use
  these clips for training, seed selection, calibration, or checkpoint selection.
  Reserve all their source recordings and derived clips against new training
  and development data. Store future benchmark predictions outside that folder.
- This benchmark was previously evaluated and includes historical training
  overlap. Freezing it now does not erase exposure or establish unseen-source
  generalization. Record this limitation with every final result.
- Use separate sources for development baselines, debugging, and model selection.
  Run the 288-clip benchmark only after freezing the candidate and decision policy.
- `reason` is optional. Omit it from synthetic training targets; do not generate
  explanations merely to fill a field. Keep provenance/evidence outside model
  targets. Missing reasons must not affect parsing, scoring, or acceptance.
- Gemini 3.8 Flash MEDIUM remains the accepted project reference annotator,
  with model/configuration, input hashes, raw responses and usage recorded.
  Human listening is optional adjudication, not a mandatory gate.
- Runtime data stays under `.data/`. No automatic publication or cloud tracking.
  This scaffold does not authorize starting paid annotation or training.

## Stages and completion gates

1. **Diagnose the all-pass student on separate development audio.** The existing
   execution log records E2B V3 accepting 288/288 clips including 129 reference
   rejects. Inspect adapter loading, audio sensitivity, preprocessing, templates,
   answer masking, gradient flow and checkpoint selection. Compare untouched
   E2B, V3 and optionally untouched E4B on new development sources. Report
   per-defect recall, bad accepts, clean rejects, invalid responses and latency.
   Gate: explain the collapse before scaling training; loss alone is insufficient.
2. **Freeze labels and source assignments.** Inventory the benchmark's recording
   identities read-only and reserve them. Assign new recordings to train/dev and
   optionally a fresh final-generalization set before creating derivatives.
   Keep parents, overlapping excerpts, donors and all descendants in the same
   split; separate known speakers where possible. Quarantine unresolved lineage.
   Gate: inspect source and duplicate-audio leakage before exporting training rows.
3. **Admit 300–500 clean seeds from roughly 20–30 new recordings.** Diversify
   speakers, accents, microphones and Vietnamese endings. Preserve surrounding
   source audio and exact sample offsets. Detector agreement screens candidates;
   uncertain boundaries stay outside the clean seed pool. Include naturally
   abrupt endings and tightly cut but complete speech as valid examples.
4. **Generate 3,000–5,000 pilot examples.** Start with clear clipped starts/ends,
   audible secondary speech, overlap, music, SFX and noise. Preserve clean parents.
   Include matching processing controls, same-speaker continuation, tight complete
   cuts and natural breaths. Audit outputs from each rule before scaling. Keep
   tiny intrusions and uncertain boundary cuts in a challenge pool. A known
   operation does not establish a perceptually detectable defect.
5. **Compare two E2B IT runs from the same untouched checkpoint.** Run A:
   decoder LoRA + trainable audio projection, frozen encoder. Run B: same setup
   plus audio encoder LoRA. Start at rank 16 with identical data/protocol and
   short structured targets. Confirm modules and gradients explicitly. Measure
   memory/throughput on the 9060 XT; validate the installed training stack before
   assuming any Unsloth flag works. Select checkpoints on real development
   decisions, not only teacher-forced loss.
6. **Expand only after transfer to real development failures.** If synthetic
   performance improves alone, revise generation and add separate-source real
   failures. If useful, expand to 10,000–30,000 examples, including separation
   artifacts and subtle transitions. Consider E4B adaptation based on evidence.
7. **Freeze and benchmark.** Freeze checkpoint, schema, prompt and decision
   policy before running all 288 fixed clips. Save predictions and metrics in
   the new experiment workspace. Report bad accepted/all accepted, per-defect
   misses, clean rejection, review rate, usable duration, latency and memory.
   Report counts and uncertainty, including recording-level variation. Keep
   optional unseen-source evaluation separate from this fixed benchmark.

Budgets are initial planning estimates, not guarantees. Target <10% bad clips
among accepts, preferably <5%, while constraining clean rejection and review
rates on development data before final evaluation. Accepting nothing cannot win;
zero accepted clips makes accepted risk undefined. No production claim follows
from teacher agreement alone without representative sampling and stated limits.

## Scaffold and usage

```bash
uv run --no-sync python scripts/scaffold_verifier_experiment.py --name e2b_acoustic_v1
```

Creates a fresh `.data/verifier_experiments/<name>/`; refuses an existing target.
Only creates directories, empty manifests and planning templates. It does not
read or modify benchmark data, load a model, call an API, or chain pipeline steps.

```text
experiment.json                 descriptive run specifications, not trainer config
templates.json                  incomplete record templates; never training rows
manifests/
  sources.jsonl                 source reservations and assignments
  seeds.jsonl                    admitted seed records
  examples.jsonl                 generated/real examples with full provenance
  train.jsonl                    future trainer exports
  dev.jsonl                      future development exports
  uncertain.jsonl                ambiguous examples, excluded from supervised pilot
audio/{seeds,synthetic}/
reports/{baseline,development,benchmark}/
checkpoints/{run_a,run_b}/
```

The benchmark path in `experiment.json` is a read-only reference. Source
reservations must be populated before admitting seeds; empty manifests do not
constitute a completed leakage audit. `templates.json` uses null for unknown
metadata rather than invented IDs, hashes, confidence scores or acoustic labels.

## Experiment manifest contract (version 1)

Paths are repository-relative. Source records contain `recording_id`,
`speaker_ids` and `split` (`benchmark`, `train`, `dev`, or `final_generalization`).
Example records contain `id`, `audio_path`, `audio_sha256`, `recording_id`,
`split`, `parent_ids`, `donor_ids`, `source_recording_ids`, `sample_rate`,
`source_start_sample`, `source_end_sample`, `operation`, `labels` and
`label_provenance`. Source bounds are half-open offsets in the recorded source
sample rate; derived operations record their own coordinate rate. Include every
donor recording in `source_recording_ids`, not just the primary recording.

`operation` records type, generator version, random seed, cut/insert sample
indices, coordinate sample rate, gains and SNR definition where applicable.
`label_provenance` records synthetic/teacher/human origin, rubric version,
evidence and any teacher model, configuration and response path. Avoid numerical
confidence unless calibrated. Physical contamination and audible contamination
are distinct; alignment estimates are not exact phonetic boundaries.

`labels` maps each defect to `present`, `absent`, or `uncertain`:
`clipped_word_start`, `clipped_word_end`, `secondary_speaker`,
`overlapping_speech`, `music_bleed`, `sound_effects`, `excessive_noise`,
`reverberation`, `distorted`. Multiple defects may coexist. Proposed short model
target: `{"labels": {...}, "decision": "pass|reject|review"}`. Any present
defect means reject; otherwise any uncertain defect means review; only all
absent means pass. Missing/invalid labels are not implicit absent values.
`reason` may be supplied but is never required or a scoring input.

These are experiment-only records. They are not yet the production verifier
response contract. Future trainer exports use existing `audio_path`, `prompt`,
`target_json` fields and preserve lineage. Export only unambiguous pilot labels;
do not pass raw provenance manifests directly to the trainer.

## Existing code and pending implementation

- `scripts/audit_tts_data.py`: reuse source/provenance inventory and teacher tooling.
- `scripts/build_distillation_dataset.py`: existing balancing groups recordings
  and checks duplicate bytes; it still needs reserved benchmark/donor lineage
  handling for this experiment. Its teacher prompt still requests a reason.
- `scripts/evaluate_verifier.py`: reuse inference/reporting, but explicitly pass
  development input because its default is the 288-clip benchmark. Add the new
  experiment-label scoring and optional-reason handling before these runs.
- `scripts/train_verifier.py` and `src/diarization/verifier_training.py`: existing
  training serializes `target_json` without requiring a reason. The loader detaches
  audio features and adapts language layers only. Run A/B require implementation;
  simply changing a specification does not enable projection/encoder training.
  Inspect checkpoint selection and local dataset loading. Future pilot execution
  must use `--no-push-hub --wandb-mode disabled` and explicit dataset/output paths.
- Synthetic generation, source/donor validation, trainer export, new response
  scoring and audio adaptation are pending. No placeholder training commands
  claim these are ready.

Next action: stage 1 inspection and development-source preparation. Do not run
the fixed benchmark as a debugging loop. No test cases, commits or pushes were
requested for the scaffold.
