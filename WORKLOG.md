## 2026-09-17 - Update AGENTS.md with WORKLOG.md tracking instructions

### Decisions
- Added explicit instructions in `AGENTS.md` under both `## Non-negotiable rules` and `## Implementation and completion` requiring coding agents to append decisions and results to `WORKLOG.md` using bash.
- Created `WORKLOG.md` to preserve historical continuity and ensure progress and architectural decisions are tracked across sessions.

### Results
- Updated [AGENTS.md](file:///home/nguyenlt/Documents/tts-data-pipeline/audio-prepare-pipeline-redo/AGENTS.md).
- Initialized [WORKLOG.md](file:///home/nguyenlt/Documents/tts-data-pipeline/audio-prepare-pipeline-redo/WORKLOG.md).

## 2026-09-17 - Plan a recursive VAD-only baseline segmenter

### Decisions
- Interpret the requested algorithm as recursive binary partitioning: for each interval longer than 15 seconds, cut at the lowest Silero speech-probability position within that interval, then apply the same rule independently to each oversized child.
- Keep this baseline independent from ASR, punctuation, word timestamps, diarization, and the existing `audio/segment_tts` strategy.
- Reuse the completed `evaluate/silero_jit` probability report rather than duplicate model loading or couple VAD inference to cutting.
- Preserve the full source timeline. VAD selects cut boundaries only; it does not trim silence or classify speakers.
- Plan a separate `audio/segment_vad` manifest planner and same-name Bash launcher, with rendering left to `audio/export_segments`.

### Results
- Inspected the existing Silero probability report, TTS segmentation planner, shared segment normalization/export helpers, CLI documentation, and manifest contract.
- Defined deterministic boundary selection, recursion, sample-accurate duration enforcement, audit metadata, CLI validation, documentation updates, and non-model verification steps for a future implementation.
- No implementation or tests were performed because this task requested explanation and planning only.

## 2026-09-17 - Implement and evaluate recursive VAD-only segmentation

### Decisions
- Added an independent `audio/segment_vad` planner rather than changing the ASR-aware `audio/segment_tts` command.
- Reused completed `evaluate/silero_jit` probability reports; the planner performs no model inference and clip rendering remains a separate `audio/export_segments` command.
- Implemented the requested literal recursion: split every oversized interval at its lowest speech-probability frame, with deterministic midpoint/earliest tie-breaking and exact source-sample duration enforcement.
- Excluded the final partial VAD frame because Silero evaluation zero-pads it beyond the real source duration.
- Used `speaker_id: unknown` because VAD provides no speaker identity.
- Evaluated only rendered intervals at least one second long to avoid paid judgments on tiny silence shards; searched existing Gemini 3.8 Flash artifacts by exact clip hash before submitting new work.

### Results
- Added `scripts/audio/segment_vad.py` and its same-name Bash launcher, and documented the CLI, manifest contract, and recorded evaluation.
- On the 89.629-second `a9aVQCWi9MY` example, the planner produced 97 gapless intervals from 96 cuts; the maximum was 14.976 seconds, but 86 intervals were under one second because clustered quiet frames were peeled off recursively.
- Reused the existing Silero report, so no new VAD inference ran. None of the 11 substantive rendered clips matched an existing Gemini result by SHA-256.
- Submitted the 11 clips in one Gemini Batch job using `gemini-3.8-flash` and the existing default verifier prompt. All 11 were judged word-complete; four passed overall and seven failed for unrelated speaker/music/singing defects. Estimated cost was $0.02435775.
- Reused the existing same-model, same-prompt `gemini_vad_protected_relaxed` aggregate as contextual evidence: 54/62 complete, two clipped starts, and six clipped ends. This does not test the exact new algorithm but confirms that VAD evidence is not a phoneme-completeness guarantee.
- Validated launcher syntax, Python compilation, CLI execution, exact timeline coverage, adjacency, and the <=15-second sample invariant. No test suite was written or run.

## 2026-09-17 - Document machine-specific quirks in CURRENT_MACHINE.md

### Decisions
- Created a local, gitignored `CURRENT_MACHINE.md` file summarizing current host hardware, ROCm/HIP device mappings, attention kernel quirks, quantization restrictions, audio loader fallbacks, and memory ceilings.
- Added `CURRENT_MACHINE.md` to `.gitignore` to keep host-specific quirks strictly local without polluting version control across different nodes.
- Documented the purpose of `CURRENT_MACHINE.md` shortly in `AGENTS.md` under `## Machines`.

### Results
- Created [CURRENT_MACHINE.md](file:///home/nguyenlt/Documents/tts-data-pipeline/audio-prepare-pipeline-redo/CURRENT_MACHINE.md) with details for AMD Radeon RX 9060 XT (gfx1200), ROCm 10.0, PyTorch `cuda:0` mapping, AOTriton Gemma 4 issue, bitsandbytes CUDA limitations, torchaudio/soundfile fallback, and memory limits.
- Ignored `CURRENT_MACHINE.md` in [.gitignore](file:///home/nguyenlt/Documents/tts-data-pipeline/audio-prepare-pipeline-redo/.gitignore).
- Updated [AGENTS.md](file:///home/nguyenlt/Documents/tts-data-pipeline/audio-prepare-pipeline-redo/AGENTS.md) to describe `CURRENT_MACHINE.md`.
- Appended tracking log to [WORKLOG.md](file:///home/nguyenlt/Documents/tts-data-pipeline/audio-prepare-pipeline-redo/WORKLOG.md).
- Validated with `git check-ignore -v CURRENT_MACHINE.md`. Tests were omitted because none were requested.

## 2026-09-17 - Download Silero JIT during environment setup

### Decisions
- Pin the Silero native JIT URL to upstream commit `41f03a954b841327835dea1ddb7bb28ae23ddc2c` and verify SHA-256 `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`.
- Cache the verified artifact at `~/.cache/silero-vad/silero_vad.jit`; reuse valid existing files and download atomically through a temporary file.
- Run the cache check from the `audio`, `align`, and `vibevoice` setup targets so every supported Silero workflow provisions the same model.
- Make `evaluate/silero_jit.py --model-file` optional, defaulting to the setup-managed cache path, while retaining explicit override support.

### Results
- Updated `envs/setup_worker_envs.sh`, the Silero evaluator, README, and segmentation documentation.
- Ran `./envs/setup_worker_envs.sh audio`: downloaded and verified the 2.16 MiB model, then reran it successfully using the cached file.
- Ran the evaluator without `--model-file` on `.data/test_amd/test_input.wav`; CPU inference completed and wrote `.data/evaluate/silero_cache_smoke/report.json`.
- Validated shell syntax, Python compilation, and diff formatting. No test suite was written or run.

## 2026-09-17 - Guide the Hana example VAD workflow

### Decisions
- Use the existing matching source under `.data/s1-download/truyen-chem/` because the requested `.data/s1-download/hana-playlist/` path is absent in this checkout.
- Reuse `.data/evaluate/hana_tts/vad/report.json` only after checking its source hash against the selected WAV; regenerate the report if a different copy is used.

### Results
- Confirmed the existing source/report pair is complete and hash-matched, with 916.448875 seconds of audio and a successful CPU VAD track.
- Prepared a step-by-step workflow covering setup/cache, source identity, report reuse or regeneration, recursive VAD planning, rendering, inspection, and optional Gemini review.

## 2026-09-17 - Make recursive VAD the default long-segment strategy

### Decisions
- Reused the existing recursive Silero VAD planner through a shared `_common/vad.py` helper so diarizers and standalone `audio/export_segments` apply identical sample-accurate cuts.
- Made `--long-segment-strategy vad` the default for turns longer than `--max-duration-s`; retained the former discard behavior as explicit `--long-segment-strategy drop`.
- Kept VAD inference independent: commands consume a matching completed report via `--vad-report`, or `<audio-stem>.json` files via `--vad-report-dir` for directory runs.
- Preserved uncut backend output in `segments.raw.json`; VAD cuts and lineage are recorded in the merged/final manifests through `long_segment_audit`.

### Results
- Wired the strategy through all six diarization backends, `audio/export_segments`, and the standalone VAD planner.
- Updated README, command cookbook, data contract, and VAD baseline notes.
- Validated CLI help exposure, Python compilation, shell syntax, and diff whitespace. No tests or model inference were run.
- Checked the Hana raw DiariZen manifest against `.data/evaluate/hana_tts/vad/report.json`: 14 raw turns exceed 15 s; the highest selected recursive cut-boundary probability is 0.0941166 (9.41%) at 45.744 s in raw turn 4 (23.6925–55.1325 s, spk00). The existing final Hana manifest predates this wiring and has no `long_segment_audit`; its maximum exported duration is 14.82 s.
