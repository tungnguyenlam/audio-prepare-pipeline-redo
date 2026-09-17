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
