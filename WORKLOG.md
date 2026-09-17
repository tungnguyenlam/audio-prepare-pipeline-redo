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
