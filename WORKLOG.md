## 2026-09-17 - Clean obsolete non-pipeline folders from .data

### Decisions
- Identified active pipeline directories: stage roots `s1-download`, `s3-diarize`, `s4-agent`, `asr`, utility roots `audio`, `cut`, `export_segments`, evaluation suites `evaluate`, frozen gold benchmark `tts_strategy/gold_benchmark_20260908`, and required runtime/model caches `huggingface`, `modelscope`, `3d-speaker`, `models`.
- Identified and purged obsolete folders no longer considered part of the pipeline:
  - Pre-rename legacy stage folders superseded by s1-s5: `download`, `separate`, `diarize`, `agent`, `diarization`.
  - Obsolete folders from removed legacy scripts / Studio UI / old crawler: `demucs`, `mel_roformer`, `test_mel_roformer`, `crawled`, `yt_crawler`, `audio_cutter`, `pipeline`, `studio`, `web_uploads`, `labeled_datasets`, `testing`, `new_video`, `notebook`, `archive`, `gemini_cache_probe`.
  - Historical experiment folders: `benchmark_v2`, `distillation`, `distillation_e2b`, `experiment_khanhvy`, `experiment_tab_sweep`, `verifier_experiments`, non-gold folders in `tts_strategy`.
  - Obsolete uv package build/wheel cache: `uv-cache`.
  - Stray root test JSON files: `test_bundle.json`, `test_export.json`.

### Results
- Reclaimed approximately 27.2 GB of disk space (reduced .data from 31 GB to 3.8 GB).
- Preserved all active pipeline artifacts, benchmark datasets, model weights, and sample test files.
- Verified workspace integrity with git status. No test suite was written or run.

## 2026-09-17 - Clarify DiariZen merge progress accounting

### Decisions

- Interpret the configured value of 100 as a maximum of 100 mean-adjustment retries after the initial merge evaluation, for 101 evaluations at most.
- Keep attempt/retry 0 as the initial pass and expose both the retry/evaluation limits and actual counts in progress output and the processed manifest.
- Separate direct merge output from post-long-segment-strategy output in live summaries so VAD expansion is not reported as additional merging.
- Preserve the detailed per-candidate `merge_audit`; add compact reason counts, VAD cut/rejection counts, and duration-filter counts to each recorded attempt.

### Results

- Renamed the internal limit to `MEAN_ADJUST_MAX_RETRIES` and added `max_retries`, `max_evaluations`, `retry_count`, and `evaluation_count` to `merge_mean_adjustment`.
- Expanded `MERGE_MEAN` and `MERGE_DONE` progress with retry/evaluation numbering, merge decisions, VAD activity, and duration-filter breakdowns.
- Updated the command cookbook and data contract to document the accounting.
- Validated Python compilation, DiariZen launcher shell syntax, CLI help wiring, stale-reference search, and diff whitespace. No tests or model inference were run.

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

## 2026-09-17 - Gate default VAD cuts at 0.1 activity

### Decisions
- Added `--vad-cut-threshold` (alias `--vad-threshold`), default `0.1`, across the shared long-segment policy, standalone VAD planner, exporter, and all diarization backends.
- A cut is eligible only when the lowest available VAD speech probability in the current oversized interval is strictly below the threshold. Intervals without an eligible boundary remain as explicit overlong rejections so the final duration filter removes them.
- Included the threshold in request parameters and standalone manifests so cache reuse cannot cross threshold settings.

### Results
- Updated README, command cookbook, data contract, and VAD baseline documentation with strict-threshold and rejection behavior.
- Rechecked Hana read-only: 14 raw turns exceed 15 seconds; all 14 are eligible at threshold 0.1, and the highest selected cut probability is 0.094116643.
- Validated Python compilation, launcher shell syntax, CLI help exposure, and diff whitespace. No tests or model inference were run.

## 2026-09-17 - Make default recursive VAD automatic with device fallback

### Decisions

- Set the shared Silero VAD selector default to auto across diarization, export, standalone VAD planning, and TTS planning; auto prefers cuda:0 and falls back to cpu when a GPU track is unavailable or fails.
- Keep explicit --vad-report and --vad-report-dir inputs authoritative, but lazily create or reuse content-addressed reports under .data/vad/auto/<source-sha256>.json when the default recursive VAD policy needs one.
- Reuse the shared recurrent Silero inference implementation for automatic reports and the silero_jit evaluator; cache the pinned model from every diarization worker setup target.

### Results

- Fixed the observed DiariZen failure: an oversized merged turn no longer fails solely because no VAD report flag was supplied; recursive cuts use the default 0.1 strict activity gate and threshold rejections still flow to the duration filter.
- Added automatic GPU-to-CPU retry reporting and auto track selection for completed reports.
- Updated README, command cookbook, and data contract documentation.
- Validated Python compilation, launcher and setup shell syntax, CLI help/default exposure, launcher help, and diff whitespace. No model inference or test suite was run.

## 2026-09-17 - Add parent-family diarization aggregates

### Decisions
- Keep per-result plots and child-family aggregates unchanged.
- For directory diarization, aggregate manifests at every output collection level from the child collection through the configured output root, so nested runs also receive a parent-family _plot/.
- Apply the same parent-and-child grouping to evaluate/plot_diarization --input-dir; retain the existing plot/ compatibility symlink behavior.

### Results
- Updated shared diarization aggregation, standalone plot evaluation, README, command cookbook, and data contract.
- The nested Hana layout now produces .../hana-playlist/9bu.../_plot/ and .../hana-playlist/_plot/, with the parent aggregate pooling all nested manifests.
- Validated Python compilation, plot CLI help, launcher shell syntax, and diff whitespace. No tests or model inference were run.

## 2026-09-18 - Make verifier analysis plotting script name explicit and update AGENTS.md

### Decisions
- Renamed generic `scripts/s4-agent/verifier/analyze.py` to `scripts/s4-agent/verifier/plot_verifier_analysis.py` so its name explicitly describes its function (analyzing verifier verdicts and rendering figures/reports).
- Added `scripts/s4-agent/verifier/plot_verifier_analysis.sh` as the canonical same-name Bash launcher.
- Maintained backward compatibility by updating `analysis.sh` and `analyze.sh` to forward arguments to `plot_verifier_analysis.py`.
- Added explicit naming guidelines to `AGENTS.md` under both `## Non-negotiable rules` and `## CLI and artifact contracts` prohibiting vague or overly general file names.
- Updated documentation across `README.md`, `docs/commands.md`, `docs/agent_verifier.md`, and `docs/data_contract.md`.

### Results
- Preserved file history via `git mv scripts/s4-agent/verifier/analyze.py scripts/s4-agent/verifier/plot_verifier_analysis.py`.
- Verified `--help` execution for `plot_verifier_analysis.sh`, `analysis.sh`, and `analyze.sh`.
- No tests were run in accordance with non-negotiable rules.

## 2026-09-18 - Automatic post-verification analysis and cost visualization

### Decisions
- Trigger `plot_verifier_analysis.sh` automatically at the end of any verifier run by default, resolving the output directory from the destination arguments or generated verdict pairs.
- Provide a `--skip-analysis` (alias `--no-analyze`) flag across all verifier CLIs via `scripts/_common/files.py` so post-verification analysis and plotting can be bypassed when desired.
- Verifiers run under `.venvs/verify` (which does not contain matplotlib), so the post-verification analysis is launched via subprocess using `scripts/s4-agent/verifier/plot_verifier_analysis.sh`, which activates `.venvs/audio` (which contains matplotlib).
- Extend `plot_verifier_analysis.py` to parse `_cost` and `_usage` from verdict artifacts and include individual sample costs (`cost_usd`, `input_cost_usd`, `output_cost_usd`) and token counts (`tokens_prompt`, `tokens_output`, `tokens_thinking`, `tokens_total`) in `all_samples.csv` and `successful_samples.csv`.
- Add `costs.png` visualization (4-panel figure) whenever cost information is present:
  1. Cost distribution histogram per sample with highlighted mean and median dashed lines.
  2. Cumulative cost progression curve across processed samples.
  3. Cost breakdown donut chart separating input vs. output token expenditure.
  4. Cost vs. audio duration scatter plot with an inset summary statistics card (total cost, mean/median, token counts, cost per audio minute).
- Add `costs` summary dictionary to `analysis.json` (totals, percentiles p25/p75/p95, mean, median, min, max, cost per audio minute, token counts) and append a formatted `## Cost analysis` section to `report.md`.
- Document the new behavior, flags, CSV fields, and plot artifacts in `docs/data_contract.md` (§7) and `docs/agent_verifier.md`.

### Results
- Validated standalone execution of `plot_verifier_analysis.sh` on `.data/evaluate/hana_tts/gemini_baseline`, confirming proper generation of `costs.png`, updated `analysis.json` (showing $0.5007 total USD, 285,916 total tokens across 55 samples), and `report.md`.
- Verified CLI `--help` flags for verifiers (`--skip-analysis`, `--no-analyze`).
- In accordance with non-negotiable rules, no tests were written or run.

## 2026-09-18 - Separate dead-simple cost plots and timestamp-ordered sample_costs.md

### Decisions
- Replaced combined multi-panel `costs.png` and removed cumulative cost curve completely per user feedback.
- Created separate dead-simple matplotlib plots:
  - `cost_distribution.png`: Clean single-panel histogram of cost per sample with mean and median dashed markers.
  - `cost_total.png`: Clean single-panel bar chart showing input, output, and total expenditure in USD with bar value labels.
- Added generation of `sample_costs.md` under `<output-dir>`:
  - Summary metrics at the top: total samples, pass/reject counts, pass rate, total and pass durations, total cost, component breakdown (input/output), cost per sample (mean, median, min, max), rate per audio minute, and token counts.
  - Markdown table ordered by timestamp (`start time`): `path`, `start time`, `end time`, `pass or not pass`, `transcripts`, `cost`.
  - Added clip filename timestamp regex extraction fallback (`_(\d{8,10})-(\d{8,10})_`) to ensure `start_s` and `end_s` are always populated and accurately sorted chronologically even when manifests are not co-located.
- Linked `sample_costs.md` from `report.md` and recorded it in `analysis.json`.
- Updated documentation in `docs/data_contract.md` (§7) and `docs/agent_verifier.md`.

### Results
- Executed `plot_verifier_analysis.sh` on `.data/evaluate/hana_tts/gemini_baseline`.
- Verified clean creation of `cost_distribution.png`, `cost_total.png`, and `sample_costs.md`, and confirmed stale `costs.png` was deleted.
- Verified timestamp ordering in `sample_costs.md` from 0.01s through 342.33s.
- No tests were written or executed in adherence with repository rules.
