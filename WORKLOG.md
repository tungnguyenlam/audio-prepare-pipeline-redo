## 2026-09-23 - Retry verifier runs on invalid JSON responses and fix sample_costs.md

### Decisions
- Added automatic retry when verifier runs receive model outputs that are not valid JSON objects or conformant verdicts (e.g. `invalid_json`, `json_not_object`, or schema validation errors):
  - In `scripts/s4-agent/verifier/_cli.py` (`verdict_processor`), if a verification attempt fails due to invalid JSON or schema error, it logs the item name, reason, and a snippet of the model's output using `logger.warning` and `progress("VERIFIER_RETRY", ...)`, then sleeps with exponential backoff and retries up to `--max-retry` times.
  - If retries are exhausted, the item failure artifact and raw response are written as before.
  - A valid verifier `reject` remains a successful completed verdict and is never retried.
  - Also added retry support to `GeminiVerifier.verify()` in `scripts/s4-agent/verifier/gemini.py` for direct callers.
- Removed the legacy mutually exclusive `--max-retries` flag in `scripts/s4-agent/gemini.py` (shared by both `scripts/s4-agent/gemini.sh` and `scripts/s4-agent/verifier/gemini.sh`). Only `--max-retry` (additional retries per transient failure) is retained, preserving backwards compatibility internally.
- Fixed `_pass_status` and summary calculation in `scripts/s4-agent/verifier/plot_verifier_analysis.py` for interrupted runs:
  - Previously, any sample without a verdict defaulted to "not pass", causing unrun samples (e.g., when a run is interrupted midway) to be falsely reported as "not pass".
  - Refined `_pass_status` to report `"not run"` for missing/unrun samples and `"fail"` for processing failures, reserving `"not pass"` solely for model rejections (`final_verdict == "reject"`).
  - Updated `sample_costs.md` summary metrics to distinguish evaluated samples from unrun/missing samples, computing pass rate against evaluated samples and calculating the $/minute cost rate using evaluated audio duration rather than unrun manifest duration.

### Results
- Verified `--help` on both `scripts/s4-agent/gemini.py` and `scripts/s4-agent/verifier/gemini.py`.
- Tested `_write_sample_costs_markdown` across multiple scenarios (normal complete run, interrupted run with missing samples, and processing error samples).
- Tested `verdict_processor` retries across 5 unit test cases (single invalid JSON retry-and-succeed, exhausted retries writing failed artifact, `--max-retry 0` disabling retry, schema validation retry, and valid `reject` avoiding retry). All 5 cases passed cleanly with log and stderr outputs.

## 2026-09-21 - Accept playlist URL lists in download playlist/channel

### Decisions
- `playlist.sh` / `channel.sh` now accept `-uf` / `--url-file` with one playlist or channel URL per line, matching `youtube.sh`. `--url` and `--url-file` are mutually exclusive; blank lines and `#` comments are skipped.
- `--limit` applies per URL, not across the file. Playlists are processed sequentially; each still resolves its own output group under `.data/s1-download/<playlist-or-channel>/`.
- Shared `read_url_file` / `resolve_url_or_url_file` helpers live in `youtube.py` so the three download commands use the same file contract.

### Results
- Validated launcher wiring with `playlist.sh -h`, `channel.sh -h`, missing-args, both-flags, missing-file, and empty/comment-only url-file cases. Did not download media or run a test suite.

## 2026-09-20 - Improve environment setup UX and launcher auto-provisioning

### Decisions
- Enhanced `scripts/_common/download_launcher.sh` and `scripts/_common/audio_launcher.sh` with interactive auto-provisioning: when an environment is missing in an interactive terminal, the launcher prompts the user `[Y/n]` to provision it on the spot and seamlessly proceeds with execution upon completion, rather than exiting with an error.
- Re-architected `envs/setup_worker_envs.sh`:
  - Replaced the confusing silent fallback of `TARGET="${1:-status}"`.
  - When invoked with no target interactively (`-t 0 && -t 1`), it renders current status and prompts with a 1-key menu (`core` by default, `download`, `audio`, `all`, `workers`, custom, or quit).
  - When invoked non-interactively without arguments, it displays the status report followed by a clear, actionable Quick Start guide.
  - Explicit `status` / `check` commands print the status report and helpful tips without prompting.
  - Robust argument parsing separates `--force` and `-h`/`--help` from target names.
- Optimized `status_report` to run environment and hardware inspections in parallel across background subshells, reducing status check latency from ~20 seconds down to ~2 seconds on AMD ROCm systems.
- Guarded JavaScript runtime installations (Deno, Node.js, Bun, QuickJS) in `envs/setup_download_env.sh` so already-installed executables are skipped unless `--force` / `--clean` / `--recreate` is specified.
- Updated 24 launcher scripts across `scripts/s2-separate`, `scripts/s3-diarize`, `scripts/s4-agent`, `scripts/speaker`, and `scripts/purity` to provide the direct, copy-pasteable provisioning command (`./envs/setup_worker_envs.sh <target>`) instead of vague pointers.
- Created symlink `~/Documents/audio-prepare-pipeline-redo -> ~/Documents/tts-data-pipeline/audio-prepare-pipeline-redo` for path compatibility across terminal sessions.

### Results
- Seamless CLI experience when running scripts on fresh clones: users can run `scripts/s1-download/youtube.sh` or `./envs/setup_worker_envs.sh` without friction or confusion.
- Fast status queries (~2s vs ~20s).
- All changes verified against local CLI commands (`youtube.sh`, `info.sh`, `setup_worker_envs.sh`); no test suite was written or run per guidelines.

## 2026-09-18 - Add Gemini Flex inference mode and confirm implicit prompt caching

### Decisions
- Audited Gemini cost drivers on `.data/evaluate/hana_tts/gemini_baseline` (59 clips, $0.50): run was `--inference-mode standard` (full price), 71.5% of cost was output where thinking tokens (87.9k) were ~12x answer tokens (7.5k), and `cached_input_tokens` was 0 on every sample because that run used an older 3,287-token prompt below the 4,096-token implicit-cache minimum for Gemini 3.x Flash.
- Current default `prompts/full-tags-prompt.md` is 6,269 tokens (`countTokens`), so implicit caching now applies to the shared prefix without any explicit `cachedContents`; explicit caching is not worth adding because audio is never shared and the prompt is only marginally above the threshold.
- Added `--inference-mode flex` to `s4-agent/gemini.sh` and `s4-agent/verifier/gemini.sh`: top-level `serviceTier: "flex"` on `generateContent` (probed; `generationConfig.serviceTier` is rejected with 400), priced as `paid_flex` at Batch rates. Flex is synchronous and returns 503 when capacity is short, so flex defaults to `--timeout-s 900 --max-retries 12` and exponential backoff is now capped at 60 s.
- Fixed a pre-existing verifier crash: `record_result(stats, verdict, error=...)` did not match the `success=`/`decision=` signature, so every verifier item was reported as `ITEM_FAIL` after its artifact was written and `TOTAL_COST` never counted decisions.

### Results
- Flex run on 3 Hana clips: all 3 `pass`, `usageMetadata.serviceTier="flex"`, `pricing_tier=paid_flex`, 4,007–4,019 of ~6.3k prompt tokens served from implicit cache; per-clip cost $0.0029–$0.0048 (baseline standard mode for the same 2.5 s clip: $0.0049 with no cache hits). Latency 19 s–414 s with 503 retries.
- Docs updated: `docs/agent_verifier.md` (flex row, caching note, `paid_flex`), `docs/commands.md`, `docs/data_contract.md`.
- Validated CLI wiring via a live 3-clip and 1-clip flex verifier run; no tests written. Remaining lever for cost is `--reasoning-effort low`, not caching.

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

## 2026-09-18 - Clickable file paths in VS Code preview for sample_costs.md

### Decisions
- In `sample_costs.md`, format file paths as clickable Markdown links pointing to the relative path from the markdown report to the actual audio file on disk.
- Enhanced `_markdown_relpath` and added `_resolve_target_file` with directory caching: if the stored path is stale or relocated, search adjacent evaluation directories to find the real existing file so links never result in 404/file not found in VS Code preview.
- Links in `sample_costs.md` display the clean project-relative path while linking via POSIX relative path (`../../diarizen/...`), ensuring VS Code Markdown Preview navigates directly to the audio clip on click.

### Results
- Regenerated `sample_costs.md` and confirmed all audio links point to existing files on disk.
- No tests were written or run.

## 2026-09-21 - Stop passing removed RoFormer session `backend`

### Decisions
- Current git `melband-roformer-infer` / `bs-roformer-infer` dropped the experimental MLX `backend=` constructor argument (Torch CPU/CUDA only). The pipeline still passed `backend=None` on every Mel-RoFormer and BS-RoFormer session, which crashes default invocations: `TypeError: MelBandRoformerSession.__init__() got an unexpected keyword argument 'backend'`.
- Remove `--backend` from both commands and stop recording it in sidecar `parameters`. Device selection remains `--device` (`auto` maps to the package default).

### Results
- `mel_roformer.py` and `bs_roformer.py` construct sessions with `model_name` and `device` only.
- Documented in `docs/commands.md`. Validated `py_compile`, launcher `bash -n`, and `--help` (no `--backend`). No model inference or tests were run.
- Existing sidecars that stored `"backend": null` will no longer cache-match and need `--overwrite` to replace.


## 2026-09-21 - Simplify Gemini mode selection and add opt-in prompt caching

### Plan and decisions
- Share Gemini CLI settings and dispatch between raw generation and verification; replace the ignored inference_mode constructor keyword and separate service_tier setting with one explicit mode.
- Preserve standard, flex, and resumable batch inference, with batch as the CLI default.
- Add --cache-prompt (false by default), cache text only, and account for cached reads plus the full configured cache storage lifetime once per cache.
- Keep provider generation separate from verdict parsing. Review official Google REST caching, Flex, and pricing documentation before changing payloads and rates.
- Validate syntax and CLI/launcher wiring only; no tests or paid inference were requested.

### Results
- Both commands now share Gemini arguments and provider dispatch, with one explicit inference_mode constructor setting and matching Batch defaults. Removed ignored constructor kwargs and the separate service_tier control. Standard/Flex use the documented REST service_tier field; returned tier metadata drives synchronous cost estimates.
- Replaced duplicate requests/urllib transports with the existing httpx[socks] dependency and one retry loop. Unified -b/-bs/--batch-size defaults at 10.
- Added --cache-prompt (false by default) and --cache-ttl-s (3600 synchronous; 90000 Batch minimum). Text/system instructions are cached once per live client cache, audio remains inline, concurrent creation is locked, and Batch creates caches only for new submissions. Batch reserves a full 24-hour queue window before reusing a cache.
- Cost accounting separates cached reads, uncached input, output/thinking, and full-TTL cache storage. Storage is assigned once to the first priced response, reported in run totals, and recorded in Batch state for resume. Unknown cache rates/counts are explicitly unpriced. Updated CSV, JSON, Markdown, and chart cost breakdowns to include storage.
- Updated focused command, verifier, and artifact-contract docs. Google REST Flex/caching/pricing references are linked in docs/agent_verifier.md; prices are dated 2026-09-21.
- Validation: Python compilation passed for all five changed Python files; Gemini launcher bash syntax and --help passed; analysis launcher --help passed; existing verify environment imports httpx; git diff --check passed. Reviewed the final diff and intended file list.
- No tests were written or run, no packages were installed, and no paid inference/cache API calls were made. Live provider behavior, numerical execution, and rendered plot output remain unverified because tests/live inference were not requested.
- Compatibility: constructor callers must use inference_mode and choose the corresponding generate/generate_batch method. New cache settings change verifier artifact identity; old outputs may require --overwrite or a new output directory. Caches expire at their TTL and are not shared across fresh invocations.

## 2026-09-21 - Explicit Gemini Batch continuation

### Decisions
- Add `--continue` to both Gemini commands, but accept it only for Batch mode because Batch exposes recoverable job IDs while Standard/Flex responses are synchronous and cannot be reconstructed after local interruption.
- Batch continuation now requires an exact saved run match. The persisted signature covers the input directory root, relative audio paths, and source SHA-256 digests, alongside prompt/model/generation settings. A changed directory returns an error before submission.
- Do not implicitly reuse Batch state. Without `--continue`, a new Batch job is submitted; this makes the cost-sensitive behavior explicit. Verifier continuation computes its signature from the complete input set even when completed verdict artifacts are filtered from processing.

### Results
- Added `--continue`, input signature persistence, prior-state mismatch detection, and full-set verifier state lookup. Updated focused command, data-contract, and verifier documentation.
- Continuation also migrates compatible Batch state written by the previous Gemini implementation, so an interrupted run from before this flag was added can be resumed without resubmission. Incomplete verifier text artifacts are safely republished from saved Batch responses.
- Validation: Python compilation passed for Gemini and verifier modules, both launcher syntax checks passed, both `--help` outputs expose `--continue`, and `git diff --check` passed. Tests remain omitted per repository instructions; paid Gemini calls were not made.

## 2026-09-22 - Resolve automatic verifier plots beside run verdicts

### Decisions and results
- Found that shared automatic analysis preferred `_default_base` over actual verdict destinations, placing `plot/` above the family directory and aggregating unrelated families. Explicit `--output-file` was also shadowed by the default base or `--output-dir`.
- Resolve explicit output files first, then explicit output directories; reuse production `resolve_output_dir` for directory inputs and actual destination parents for single-file defaults. Keep directory roots stable for nested inputs and retain the original destination list for cached reruns. Applies to all nine verifier backends through their shared runner; raw agent commands do not automatically plot.
- Updated verifier documentation and the artifact contract. Existing plots were not moved or regenerated.
- Validation: Python compilation, all verifier Bash launcher syntax checks, and git diff whitespace checks passed. Launcher --help passed for Gemini, HF, endpoint, MOSS, MiniCPM, Kimi, VibeVoice, plot_verifier_analysis, analysis, and analyze.
- vLLM and Unsloth --help encounter an existing EndpointVerifier import collision with the raw agent endpoint module, outside this path-resolution fix. Their unchanged imports fail before command parsing.
- No tests were written or run as requested by repository instructions. No model calls or rendered-output validation were performed.

## 2026-09-22 - Continue Gemini Standard, Flex, and Batch runs

### Decisions
- Expose --continue on both Gemini commands for every inference mode; preserve matching completed artifacts, retry incomplete/failed synchronous work, and reject mismatched metadata unless --overwrite is explicit.
- Make --continue and --overwrite mutually exclusive. Batch continuation must retain the original submitted subset and validate full input/settings identity before reconnecting to saved jobs.
- Repair raw Gemini output preflight and verifier response-path resolution as required for reliable completion detection. Keep endpoint/HF CLI scope unchanged.
- Validate syntax, launcher help, and the final diff only; no tests or paid inference.

### Results
- Added --continue to raw Gemini and Gemini verifier launchers in Standard, Flex, and Batch modes. Standard/Flex skip matching complete outputs and regenerate missing, failed, or incomplete pairs; conflicting metadata requires --overwrite.
- Added raw Gemini metadata/digest preflight and fixed verifier repository-relative response lookup. Verifier completion now requires an actual matching raw response, including legacy artifacts; other verifier backends share that integrity check but do not gain a new CLI flag.
- Batch CLI reuse is now explicit. Run manifests retain the original pending subset and validate full input/settings identity; continuation finds the latest matching saved job state, retries failed jobs without resubmitting successful jobs, and refuses conflicting state. Python generate_batch retains its existing reuse_state default.
- Batch per-request errors and invalid verifier responses remain saved responses on continuation; obtaining fresh responses for those clips requires --overwrite. Standard/Flex cannot recover responses interrupted before publication, so retries may incur another charge.
- Updated command examples and continuation contracts, including the actual Standard default. Reviewed every changed file and the final diff.
- Validation passed: Python compilation for all three changed modules, Bash syntax for both Gemini launchers, both launcher --help outputs exposing --continue, and git diff --check. No tests were written or run, and no paid model calls were made; runtime recovery remains untested per repository instructions.

## 2026-09-23 - Confirm Gemini verifier Standard continuation

- Requested Standard-mode `--continue` support is already implemented in commit `d95f0ae` and present on the tracked upstream branch. No code or documentation changes were needed: shared Gemini arguments accept the flag for all modes, synchronous dispatch uses the existing generation path, and verifier preflight skips matching complete artifacts while retrying matching failed/incomplete work.
- Validated the verifier Bash launcher syntax and its `--inference-mode standard --continue --help` invocation; help exposes both options and Standard remains the default. Reviewed existing continuation documentation and command examples.
- No tests were written or run, and no paid model calls were made. Runtime recovery was not exercised because tests were not requested.

## 2026-09-23 - Plan verifier run handoff for nontechnical recipients

- Inspected verifier artifact/analysis documentation, shared verdict validation and completion paths, dataset indexing/bundling, and the bundle launcher.
- Added docs/plans/verifier_handoff_export_plan.md proposing a standalone verifier-aware ZIP export with accepted audio, offline listening/search page, CSV catalog, instructions, summary, exclusions and portable provenance/checksums.
- Defined explicit completeness evidence, partial delivery labeling, source integrity checks, configuration conflict handling, and metadata exceptions across verifier profiles. Existing bundle/index commands do not implement this handoff.
- Planning only: no implementation, media export, model calls or tests. CLI/launcher changes and runtime validation are not applicable; reviewed the plan and checked the final diff for whitespace errors.

## 2026-09-23 - Implement verifier audio/transcript review handoff

### Decisions and results
- Updated the handoff plan for an actual human review package: offline HTML listening/search/editing, matching CSV and XLSX catalogs with relative audio paths, original transcripts and separate reviewer status/corrections/notes. HTML saves/loads release-specific progress JSON and downloads review CSV; spreadsheet edits are a separate workflow.
- Added standalone export_verifier_handoff.py/.sh and its HTML template. XLSX uses stdlib ZIP/XML with inline text, relative audio hyperlinks, frozen/filterable headers and a review-status dropdown; no packages installed. Copy original audio bytes into passed/needs_attention folders and publish a versioned ZIP atomically under .data/ with instructions, summary, attention catalog, provenance and checksums.
- Separate verifier completion from pending human review. Explicit expected indexed/exported-segment inventories prove coverage; missing/failed/invalid/uncertain/incomplete processing requires remediation or --allow-partial. Partial delivery is labeled visibly, rejected clips remain available for review, and absent transcripts require manual transcription. Missing/changed passed audio, duplicates, mixed settings and conflicting inventories block export. No inference or automatic pipeline chaining.
- Extracted existing verifier discovery and response integrity into _verifier_artifacts.py for current production/analysis/export consumers; reuse production schema validation across verifier backends. Preserved analysis walk-error handling and production response integrity semantics.
- Renamed index -> index_audio_manifest, filter -> filter_audio_manifest, export -> export_manifest_table and bundle -> bundle_manifest_audio (Python and Bash). Removed ambiguous old entrypoints without aliases and updated active README/command/verifier/data-contract documentation.

### Validation
- Python syntax compilation passed for all five export commands and the three changed verifier modules. Bash syntax and --help passed for all five s5-export launchers. Verifier analysis and Gemini launcher --help passed. Embedded browser JavaScript syntax passed node --check.
- Reviewed new implementation, template, shared helper changes and final diff; checked renamed-command references and git diff whitespace. Included intended new files.
- No tests were written or run per repository instructions. No model calls, media exports or package installation. End-to-end export, archive relocation, browser playback/save/load and Excel opening remain unverified; syntax/help checks do not establish runtime acceptance.

## 2026-09-23 - Simplify verifier handoff invocation after operator feedback

- Made --output-file optional with a visible .data/s5-export/<manifest-family>_<version>.zip default; keep explicit output paths exact and provide concrete path/collision guidance. Dataset names derive from the first inventory or input directory.
- Apply the expected inventory selection before configuration grouping, ignoring unrelated source clips in a broad model directory. Allow differing settings when each selected clip has one result, recording all configuration hashes and each clip's configuration. Competing results now list folder/hash choices and support an explicit --configuration selector; never automatically pick a verdict or drop missing expected inputs from coverage.
- Removed handoff audio decoding dependency: use finite inventory duration metadata, leave missing durations blank, and label known hours/unknown counts. The dedicated launcher uses existing Python and never invokes provisioning; this avoids the reported missing envs/requirements-audio.txt path without changing unrelated audio setup or installing packages.
- Improved incomplete-run errors with coverage/outcome/unresolved counts. Updated command examples, verifier instructions, data contract and plan to reflect the simpler workflow.
- Validation: Python syntax, Bash syntax, normal launcher --help and --help with system Python override passed; embedded JavaScript syntax and final whitespace checks passed. Reviewed final diff. No tests, package installation, inference or actual exports were run; the operator's remote run was not available here for runtime verification.

## 2026-09-23 - Fix Unsloth and vLLM EndpointVerifier import

### Decisions and results
- `unsloth.sh -h` failed because `unsloth.py` and `vllm.py` searched `scripts/s4-agent` before the verifier directory, so `endpoint` resolved to the raw agent module. That module exposes `EndpointAgent` and `send_http_request`, while `EndpointVerifier` is defined in `verifier/endpoint.py`.
- Load the raw agent by file path under a separate module name so the verifier module can be imported as `endpoint` without binding that name to itself. Search the verifier directory first in the Unsloth and vLLM commands. Unsloth still uses the agent `send_http_request` through the verifier re-export; vLLM server mode uses the same `EndpointVerifier`.
- Validation: Python compilation passed for the verifier endpoint, Unsloth, and vLLM modules. Bash syntax and `--help` passed for `unsloth`, `vllm`, verifier `endpoint`, and raw `endpoint` launchers. Import check confirmed `UnslothVerifier` subclasses `EndpointVerifier`, which subclasses `EndpointAgent`. `git diff --check` passed. No tests were written or run. No model or endpoint calls were made.

## 2026-09-23 - Review attached Gemini replacement files

- Compared both attachments against the raw Gemini agent and verifier, including callers, shared prompt/artifact/continuation helpers, launcher paths, dependency declarations and documented CLI examples. No implementation files replaced.
- The small attached verifier is byte-identical to the existing verifier. The larger raw agent is not a behavior-compatible replacement: Standard generation switches from REST to google-genai, required in all modes but absent from verifier requirements; raw prompt CLI flags are removed in favor of GEMINI_SYSTEM_PROMPT/default system instructions; Standard explicit caching is rejected; metadata preflight and more descriptive errors are added.
- The unchanged verifier CLI still passes its prompt through generation_callback, now as a system instruction, while GeminiVerifier.verify retains an incompatible positional generate call. GEMINI_SYSTEM_PROMPT only controls the raw CLI. Verifier continuation metadata does not distinguish old user-prompt behavior from new system-prompt behavior; raw metadata and Batch manifests change and old continuation state can conflict.
- Validation: all four source files parsed successfully with Python AST; both existing Bash launchers passed syntax checks and their forwarding/path wiring was inspected. No tests, inference, package installation or attachment execution. SDK/provider behavior and runtime compatibility remain unverified. Reviewed the final worklog diff.

## 2026-09-23 - Explain Gemini cache cost accounting

- Confirmed current and attached Gemini implementations use Google response token counts with local pricing estimates, not returned billed dollar amounts. Cached input uses generation usageMetadata.cachedContentTokenCount; storage uses cache creation usageMetadata.totalTokenCount multiplied by requested TTL hours and the local model/mode storage rate, booked once when created.
- Both use the existing _gemini_pricing.py helper. Storage assumes the full requested TTL; no billing reconciliation is performed. Read-only source inspection only; no code/launcher changes, tests or API calls were needed.

## 2026-09-23 - Recover local progress and integrate remote Gemini generation

- Merged origin/main c6b2ebc with all eight local commits retained; remote changed only raw gemini.py, while verifier gemini.py was identical. Preserved the remote SDK Standard path, system instructions, preflight, REST Flex/Batch, and cache fallback.
- Added --max-retry N to both Gemini commands for N additional transient HTTP failure retries; zero disables retries. Kept legacy --max-retries total-attempt semantics mutually exclusive and disabled nested SDK retries. Valid verifier rejections and invalid response parsing never trigger transport retries. Batch result resubmission remains governed by existing continuation/overwrite behavior.
- Fixed direct verifier generation to use keyword-only system_prompt, distinguished new system-prompt verifier metadata, declared the SDK dependency, and corrected affected prompt/cache/retry documentation. No packages installed.
- Validation: Python AST syntax, both Bash launcher syntax checks and both launcher --help invocations passed with system Python; reviewed the merged diff, retained local file inventory, and whitespace checks. Verified SDK retry/ThinkingLevel definitions in upstream v1.56.0 and set that dependency floor. The SDK is absent locally; no installation, tests or paid inference were performed, so runtime provider behavior remains unverified.

## 2026-09-23 - Ground full-tags pronunciation and disfluencies in audio

- Request: revise the working-copy full-tags prompt without word-specific examples, preserving heard pronunciation across languages, fillers, and unexpected pauses. Existing uncommitted prompt edits are the starting point; unrelated local files are outside this task.
- Inspected Gemini raw/verifier prompt loading, adjacent endpoint/HF backends, shared verdict validation, launchers, and documented contracts. Keep the JSON fields, verifier profile, and provider paths; no inference implementation changes.
- Decisions: require audible evidence for native British/American English IPA; write other foreign pronunciations directly as open ViePhoneme, without IPA conversion tables, forced Vietnamese rhymes, dropped codas, or automatic tones. Permit transcribable multilingual speech; retain unsupported_language only for language content that cannot be reliably transcribed. Distinguish real fillers and mid-sentence silences from lexical normalization and editorial punctuation.
- Result: shortened the working prompt from 4,098 to 3,136 whitespace-delimited words (214 to 141 logical lines), removed all word-specific examples and lossy conversion/rhyme/tone rules, and retained the response schema, tags, and emotion catalog. Updated focused verifier documentation and corrected the default-versus-legacy emotion-field description in the data contract.
- Validation: reviewed the prompt against both the pre-edit working copy and HEAD, reviewed documentation diffs, and passed git diff --check and Bash syntax checks. Both Gemini launchers passed --help with the existing Miniforge Python via VERIFIER_PYTHON; system Python and the default verify environment lack httpx, so their help attempts failed before argument parsing. No packages installed. Static inspection confirmed both Gemini paths load the prompt as system instructions and the validator still recognizes its acoustic_defect_v3 profile.
- No tests were written or run because the user did not request them. No model/API calls or audio evaluation were performed; actual transcription/phonetic accuracy remains unverified. Only the prompt, two affected documentation files, and this worklog are included; unrelated pre-existing untracked files are preserved.

## 2026-09-23 - Audit Gemini 3.8 API requests and empty-response retries

- Inspected raw/verifier Gemini generation, adjacent endpoint, parsing/validation, artifact/continuation helpers, launchers, dependencies and contracts; preserve pre-existing prompt/export/doc edits.
- Local saved Gemini artifacts: 506 successful pass verdicts and 5 invalid_json failures, all five with empty text; existing failure artifacts omit provider finish/block diagnostics, so their upstream cause cannot be recovered.
- Google's Gemini 3.8 migration checklist explicitly requires nonempty text in the final user turn; current requests contain audio only. Keep the system rubric and add a visible, configurable user instruction consistently across SDK Standard and REST Flex/Batch.
- Keep bounded retries for transient failures; add a separate bounded response retry budget for empty/transient incomplete answers, preserve attempt diagnostics and usage, and report terminal blocks/token exhaustion as generation failures. Valid acoustic reject verdicts must remain completed verdicts.
- Implemented nonempty configurable --user-prompt across SDK/REST payloads; explicitly selected Developer API/v1beta for Standard. Model, thinking and output-token settings already match the 3.8 guide; no sampler/SDK migration needed. Installed system google-genai is 2.18.1 and exposes the used configuration and response-error fields.
- Added --max-response-retries (default 3 additional Standard/Flex generations), separate from transport attempts, plus Retry-After handling. Preserve successful and failed provider evidence in raw/verifier artifacts; retain every received retry response and aggregate usage/cost without double counting the session. Never retry a valid acoustic reject, permanent provider block, or MAX_TOKENS; nonempty JSON/schema errors and individual Batch resubmission remain explicit exceptions.
- Corrected docs that assumed Standard guarantees AI Studio parity. Added comparison guidance and new artifact/CLI contracts. Existing outputs need a fresh directory or explicit overwrite because user text and response retry settings change run identity.
- Validation: changed Python files passed AST syntax checks; both Bash launchers passed bash -n and --help through system Python. Inspected installed SDK definitions without creating a client. Reviewed implementation/documentation diff and whitespace. No tests written or run, no package installation and no model/API calls; runtime retry behavior and AI Studio quality equivalence remain unverified.
- Publication: implementation committed locally; push to origin was blocked because this environment has no GitHub HTTPS credentials (no credential helper, gh CLI, GH_TOKEN or GITHUB_TOKEN). Existing unrelated staged/working edits were excluded using an isolated commit index.

## 2026-09-23 - Full-tags prompt delivery status

- Prompt, focused documentation, and task results are committed locally. Push to origin/main is blocked because HTTPS GitHub authentication is unavailable in this session; GitHub CLI is not installed, and strict SSH host verification cannot establish the alternative connection. No credentials were read, recorded, or changed. Remote delivery remains pending authentication.

## 2026-09-23 - Diagnose Gemini launcher missing httpx

- The default Gemini launchers select .venvs/verify/bin/python even when Conda base is active. On this machine that venv lacks both httpx and google-genai; /home/hault16/miniforge3/bin/python already has httpx and google-genai 2.18.1. Requirements already declare both dependencies, so no source/dependency changes or package installation are needed.
- Confirmed the existing VERIFIER_PYTHON override selects the usable Miniforge environment for both raw and verifier commands. Documented the override and existing dedicated-env provisioning alternative. The reported input path also requires quotes around its space-containing directory.
- Validation: dependency imports in Miniforge, both Gemini launcher --help invocations with the override, and Bash syntax passed. Reviewed the focused documentation/worklog diff. No tests, model/API calls, package installation, or changes to the user's .gitignore/prompt/untracked files.

## 2026-09-23 - Persist partial verifier sample cost reports

- Inspected shared verifier execution, Gemini/endpoint/HF and other backend call sites, artifact discovery, report rendering, launchers and file contracts. Existing reporting ran only after plot rendering at shutdown.
- Update plot/sample_costs.md after every saved verdict through the shared verifier path, including failures and --skip-analysis. Seed from existing artifacts for continuation and replace rows by artifact path to avoid duplicate samples; this is a cumulative current-artifact report, not overwritten-attempt billing history.
- Reuse production analysis extraction/rendering without plotting or provider calls; serialize worker updates and publish Markdown atomically. Standalone analysis now publishes sample costs before plotting. Document partial-run and in-flight limitations.
- Validation: changed Python files passed AST syntax checks; Gemini and analysis Bash launchers passed bash -n and --help using the existing Miniforge Python. Reviewed the final focused diff and git diff --check. All verifier backends use the updated shared runner. No tests were written/run (not requested), no models/API calls, package installation or runtime dataset modifications; interrupted/concurrent execution was reviewed statically only.
- Publication: committed the five intended source/documentation/worklog files, excluding unrelated existing changes. Push failed because GitHub HTTPS credentials are unavailable (could not read Username); remote delivery remains pending authentication.

## 2026-09-23 - Restore compact Gemini verifier JSON

- Compared the requested assistant-response reference with the Nga Levi artifact: the extra top-level generation object embeds provider candidates/thoughtSignature and duplicates transcript, usage/cost and model metadata already represented by the text sidecar/verdict.
- Removed full generation attachment from parsed Gemini verdicts and provider-body serialization from the shared verifier writer (Gemini was its only producer). Successful artifacts retain the reference layout; failures keep error/invalid_verdict and exact final text without a provider dump. Current generation parameters remain for continuation identity.
- Generation exceptions still carry in-memory information for safe error classification and run cost reporting; raw generation, retry behavior and aggregate verdict usage/cost are unchanged. Updated focused artifact documentation. Existing runtime outputs are not rewritten.
- Validation: Python AST syntax passed for both changed modules; raw/verifier Gemini launchers passed bash -n and --help through Miniforge. Reviewed final source/documentation diff and git diff --check. No tests written/run (not requested), no inference/API calls and no runtime artifacts modified.

## 2026-09-23 - Preserve audible contrasts in ViePhoneme

- Follow-up: user reports persistent loss of fricative distinctions, incorrect onset substitution, and flattened vowel/glide transitions in foreign names. Treat the supplied corrections as user reference, not independently verified listening; do not hardcode the names or corrected words into the prompt.
- Reviewed the working prompt, shared schema/loading, adjacent endpoint/HF paths, Gemini launchers and focused documentation. Existing user change renaming section 2.5 to Non-verbal is preserved; .gitignore and runtime artifacts are outside the edit scope. Previously found local example paths disappeared before they could be read, so no saved-prompt or acoustic verification is claimed.
- Decision: distinguish word recognition, acoustic discrimination and spelling the heard sounds. Add a small sound-level notation convention for fricatives and glides, then compare ambiguous candidates by the audible feature that separates them. Keep the IPA eligibility, JSON schema, filler and pause contracts intact.
- Result: refined only the pronunciation listening/representation sections and their documentation. The prompt now separates consonant manner/place/voicing, vowel trajectories, glides and syllable count; compares ambiguous readings using distinguishing audible cues; defines reusable extended-letter sound notation; and checks the bracketed sounds independently of the recognized word. No word-specific pronunciation examples or automatic substitutions were added.
- Validation: reviewed the final focused diff and passed whitespace checks, both Gemini launcher Bash syntax checks, and both --help invocations using existing Miniforge Python. No tests were written or run because none were requested; no model/API calls, audio evaluation, installation, or runtime artifact changes. The expected improvement is unmeasured.
- Delivery: stage only this task's prompt changes, focused documentation and worklog; retain the user's section-heading edit and .gitignore change in the working tree. Push preflight failed because HTTPS GitHub authentication is unavailable; remote publication remains pending authentication.

## 2026-09-23 - Sound-first foreign-word transcription prompt

- Follow-up: Gemini 3.8 Flash still wrote brackets from spelling or familiar Vietnamese syllables (onset changed, back fricative merged into front, diphthong/glide flattened) across non-English names. User corrections are treated as reference, not verified listening; no word-specific examples were added.
- Decision: replace accumulated rules with one ordered procedure ("sound first, spelling after"): describe each occurrence as meaningless sound, write it, then attach spelling; check three named pulls (Vietnamese reading of spelling, remembered/dictionary/common Vietnamized pronunciation, snapping to nearest Vietnamese syllable). ViePhoneme explicitly need not be a valid Vietnamese syllable. IPA decision now names both error directions (false IPA, false downgrade) and is limited to native American English. Brackets are decided before the final transcript is written. Punctuation restated as rhythm, not grammar; filler/pause contracts unchanged.
- Validation: reviewed diff only. No tests (not requested), no model/API calls; improvement unmeasured.

## 2026-09-23 - One-sound consonant letters in ViePhoneme

- Follow-up run of the sound-first prompt still merged back fricatives into `s`, wrote velar `gh`/`d` for a front voiced onset, replaced a foreign affricate coda with a Vietnamese stop coda plus tone, dropped an inter-syllable glide, misheard a sentence-final particle as a question and returned an empty `reason`.
- Diagnosis: Vietnamese letters `s`, `d`, `gi`, `tr` have dialect-dependent or merged readings, so the model could treat Vietnamese-style spellings as faithful; the coda rule only listed some Latin codas.
- Change: ViePhoneme consonant table with one sound per letter (and `d`/`gi`/`tr` banned for foreign sounds), explicit contrast decision for every fricative/affricate, general rule for codas outside the Vietnamese set, glide check after diphthongs, sentence-final particle and `?` by audible tone/intonation, non-empty `reason`. No word-specific examples.
- Validation: diff review only; no tests (not requested), no model/API calls.

## 2026-09-23 - Ear-only foreign-word brackets

- Follow-up run still produced brackets from spelling/romanization or remembered native readings (velar onset for a front fricative, back fricative merged into `s`, diphthong and glide flattened) and missed audible frication after a coda. User notes speakers mix reading styles and languages freely; high reasoning was worse than medium, so medium stays the default.
- Diagnosis: the sound-first procedure named the "correct/dictionary/common" pronunciations as pulls to check and asked for IPA-like feature analysis, which invites the model to recall exactly those readings; longer text reasoning drifts further toward knowledge.
- Change: section 3.1 is now "ear only" — no language identification, spelling, romanization, dictionary/IPA or intermediate transcription when writing brackets; knowledge-style reasoning is flagged as a signal to re-listen. Native American English comparison is confined to the IPA decision for English words. ViePhoneme is framed as a script to mimic the speaker; the consonant table became a letter reading key; post-coda frication is kept as a consonant block. No word-specific examples.
- Validation: diff review only; no tests (not requested), no model/API calls.

## 2026-09-23 - Why six prompt edits did not move the pronunciation errors

- Checked delivery: every verifier metadata JSON stores the full system prompt it sent. The 22:08 medium run used the current file byte-for-byte (minus trailing newline), so edits were loaded; nothing stale was served from cache.
- Outputs are overwritten per `<model>/<reasoning>/<family>` directory, so earlier prompt versions' outputs for the same clip are lost; comparisons between versions relied on one clip and one sample each.
- Across versions and reasoning levels the same clip kept `ghen-zi`/`si-ki-bu` while only the diphthong moved (`hê-an` → `hây-an`). High reasoning spent 3–16k thinking tokens with no gain. Likely limit: reasoning operates on the already-encoded audio, so "listen again"/"back-check" instructions cannot add acoustic detail; text reasoning pulls toward priors. Prompt text alone may not fix these errors.
- Change: IPA decision made symmetric (native-sounding English words must get IPA; removed "uncertain → ViePhoneme" default that biased toward ViePhoneme). Emotion is decided from the voice before transcription and follows the voice when content differs; content never selects a label.
- Validation: diff review only; no tests (not requested), no model/API calls.

## 2026-09-23 - IPA chosen first, by phonemes

- The 23:36–23:41 run used the current prompt and still downgraded English words the user heard as correctly pronounced to ViePhoneme.
- Diagnosis: the ear-only section preceded the IPA decision and applied to every bracket ("meaningless sound", no language identification, no known IPA), so the model wrote ViePhoneme first; "sounds like a native speaker" let any Vietnamese voice quality disqualify IPA.
- Change: 3.1 now chooses the system first. English words matching American English phonemes, codas/clusters, syllable count and stress must get IPA; voice quality, rate and surrounding Vietnamese intonation do not count. ViePhoneme requires a named Vietnamese-style cue. The ear-only rule moved to 3.2 and applies only to writing ViePhoneme.
- Validation: diff review only; no tests (not requested), no model/API calls.

## 2026-09-24 - Filler sweep, breath pauses, focused re-listen

- Across 614 saved Gemini transcripts only 8 contain any tag and 167 any `~`; the user reports missed breath-catch pauses and missed clear (`ừm`, `ừ`) and vague (`<hm>`, `<mm>`) fillers.
- Causes in the prompt: pauses were defined as "real silence", so breath intakes could not carry `~`; "not sure there is a sound → add nothing" applied to fillers too.
- Change: a pause is silence or breath intake (`~` mid-flow, `,` at a phrasing boundary; breathing still untagged); a dedicated word-boundary sweep for fillers after transcription; vague fillers must be written as tags rather than dropped. Bracket decisions now require a focused second listen to the word's own segment, syllable by syllable, including aspiration, with ties resolved by the distinguishing cue rather than the spelling/standard/Vietnamized default. This supersedes the earlier "long reasoning does not help, write the first impression" line; the re-listen is targeted, not open-ended.
- Validation: diff review only; no tests (not requested), no model/API calls.

## 2026-09-24 - Short s5-export command names

- User chose the short names `index`, `filter`, `export`, `bundle` (byte-identical copies already existed untracked); removed the long-named Python/Bash entrypoints and updated README and `docs/commands.md`. This reverses the earlier long-name rename.
- Also committed pending local files: per-backend `requirements-*.txt`, `.python-version`, `playlist-url.txt`, and the `.gitignore` change (root `/.data` ignore).
- Validation: `bash -n`, `py_compile`, and `--help` through each renamed launcher; no tests (not requested).

## 2026-09-24 - Diagnose and fix Lam sync configuration

- Inspected all four new Lam launchers, adjacent host launchers, legacy server launcher, and command documentation. Lam scripts copied override names from Loi, Server, and Anhnct, so `SYNC_LAM_HOST`/`SYNC_LAM_REPO` were ignored; code launchers also dropped rsync arguments.
- Changed all Lam launchers to use `SYNC_LAM_HOST`/`SYNC_LAM_REPO`; code launchers now forward arguments, including `--dry-run`. Documented defaults, separate code/data commands, destination prerequisites, and connectivity diagnosis.
- Read-only SSH diagnosis to the default `hault16@10.148.0.90` with BatchMode and a 10-second connection timeout failed with port 22 connection timeout. Local routing sends this private address through the Wi-Fi default gateway. No sync override variables were set in this session. This connection failure precedes authentication/rsync; remote path, permissions, and rsync availability remain unverified.
- Validation: Bash syntax checks passed for all four launchers; reviewed argument wiring and final diff; `git diff --check` passed. No tests were written or run (not requested), no paid models invoked, and no files transferred to Lam. Left the existing untracked `data` symlink untouched.

## 2026-09-25 - Recalculate s4 verifier statistics and costs

- Added Vietnamese `statistíc.md` from 1,633 verifier artifacts and 17 matching manifests: 1,626 pass, 2 reject, 5 processing failures; 13,578.94 seconds input and 13,537.24 seconds pass. Explained diarization gaps, merge recovery and 85 short clips removed (77.88 seconds), with mean/min/max and per-source breakdowns.
- Decimal cost totals: USD 16.254857750 in verdicts, separate pending batch cache USD 0.104487500, and user-confirmed separate testing USD 7.400700000: USD 23.760045250 estimated total. Documented billing uncertainty and unavailable testing breakdown. Existing costs match the stale 1,348-row CSV; 285 new artifacts add USD 2.686015375.
- Verified official pricing; thinking accounts for 95.9374% of billed output tokens. Audited verdicts with the shared validator, response hashes, manifest source metadata, report links and arithmetic; read relevant runners, launchers, helpers and docs. No tests (not requested), paid model calls or listening-based accuracy claims. Unrelated changes left untouched.

## 2026-09-25 - Add management summary of verifier results

- Added `bao_cao_tom_tat_verifier.md`, a concise Vietnamese summary of the detailed report covering production yield, mean/min/max durations, silence versus short-clip losses, thinking and cache economics, estimated costs including separate testing, and all reject/processing-failure categories.
- Preserved the distinction between schema consistency and listening-based accuracy, and between recorded estimates and final billing. Reviewed figures and links against `statistíc.md`; no tests requested or run, and no paid model calls.
- Push remains blocked by the previous automatic approval review pending confirmation of the GitHub destination; the user has not supplied that confirmation.

## 2026-09-25 - Recalculate s4 verifier statistics and costs

- Added Vietnamese `statistíc.md` from 1,633 verifier artifacts and 17 matching manifests: 1,626 pass, 2 reject, 5 processing failures; 13,578.94 seconds input and 13,537.24 seconds pass. Explained diarization gaps, merge recovery and 85 short clips removed (77.88 seconds), with mean/min/max and per-source breakdowns.
- Decimal cost totals: USD 16.254857750 in verdicts, separate pending batch cache USD 0.104487500, and user-confirmed separate testing USD 7.400700000: USD 23.760045250 estimated total. Documented billing uncertainty and unavailable testing breakdown. Existing costs match the stale 1,348-row CSV; 285 new artifacts add USD 2.686015375.
- Verified official pricing; thinking accounts for 95.9374% of billed output tokens. Audited verdicts with the shared validator, response hashes, manifest source metadata, report links and arithmetic; read relevant runners, launchers, helpers and docs. No tests (not requested), paid model calls or listening-based accuracy claims. Unrelated changes left untouched.
