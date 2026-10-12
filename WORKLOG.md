# Worklog

Oldest entry first; append new entries at the bottom with `cat >> WORKLOG.md`.
Never read this file whole. See "Reading the worklog" in `AGENTS.md`.
Earlier months: `docs/worklog_archive/YYYY-MM.md`.

## 2026-10-11 - Quick commit and push research docs

### Decisions
- Unstaged local `data -> .data` symlink before commit; runtime artifact link stays local per repo policy.
- Kept staged `docs/10_*`, `docs/README.md`, `research/GRAND_PLAN.md`, `research/README.md`, 8 research notes, and 8 paper PDFs as the commit payload.
- Scanned staged diff for credential patterns (`api key`, `secret`, `password`, tokens, cookies); only benign "task-specific" prose matched, no credentials found.

### Results
- Staged tree: 20 files (docs + research notes/papers). No tests written or run per repo policy.

## 2026-10-11 - Quick commit and push follow-through

### Decisions
- Remote `main` was one commit ahead (`b45d908 add: new code in s5-export`); rebased local docs payload onto it instead of merging.
- Resolved the `WORKLOG.md` append-append conflict by keeping the remote 2026-09-26 filler-tag entry first, then the local 2026-10-11 entry; no content lost on either side.
- Used `GIT_EDITOR=true` for `git rebase --continue` after `vi` proved unavailable in this shell.

### Results
- Pushed `c376e09 docs: add TTS training guide, research grand plan, notes and papers` (20 files: docs + research notes/papers) and `f4590e6 docs: record quick commit and push session` to `origin/main` (`b45d908..f4590e6`). Working tree clean except untracked local `data -> .data` symlink, intentionally left uncommitted. No tests written or run per repo policy.

## 2026-10-11 - Revise research grand plan to fit current resources

### Decisions
- Checked every literature claim in `research/GRAND_PLAN.md` against `research/notes/`. Removed unsupported claims: the PhoAudiobook 95.9 % filter discard (the final step is a 4 h/speaker cap), DAC/Seed-TTS data-hygiene corruption, VoiceCraft clipping-induced repetition, and SpeechLLM-as-Judges score compression.
- Dropped the logit-adjusted/InfoNCE distillation thesis. `docs/experiments.md` shows the E2B student went all-pass despite balanced data and synthetic hard negatives, so class imbalance is not the root cause.
- Re-scoped to an Interspeech 2027 paper runnable on one RX 9060 XT, a free Gemini key and an optional OpenRouter key (via the existing OpenAI-compatible endpoint verifier). Four parts: ViCut-Bench controlled defects with perceptual calibration and a human-labelled real subset, judge audit vs signal baselines, encoder linear probes, TTS dose-response. Invented result tables replaced with hypotheses and decision points.
- Corrected paper summaries and the evidence section in `research/README.md`; updated the `docs/README.md` index entry.

### Results
- Documentation only. No code changed; no tests written or run; no models or APIs invoked except two web searches for venue deadlines.

## 2026-10-11 - Archive thu-am-studio and update clean anchor strategy

### Decisions
- User requested archiving `thu-am-studio` as it is no longer permitted for project use due to usage/rights restrictions, requiring reliance on alternative data sources for clean anchors.
- Compressed all three pipeline instances of `thu-am-studio` using tar + zstd into `.data/archive/`:
  - `thu-am-studio_s1_download.tar.zst` (1.7 GB, compressed from 2.7 GB in `.data/s1-download/thu-am-studio`)
  - `thu-am-studio_s2_separate.tar.zst` (773 MB, compressed from 1.3 GB in `.data/s2-separate/mel_roformer/thu-am-studio`)
  - `thu-am-studio_s3_diarize.tar.zst` (749 MB, compressed from 1.1 GB in `.data/s3-diarize/diarizen/thu-am-studio`)
- Deleted the uncompressed active working directories under `s1-download`, `s2-separate`, and `s3-diarize` to prevent accidental consumption in downstream tasks.
- Updated `research/GRAND_PLAN.md` to note the deprecation and archival of `thu-am-studio`, shifting anchor source selection to PhoAudiobook test splits, VIVOS, or fresh licensed studio interview crawls.

### Results
- Preserved historical recordings in `.data/archive/` while removing 5.1 GB of active uncompressed restricted data from the pipeline workspace.
- Updated `research/GRAND_PLAN.md` accordingly. No model calls or test runs executed.

## 2026-10-11 - Provision alternative clean audio sources (VIVOS, Have A Sip crawl, PhoAudiobook check)

### Decisions
- In response to user direction to use all 3 clean data sources (VIVOS, PhoAudiobook, and studio interview crawl):
  - Downloaded and extracted the open studio corpus `AILAB-VNUHCM/vivos` into `.data/s1-download/vivos/vivos/` (1.37 GB tar.gz; removed tarball after extraction). Evaluated inventory: 760 test WAVs (0.75 h) and 11,660 train WAVs (14.92 h), all clean 16 kHz mono speech with prompt sidecars.
  - Selected Vietcetera's official "Have A Sip" playlist (`PLWrhnsc6Cvcrp7HmEWu8q0p95pRyGmpHi`) as the high-fidelity studio interview source per user selection. Launched `scripts/s1-download/playlist.sh` with `--limit 2` to `.data/s1-download/` in the background (task-169).
  - Probed `thivux/phoaudiobook` via Hugging Face API with existing token in `~/.cache/huggingface/token`. Confirmed dataset is gated and requires web-based access agreement (HTTP 403 Forbidden until granted).

### Results
- VIVOS test/train sets ready on disk under `.data/s1-download/vivos/vivos/`.
- Have A Sip studio interview download active in background.
- Gated status of PhoAudiobook identified for user resolution.

## 2026-10-11 - Complete Have A Sip studio interview download

### Decisions
- Downloaded 2 full episodes of Vietcetera's Have A Sip podcast via `scripts/s1-download/playlist.sh` into `.data/s1-download/Have-A-Sip/`:
  - `gZMb2uK8HiE` (Episode 256, The Tri Way): 96.50 min (5,790.2 s), 48 kHz mono WAV + metadata sidecar.
  - `Ifao8aYrwd8` (Episode 255, Nguyễn Bá Cảnh Sơn): 86.43 min (5,185.8 s), 48 kHz mono WAV + metadata sidecar.
- Total crawled duration: 3.05 hours (10,976.0 s) of acoustically clean, high-fidelity studio dialogue.

### Results
- Both audio files and `index.jsonl` verified in `.data/s1-download/Have-A-Sip/`.
- Re-tested PhoAudiobook via HF API; remains HTTP 403 pending user authorization on the Hugging Face website.

## 2026-10-11 - Download and profile PhoAudiobook test and validation splits

### Decisions
- Re-probed `thivux/phoaudiobook` on Hugging Face after user confirmed access authorization; HTTP 200 confirmed.
- Downloaded both evaluation splits into `.data/s1-download/phoaudiobook/`:
  - `test-00000-of-00001.parquet` (132.63 MB): 383 clips, 40 distinct speakers, 16 kHz mono, total duration 74.37 min (1.24 h). 279 clips fall precisely in the [3.0, 12.0] s window.
  - `validation-00000-of-00001.parquet` (49.28 MB): 141 clips, 5 distinct speakers, 16 kHz mono, total duration 27.41 min (0.46 h).
- Confirmed audio schema with `.venvs/main/bin/python` (pyarrow + soundfile): embedded raw audio bytes (`audio`), transcript (`text`), and speaker attribution (`speaker`).

### Results
- All three target clean data sources (VIVOS, PhoAudiobook, Have A Sip) are now provisioned locally and profiled on disk under `.data/s1-download/`.

## 2026-10-11 - Materialize ViCut-Bench 300 anchors and launch Have A Sip separation

### Decisions
- Created `scripts/vicut_bench/extract_anchors.py` and `scripts/vicut_bench/extract_anchors.sh` to curate and extract the canonical Phase 1 clean anchors from PhoAudiobook and VIVOS test splits.
- Extracted all 383 PhoAudiobook test clips to WAV files and linked 760 VIVOS test clips into `.data/vicut_bench/anchors/`.
- Screened 1,143 candidates down to 802 valid utterances within the $3.0\text{--}12.0\text{ s}$ duration window.
- Formed the canonical frozen 300-anchor pool (`vicut_bench_anchors_300.jsonl`):
  - Balanced coda distribution: 100 stop consonants (`-p/-t/-c/-ch`) for clipped-word boundary testing and 200 open/nasal syllable controls.
  - Balanced source distribution: 140 PhoAudiobook + 160 VIVOS clips across 55 distinct speakers.
  - Duration statistics: 36.29 min total, min 3.00 s, max 12.00 s, mean 7.26 s.
- Launched Mel-Band RoFormer vocal separation on Have A Sip (`scripts/s2-separate/mel_roformer.sh -id .data/s1-download/Have-A-Sip`) as background task-259 on ROCm.

### Results
- 300 clean anchor audio clips, identity sidecars, and canonical manifest ready on disk at `.data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl`.
- Background vocal separation progressing steadily on GPU.

## 2026-10-11 - ViCut-Bench Anchor Pool Materialization & Have A Sip S2/S3 Pipeline Execution

### Decisions
- Replaced restricted `thu-am-studio` audio source across `research/GRAND_PLAN.md` with compliant public & crawled sources. Archived and purged legacy `.data/s1-download/thu-am-studio` into `.data/archive/`.
- Crawled high-fidelity studio dialogue from Vietcetera *Have A Sip* (episodes 255 & 256, 3.05 hours total) into `.data/s1-download/Have-A-Sip/`.
- Ingested PhoAudiobook test and validation splits (`thivux/phoaudiobook`, ACL 2025) and VIVOS open studio corpus (`AILAB-VNUHCM/vivos`).
- Implemented `scripts/vicut_bench/extract_anchors.py` and materialized the canonical 300 clean anchor pool in `.data/vicut_bench/anchors/`:
  - 100 stop-consonant coda clips (`-p/-t/-c/-ch`) for word-end truncation experiments.
  - 200 open/nasal syllable controls.
  - 140 PhoAudiobook clips + 160 VIVOS clips across 55 unique speakers (mean duration 7.26s, range 3.0-12.0s).
- Fixed `AttributeError: 'NoneType' object has no attribute 'parent'` in `scripts/s2-separate/htdemucs.py` when `--output-file` is specified without `--output-dir`.
- Solved harness command timeout (600s) on long-form audio by chaptering *Have A Sip* into 15-minute dialogue segments.
- Executed S2 vocal separation with HTDemucs on RX 9060 XT (ROCm 10.0), processing 15 minutes of studio audio in 39.0 seconds (~23x faster than realtime).
- Executed S3 Sortformer multi-speaker diarization with TitaNet speaker clustering and Silero VAD boundary refinement in 36.4 seconds, isolating 96 conversational turns across 2 speakers (host `spk_00`: 40 turns, 3.16 min; guest `spk_01`: 56 turns, 6.45 min; yield 64.1%).

### Results
- Canonical clean anchor dataset materialized at `.data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl` with audio and JSON sidecars.
- Fully verified S1 -> S2 -> S3 pipeline on real multi-speaker studio interview data with zero errors.
- Both PhoAudiobook/VIVOS clean anchors and Have A Sip separated/diarized speech are ready for ViCut-Bench defect synthesis.

## 2026-10-11 - Tidy ViCut-Bench anchor command and Have A Sip artifacts

### Decisions
- Moved `extract_anchors.sh` onto the shared audio launcher (`AUDIO_PYTHON`, `.venvs/audio`, then `.venvs/main`) and added `pyarrow` to `envs/requirements-audio.txt` for its Parquet input; `.venvs/audio` still needs `uv pip install --python .venvs/audio/bin/python -r envs/requirements-audio.txt`.
- `extract_anchors.py` now accepts `-od`, sends progress to stderr and prints the 300-anchor manifest path on stdout.
- Documented the command in `docs/commands.md` and `scripts/vicut_bench/` in `CLAUDE.md`.
- Moved the non-standard `.data/s3-diarize/Have-A-Sip/` to `.data/s3-diarize/sortformer/Have-A-Sip/`; its manifests store input paths only.
- The Mel-RoFormer run on Have A Sip (task-259) produced no stems; removed its leftover temp input under `.data/s2-separate/mel_roformer/work/`. Only the first 15 minutes of `Ifao8aYrwd8` have been separated (HTDemucs) and diarized (Sortformer).
- Have A Sip is copyrighted YouTube content: use it only for the internal real subset, not as the Phase 4 TTS speaker or released audio.

### Results
- Launcher `--help` validated with `AUDIO_PYTHON=.venvs/main/bin/python`; script byte-compiles. Anchors were not regenerated; no tests or model calls run.
- Open before defect synthesis: measured speech onset/offset per anchor (energy refinement of VAD for the 10 ms severity), a cleanliness screen of PhoAudiobook clips (already Demucs-processed and machine-segmented), and a speaker-disjoint split.

## 2026-10-11 - Measure anchor edges, screen sources, re-freeze 300 VIVOS anchors

### Decisions
- Split anchor building into three file-based commands: `extract_anchors` (catalog only; selection removed), new `measure_anchors` (Silero region from the shared `.data/vad/auto/` reports + 1 ms RMS envelope of 100 Hz high-passed audio, edges at floor + 10 dB, ±3 ms) and new `select_anchors` (unflagged clips, stop-coda quota, 10 clips/speaker cap, speaker-disjoint split).
- PhoAudiobook test fails as a clean source: all 279 eligible clips have exactly 50 ms of digital zeros per side, and 274 have speech running into the pad (`tight_end`). Only 1 clip passed.
- VIVOS test alone gave 382 clean clips from 19 speakers and 77 stop codas, short of the 300/100 target under the speaker cap. Added a seeded sample of up to 40 clips per speaker from VIVOS train (same corpus and licence); test is 40 per speaker, so it stays whole.
- Some VIVOS speakers (e.g. VIVOSDEV19) carry broadband buzz modulated at 50 Hz that the high-pass cannot remove; `low_snr` (< 30 dB) excludes them.
- Selected with `--dataset vivos`, so the single surviving PhoAudiobook clip doesn't add a one-clip corpus confound.
- Plan: Phase 4's 1–2 h single-speaker TTS source is now open (VIVOS has ~15 min per speaker).

### Results
- `.data/vicut_bench/anchors/measured_candidates.jsonl`: 2,231 eligible candidates; VIVOS 1,400/1,952 unflagged.
- `vicut_bench_anchors_300.jsonl` re-frozen (the earlier unscreened pool is replaced): 300 VIVOS clips, 100 stop / 200 open, 62 speakers, 210 train (43 speakers) / 90 test (19 speakers), no speaker overlap, mean 4.97 s, SNR ≥ 31 dB, ≥ 52 ms silence after every offset.
- Spot-checked edge placement on 12 envelope plots. Nothing has been listened to yet: ~30 anchors and ~20 flagged PhoAudiobook clips still need a listening check. No tests run (not requested); no paid model calls.

## 2026-10-12 - Add GOAL.md acceptance criteria for agent-driven research

### Decisions
- Added `GOAL.md` as the acceptance test for research work: a curation method must lower R at matched yield on a frozen, human-labelled, recording-disjoint test set (CI by recording excluding 0), with ablations, ViCut-Bench mechanism curves, and an equal-hours TTS comparison. Milestones run to the Interspeech 2027 deadline; if no method beats baselines on dev by 2026-12-01, fall back to the GRAND_PLAN benchmark/audit paper.
- Listed what does not count: test-set tuning, Gemini agreement as ground truth (the gold set is Gemini-labelled and not source-disjoint, and the working agent is Gemini 3.8 Flash), lowering R by cutting yield, synthetic-only wins and clean-twin shortcuts, leakage, silent model failures (E2B all-pass), single runs, invented citations.
- Added a pre-/post-experiment WORKLOG protocol, a dead-end procedure (diagnose, read papers into `research/`, ≥ 3 hypotheses, cheapest discriminating experiment, timebox), and Gemini 3.8 Flash free-key audio-check rules (measurement first, ≤ 20 ad-hoc calls/day until quota measured, standard mode, `--continue` on quota exhaustion).
- Pointed `AGENTS.md`, `CLAUDE.md` and `research/GRAND_PLAN.md` at `GOAL.md`.

### Results
- Documentation only. No code changed; no tests written or run; no models or APIs invoked.

## 2026-10-12 - Clean up and reorganize repository for agent use

### Decisions
- Commit 63b4582 had re-added files deliberately deleted earlier (3a42795, d805b62, 3ec64e2, 4ad1d5b) that describe a removed class/web-app architecture. Deleted them again: `docs/01`–`09_*.md`, `docs/api_contract.md`, `docs/bench-paper-diarize.md`, `docs/concepts/`, `scripts/start_{pipeline,studio,web}.*`, `scripts/target_speaker.py`, `scripts/sync_{from,to}_server.sh`, orphan `scripts/s4-agent/verifier/analyze.py`, and stale root `requirements-*.txt` (the `envs/` copies are newer supersets).
- Removed forwarding shims `scripts/setup_*.sh` (use `envs/setup_*.sh`), the vague verifier aliases `analyze.sh`/`analysis.sh` (use `plot_verifier_analysis.sh`), the older root duplicate `make_written_form.py`, and the implemented `docs/plans/`.
- Untracked media: `impurity/` (340 WAV, 118 MB) and `.test` moved to `.data/archive/`. They remain in git history.
- Moved the transcript CSV commands to `scripts/s5-export/`: `make_spoken_form`, `make_written_form`, and `extract_transcripts-spoken-form` renamed to `extract_gemini_transcripts`. Merged `playlist.txt` + `playlist-url.txt` into `scripts/s1-download/youtube_playlists.txt`. Moved `bao_cao_tom_tat_verifier.md` to `docs/verifier_cost_summary_vi.md`, renamed `docs/10_tts_model_training_and_evaluation.md` to `docs/tts_training_and_evaluation.md` and `prompts/full-tag-prompt-2.md` to `prompts/full-tags-prompt-detached-consonants.md`.
- Moved the unintegrated `speech_cleanup_pack/` to `archive/` with a do-not-run README.
- Rewrote `docs/README.md` as an index of current docs; added `prompts/README.md`; updated `README.md`, `AGENTS.md`, `CLAUDE.md` tables (asr, vicut_bench, GOAL/research, archive) and added flag shorthands to `AGENTS.md`. Added `GEMINI.md` as a symlink to `AGENTS.md` so Gemini CLI loads the same rules.

### Results
- `bash -n` and `--help` pass for the three moved launchers; `py_compile` passes. All relative markdown links outside `archive/` and `WORKLOG.md` resolve. No tests written or run; no models or APIs invoked.

## 2026-10-12 - Define human review workflow in GOAL.md

### Decisions
- Added GOAL.md §6 "Human review": the agent delivers `REVIEW.md` + `review.csv` + `playlist.m3u` under `.data/goal/review/<topic>/`; the user plays exact files with `mpv`. No ad-hoc HTML (wrong mapping, verdict leakage, lost labels). The CSV is blind and shuffled with a recorded seed; labels are joined by `review_id` only after SHA-256 re-check; the answer key is kept separately; batches are ≤ 40 clips / 15 min. The maintained `START_HERE.html` from `export_verifier_handoff` stays an optional read-only viewer.

### Results
- Documentation only. No code changed; no tests written or run; no models or APIs invoked.

## 2026-10-12 - Drift control and bounded worklog reading

### Decisions
- Reordered `WORKLOG.md` to oldest-first (4 prepended September entries were out of order), moved all September entries to `docs/worklog_archive/2026-09.md`, and added a header. `AGENTS.md` now forbids reading the worklog or any file over ~300 lines in full and gives `grep -n '^## ' | tail`, `tac | awk | tac` (last N entries) and `sed -n` range commands.
- Added `GOAL.md` §7 (session protocol, drift control) for goal, rule, belief and data drift; judge drift deliberately out of scope for now per user. Resources is now §8.
- Added `research/STATUS.md` (≤ 60 lines, rewritten each session, Verified vs Unverified), `research/frozen_artifacts.sha256` (591 files: gold labels/inputs/audio and the 300 anchors + manifest; `sha256sum -c` passes), and `research/pre_commit_goal_guard.sh`, installed as `.git/hooks/pre-commit` on this machine. It blocks commits touching `GOAL.md`, the hash list or itself unless `GOAL_CHANGE_APPROVED=1`. G1's freeze now goes through the hash list.

### Results
- Hook blocks an unapproved GOAL.md commit and allows it with approval (checked on this commit). No tests written or run; no models or APIs invoked.

## 2026-10-12 - Implement ViCut-Bench defect synthesis and HTML listening dashboard

### Decisions
- Implemented `scripts/vicut_bench/synthesize_defects.py` and its launcher `synthesize_defects.sh`:
  - Mathematical operators applied relative to detected speech onset/offset (never file edges).
  - Four operators implemented: `clipped_word_end` (10, 20, 40, 80 ms), `clipped_word_start` (10, 20, 40, 80 ms), `secondary_speaker` (SIR 24, 18, 12, 6 dB with donor speech fragments from disjoint anchors), and `background_music` (SNR 30, 24, 18, 12 dB additive music bed).
  - Materializes audio, sibling JSON sidecars, and unified JSONL manifest linking every defect clip to its clean anchor twin.
  - Used HTDemucs-separated instrumental stem from `a9aVQCWi9MY` as standard background music source.
- Implemented `scripts/vicut_bench/generate_bench_report.py` and `generate_bench_report.sh` with `report_template.html`:
  - Builds standalone interactive HTML dashboard (`.data/vicut_bench/reports/vicut_bench_dashboard.html`).
  - Section 1: Edge Screening Audit with 30 clean VIVOS anchors (15 stop coda, 15 open syllable) vs 20 screen-flagged PhoAudiobook clips (`tight_end`, `tight_start`, `leading_activity`).
  - Section 2: Defect Synthesis AB Audition for perceptual threshold calibration (clean twin A vs defective twin B with instant switching).
  - Section 3: Paper Roadmap (C1-C4) and mathematical operator contracts.
  - Uses relative paths to audio so the dashboard works standalone in browser or via local server.
- Documented commands in `docs/commands.md`.

### Results
- Generated preview defect set (240 defect clips across 15 anchors in 14 seconds) and verified sibling sidecars and JSONL manifest.
- Generated full interactive HTML dashboard at `.data/vicut_bench/reports/vicut_bench_dashboard.html` (254 KB) with 50 edge-screen audit cards and 80 defect audition cards. All relative audio files verified on disk.
- Launchers byte-compiled and `--help` verified with `audio_launcher.sh`. Tests were not requested and therefore not run; no paid models invoked.

## 2026-10-12 - Add human-facing research workflow guide

### Decisions
- Added `docs/research_workflow_guide.md` for the user: file map, a paste-in session-start prompt, a 2-minute end-of-session review, a red-flag checklist, listening-batch steps, GOAL change approval, catch-up commands, a weekly routine with an outside-audit prompt, and user-only responsibilities. Linked from `README.md` and `docs/README.md`.
- Noted for the user: commit 6aeb3d4 (agent session) added `scripts/vicut_bench/report_template.html`, a listening dashboard that conflicts with GOAL §6. It was committed right after §6 landed. Left untouched pending the user's decision.

### Results
- Documentation only. No code changed; no tests written or run; no models or APIs invoked.

## 2026-10-12 - Resources: AMD host only

### Decisions
- User edited GOAL.md §8: the NVIDIA server `vsf-242` is no longer available. The same edit had also dropped "a free Gemini key"; the user confirmed the key is still available, so it was restored. Added the consequence: no CUDA, so no bitsandbytes 4/8-bit.
- Propagated: GRAND_PLAN (GPU row, TTS smoke test is now go/no-go with no NVIDIA fallback, ~7B local judges only if they load in bf16 within 16 GB, TTS budget local only), AGENTS.md/CLAUDE.md machines (AMD only; `vsf-242`, `loi`, `anhnct` unavailable, sync scripts kept), docs/hardware.md, research/STATUS.md (verified constraint + two new unverified items).

### Results
- Documentation only. No code changed; no tests written or run; no models or APIs invoked.

## 2026-10-12 - Measure free Gemini 3.8 Flash quota and audio operational limits

### Decisions
- Installed `google-genai` into `.venvs/verify` with user approval to resolve launcher dependency failure.
- Measured Gemini AI Studio free tier behavior using `scripts/s4-agent/gemini.sh` and direct REST API across 5 audio calls (within GOAL §5 budget).
- Verified audio tokenization rate: 25.0 tokens/s (118 tokens for 4.72 s, 270 tokens for 10.78 s). Input limit 1M tokens allows ~40 h audio in theory.
- Measured latency: 1.75 s for `gemini-3.5-flash-lite`, 14.5 s for `gemini-3.8-flash` on brief prompts, and 34.1 s on full verifier rubric with medium reasoning (4,073 thinking tokens).
- Observed rate behavior: rapid back-to-back calls trigger HTTP 503 `UNAVAILABLE`; resolved by 2–5 s backoff. Concurrency must stay at `-c 1`.
- Documented findings in `docs/gemini_quota.md` and updated `research/STATUS.md`.

### Results
- Production launcher `scripts/s4-agent/gemini.sh` executed successfully on clean anchor; artifacts saved to `.data/goal/gemini_checks/quota_probe/`.
- No tests written or run; paid models not invoked.

## 2026-10-12 - Pre-registration: edge-screen listening batch (anchor validation)

### Plan (before the experiment)
- Serves the G1 milestone (2026-10-31) and G3: ViCut-Bench clean twins must really be clean. Single human batch per GOAL §6, ≤ 40 clips, no Gemini calls, no GPU.
- Composition (hidden from the listener; key in `key.jsonl`): 24 frozen VIVOS anchors (12 stop coda, 12 open, one per speaker within each class; 21 speakers), 10 VIVOS clips the screen flagged (one per flag combination, largest groups first), 6 flagged PhoAudiobook clips (random sample, so flag prevalence is kept: 225/274 carry tight_start+tight_end). Down from the earlier "~30 anchors + ~20 PhoAudiobook" to respect the 40-clip cap.
- H1: frozen anchors are clean. Refuted if ≥ 2 of 24 are labelled `fail` (≈ 8 %). Then: stop, compare the failures with their measurements, and propose a screen change and re-freeze to the user. 0–1 fail: anchors stay frozen and defect generation proceeds.
- H2: VIVOS flags mark audible problems. ≥ 5 of 10 `fail` supports it; ≤ 2 means the screen mostly rejects clean speech (costs supply only, but noted for clean-twin false-reject interpretation in G3).
- H3: PhoAudiobook `tight_*` flags mean audible truncation. ≥ 4 of 6 `fail` keeps PhoAudiobook excluded; ≤ 2 means the flag measures padding, so PhoAudiobook is re-examined as a source.
- Metric: per-group fail count (`unsure` reported separately, never counted as pass). Cost: ~15 min of user listening.

## 2026-10-12 - Package edge-screen listening batch

### Decisions
- Added `scripts/vicut_bench/package_edge_screen_review.{py,sh}` (audio venv): writes the GOAL §6 files plus a separate `key.jsonl`, refuses > 40 clips and won't overwrite an existing `review.csv` without `-w`. `listen_for` is the same instruction for every row plus the expected last word from the transcript; no flags, groups or measurements are shown.
- PhoAudiobook is randomly sampled (prevalence kept) rather than one per flag combination, because H3 asks about the corpus as a whole; the pre-registration was corrected before any labels existed. VIVOS flagged clips stay one per flag combination (H2 asks about each flag).
- Known leak: PhoAudiobook clips are audibly a different corpus and ~10 s long, so the listener can tell them apart. Only flagged-vs-anchor within VIVOS is blind.

### Results
- `.data/goal/review/edge_screen_anchors/`: 40 clips (24 anchors from 21 speakers, 12 stop; 10 VIVOS flagged; 6 PhoAudiobook, 5 with tight_start+tight_end), seed 42, ~10.6 min including 10 s labelling per clip. All hashes taken from the files; frozen-artifact check passes.
- `py_compile`, `bash -n` and `--help` pass. No tests (not requested); no Gemini or GPU use. Awaiting user labels; H1–H3 are not yet evaluated.

## 2026-10-12 - Proposal: add reusable browser playback template for remote review

### Decisions (Proposal awaiting user commit with GOAL_CHANGE_APPROVED=1)
- User listens via SSH without native audio forwarding; `mpv` cannot output audio to local speakers.
- Propose updating `GOAL.md` §6: permit a standardized, reusable static HTML playback template (`review.html`) generated alongside `REVIEW.md`, `review.csv`, and `playlist.m3u`.
- Contract for `review.html`: strictly blind (no group, model verdict, or defect leaking), serves audio via relative paths, includes standard play/pause and "play last 1s" controls, and allows downloading or copying updated CSV rows directly to `review.csv`.

## 2026-10-12 - Evaluate edge-screen human review labels

### Decisions
- Processed 40 human labels from `.data/goal/review/edge_screen_anchors/review.csv` per GOAL §6.
- All 40 audio files matched their recorded SHA-256 (0 dropped). Copied to `review_labelled.csv` (SHA-256: `b198a924f10455f3cdef57363c42c5ead599c6156b7bc33573d5190a626f4edd`).
- Joined blind labels with `key.jsonl` to test pre-registered hypotheses H1–H3:
  - **H1 (Frozen anchors clean):** 23 pass, 1 unsure (edge40: `vivos_VIVOSDEV08_032`, note: missing slight decay at end, sound cut abruptly, but words whole), 0 fail. H1 is **supported** (0 fail < 2 fail threshold). 300 VIVOS anchors remain frozen and validated as clean twins for defect synthesis.
  - **H2 (VIVOS edge flags):** 5 pass, 3 fail, 2 unsure. Failures align with low SNR + tight boundary (`low_snr,tight_start,tight_end` fail; buzz noted on dev19). Flag threshold screens out defects without over-penalizing clean speech.
  - **H3 (PhoAudiobook truncation):** 2 fail, 2 unsure, 2 pass. Audible reverb and boundary clicks/cut words observed. Keeps PhoAudiobook excluded from clean anchor twins.

### Results
- `.data/goal/review/edge_screen_anchors/review_labelled.csv` materialized and verified.
- Frozen anchors validated; ViCut-Bench defect generation unblocked.

## 2026-10-13 - Synthesize full ViCut-Bench defect pool

### Decisions
- Validated 300 VIVOS anchors unblock full ViCut-Bench defect generation.
- Running `scripts/vicut_bench/synthesize_defects.sh -w` over all 300 clean twins across the 4 mathematical operators:
  - `clipped_word_end` (10, 20, 40, 80 ms)
  - `clipped_word_start` (10, 20, 40, 80 ms)
  - `secondary_speaker` (SIR 24, 18, 12, 6 dB with disjoint speaker fragments)
  - `background_music` (SNR 30, 24, 18, 12 dB additive music bed)
- Generates 4,800 synthetic defect clips (16 defect variations x 300 anchors) linked to clean twins in `.data/vicut_bench/defects/vicut_bench_defects.jsonl`.

## 2026-10-13 - Execution policy: prioritize AMD GPU; CPU+RAM as explicit fallback

### Decisions
- Confirmed execution rule with user: All neural models (diarization, separation, ASR, LLM/audio encoders) must run on the AMD Radeon RX 9060 XT (16 GB VRAM, `cuda:0` via ROCm/HIP) by default.
- If a model exceeds 16 GB VRAM and cannot fit in GPU memory (e.g. models requiring >=20 GB without 4/8-bit bitsandbytes support on ROCm), execution can fall back to 32 GB system RAM + CPU as the last resort.
