# TTS strategy execution log

## Current handoff — 2026-09-08

- Strategy: [TTS_PRODUCTION_STRATEGY.md](TTS_PRODUCTION_STRATEGY.md).
- Active phase: 1 evidence is in place; Phase 2 lock is a competitor-conflict gate,
  not a clip repairer.
- Authorization: user requested execution with regular commits and pushes.
- User clarification: Gemini 3.8 Flash MEDIUM is accepted as human-quality ground
  truth; API hearing is explicitly authorized. Separate human review is not a gate.
- **Unified MEDIUM gold / benchmark pool:** `pool_20260908_v4` /
  `gold_benchmark_20260908` = **288 clips / 11 recordings** (159 pass / 129 reject).
  This merges v3 (241) with the former held-out Khanh Vy challenge (31 originals +
  16 relocked boundary children from `youtube:H0VpjeULCck`). User requested the
  merge because n=31 was too small for benchmarking. **There is no longer a
  source-disjoint held-out recording inside this file**; reserve a fresh source
  later if unseen-source generalization must be measured.
- Completed MEDIUM API audits feeding v4: phase1 challenge 31, source-resolved
  pool 97, boundary children 16, locked extracts 36, Experiment-tab sweep 110
  unique (plus earlier lineage joins). v1/v2/v3 left intact.
- Experiment-tab measured harvest recipe (`recipe_nolock`) is the Studio Reset /
  status default. See checkpoint 8.
- **Local verifier vs gold (checkpoint 10):** Gemma 4 E2B LoRA V3 on the 288-clip
  MEDIUM gold is still an all-pass classifier (288/288 `pass`). Agreement
  **55.2%** equals the Gemini pass rate; F1 **71.1%**; 129 contamination leaks
  (0 true rejects). Prior 31-cut Khanh Vy figure was **38.7%** agreement.
- **Gemini 3.5 Flash-Lite vs gold (checkpoint 11):** same 288-clip MEDIUM gold,
  `--reasoning-effort medium`. Agreement **63.2%**, F1 **73.5%**, 47 rejects
  (35 true / 12 false), 94 contamination leaks. Beats E2B V3 on agreement/F1
  because it can reject; still far from teacher quality. Cost was not recorded
  on that run; `GeminiVerifier` now logs USD automatically (checkpoint 12).
  Portable full-gold runner: `./scripts/run_gold_verifier_eval.sh`.
- Next: grow studio-interview / narration sources with the measured harvest
  recipe; retrain or calibrate the student so it can reject (music / secondary
  speaker first). Prefer a *new* held-out recording for future generalization
  checks. Do not prefer music-backed vlogs.

## Checkpoints

### 1. Durable strategy and Phase 1 brief

Saved decision order, metrics, local-first architecture, teacher restrictions,
hardware assumptions, human-label requirements, and explicit continuation steps.
The existing pipeline is not qualified for the target error rate.

### 2. Offline audit and explicit MEDIUM ground truth

Commands (use `UV_CACHE_DIR=.data/uv-cache` in this sandbox):

```bash
uv run --no-sync python scripts/audit_tts_data.py prepare --output .data/tts_strategy/phase1_20260908
uv run --no-sync python scripts/audit_tts_data.py teacher --packet .data/tts_strategy/phase1_20260908 --limit 31
```

Inventory: 263 train + 65 validation rows, no missing audio, all 328 missing
explicit recording_id; 48 train + 15 validation outside 2–15 s. Ten shared filename
groups flag source leakage. One exact SHA-256 audio duplicate crosses splits
(train row 108, validation row 20), despite different paths. Identity coverage
must be restored from source manifests/generators, not inferred from no duplicates.

Legacy evaluation: 31 accepts / 19 reference rejects (61.3%); restricting to
2–15 s gives 15 accepts / 11 reference rejects (73.3%). All 31 reference records
omit audio_quality. Archived `compare_verifiers_khanhvy.py` configures LOW, so
historical titles calling these MEDIUM do not establish actual provenance.
The fresh audit uses model `gemini-3.8-flash`, `thinkingLevel=MEDIUM`, a structured
nine-dimension rubric, exact input hashes, raw responses and usage metadata.

New script: `scripts/audit_tts_data.py` (`prepare`, `teacher`, `report`, `lineage`). Artifacts:
`.data/tts_strategy/phase1_20260908/`. No inference runs during inventory; no
pytest/unit cases were written or run. Production risk remains unqualified due
to challenge sampling, not due to a requirement for additional human review.

### 3. Legacy recording lineage recovery

Command:
```bash
uv run --no-sync python scripts/audit_tts_data.py lineage --output .data/tts_strategy/lineage_20260908_v2
```

Lineage recovery results:
- Unique known lineage: 233 clips (95 quarantined: 76 HaveASip with missing original video IDs, 15 unlinked augmentations, 4 duplicate audio across recordings).
- Challenge recording reserved: 136 clips from `youtube:H0VpjeULCck` permanently isolated from unseen evaluations.
- Source-resolved eligible pool: 97 unique clips across 7 YouTube source recordings
  (2–15s duration); eligibility does not mean acoustically clean.
- Artifacts: `.data/tts_strategy/lineage_20260908_v2/` and `.data/tts_strategy/pool_20260908/`.

### 4. Fresh ground truth and source-disjoint pilot data

- Explicit MEDIUM full-rubric challenge results: 21 rejects / 10 passes out of 31.
  E2B accepts all: accepted error 67.7% overall; 11/15 = 73.3% for 2–15 s clips.
  Defect occurrences: secondary speaker 10, clipped end 6, reverb 4, clipped start
  4, music 3, effects 2, overlap 2, excessive noise 1 (multi-label, not additive risk).
- Challenge API usage: 9,883 prompt + 6,296 answer + 13,995 thinking = 30,174 tokens.
  All 31 responses identify `gemini-3.8-flash`. No prices assumed.
- Fifteen challenge audio files are byte-identical to old training audio. This
  challenge cannot establish generalization for the old E2B model.
- The 97-clip source-resolved pool has 42 passes / 55 rejects after fresh MEDIUM
  evaluation. Export: `.data/tts_strategy/pool_20260908/labeled_pool.jsonl`.
- Pilot training split: 50 clips (25/25), five recordings. Calibration: 26 clips
  (17 pass / 9 reject), two recordings (`youtube:fwN5VT_QxkY`, `youtube:Oa-mVxGS4cw`).
  Training balancing discarded 21 rows; full pool is preserved. No new untouched
  release split exists. 26 calibration clips cannot support a <10% one-sided 95%
  bound even with zero errors; do not claim production qualification.
- Sandbox masks GPU availability. Outside it: AMD RX 9060 XT, PyTorch
  `2.13.0+rocm10.0.0`, HIP `7.15.26333`, one device. 4090 access unverified.
  Default-cache lookups for PhoWhisper-small and WavLM Base+ configs found nothing.
- The original source URL for HaveASip was requested asynchronously; continue
  independent work while its 76 clips and unresolved derivatives remain quarantined.

### 5. Boundary correctness engineering (no measured quality claim yet)

- Word locking does not expand into inter-turn gaps. Completing a recognized
  word that would enter a competitor or adjacent turn is rejected. ASR overlap
  into an undiarized gap is not treated as clipping proof (see checkpoint 6).
- Requested alignment loading/inference failures return no candidates with
  rejection audits. MMS remains an acoustic CTC-blank heuristic, not actual
  transcript-conditioned word alignment; docs now state that limitation.
- Smart segmentation requires ASR word gaps, checks overlapping/nested word
  coverage, keeps acoustic movement inside the gap, and enforces configured
  min/max lengths. Removed arbitrary valley fallback and oversized tail merging.
  Unsupported remainders are rejected rather than forced into training clips.
- Static AST parsing and `git diff --check` pass. No test cases were written/run.
  A real-audio before/after assessment is still required before a quality claim.
- Source context caution: first legacy clip was not byte-for-byte equal to a
  crop of `khanhvy_180s_slice.wav` at its rounded metadata timestamps. Establish
  processing/timestamp correspondence before using that file to repair old cuts.

### 6. Real-audio boundary evaluation (no quality win)

Commands:
```bash
uv run --no-sync python scripts/audit_tts_data.py boundaries \
  --packet .data/tts_strategy/phase1_20260908 \
  --source .data/experiment_khanhvy/khanhvy_180s_slice.wav \
  --output .data/tts_strategy/boundaries_20260908 \
  --device cuda:0 --aligner-model vinai/PhoWhisper-small
uv run --no-sync python scripts/audit_tts_data.py teacher \
  --packet .data/tts_strategy/boundaries_20260908/review_packet --limit 16
```

Correspondence: all 31 challenge cuts are exact PCM crops of
`.data/experiment_khanhvy/khanhvy_180s_slice.wav`. Filename/metadata timestamps
are rounded to 0.01 s and miss the true start by −4.7 ms to +4.7 ms. Repair work
must use located sample indices, not `round(start_s * sr)`.

Relock used public `align_and_lock_syllable_boundaries` and
`smart_segment_speaker_turns` with PhoWhisper-small on `cuda:0` (RX 9060 XT,
PyTorch `2.13.0+rocm10.0.0`). Incoming turns were the 31 located cuts, with
other extracted speakers as competitor intervals. Duration policy was the TTS
2–15 s window, not the pipeline 3–10 s default.

Outcomes: lock rejected 9/31 (8 `word_boundary_conflicts_with_safe_bounds`,
1 `no_complete_words_in_safe_interval`). Segmentation rejected 7 more as
`below_min_duration` (four of those were old MEDIUM passes shorter than 2 s).
16 children were emitted; all had moved bounds. Fresh MEDIUM: 15 reject / 1 pass
(R = 93.8% among emitted children). Usage: 5,619 prompt + 3,738 answer + 7,623
thinking = 16,980 tokens. All 16 responses identify `gemini-3.8-flash`.

Repair scorecard for the 10 old clipped rejects: 5 dropped by lock/duration,
0 became a MEDIUM pass, 1 still clipped, 4 lost the clip defect but gained
secondary-speaker/music/effects. Of 6 old MEDIUM passes that still emitted,
only `clip_0009` remained a pass; expansions created new clipping or imported
neighbors on the others.

This is not a production quality claim. Expanding an intersecting word into the
gap between extracted turns is unsafe: those gaps were never shown to be
trusted same-speaker audio. Next boundary change: reject that edge instead of
filling the gap, then repeat this same located-cut evaluation.

`export` and recording-group `balance` now exist so source-disjoint splits can
be rebuilt without overwriting the 2026-09-08 pool. No new training run.

Follow-up on the same 31 located cuts after disabling gap expansion:

- v2 (`boundaries_20260908_v2`): reject every ASR-overlapping edge. Lock
  dropped all 31 (30 `word_boundary_conflicts_with_safe_bounds`, 1 no words).
  PhoWhisper-small timestamps overlap essentially every challenge edge, so this
  is not a usable production gate. Yield zero; R undefined.
- v3 (`boundaries_20260908_v3`): do not expand into gaps; reject only when
  completing a word would enter a competitor or adjacent turn. Lock again
  rejected 9/31 (same 8 safe-bound conflicts + 1 empty interval). Duration
  policy dropped 13 more as `<2 s` and 1 for `no_supported_word_gap`. 8 children
  emitted. Four are byte-identical to the original crop; four differ by at most
  one sample from 0.1 ms timestamp rounding. Inherited MEDIUM labels on those
  eight: 4 pass / 4 reject, R = 50%. No new API calls. Still not a production
  claim. Remaining rejects are in-interval secondary speech, music, effects,
  and reverb — not gap expansion.

Word lock is now a measured competitor-conflict gate, not a clip repairer.

### 7. Source-disjoint pool growth (Sortformer + measured lock)

Commands:
```bash
uv run --no-sync python scripts/crawl_channels.py --urls \
  'https://www.youtube.com/watch?v=3Nll-JLzvvE' \
  'https://www.youtube.com/watch?v=NI8JVXNWlN8' \
  'https://www.youtube.com/watch?v=zK3qFnKZFRo'
.venv-sortformer/bin/python scripts/audit_tts_data.py extract \
  --manifest .data/crawled/crawled_manifest.json \
  --only-ids 3Nll-JLzvvE NI8JVXNWlN8 zK3qFnKZFRo \
  --output .data/tts_strategy/extract_20260908 --device cuda:0
uv run --no-sync python scripts/audit_tts_data.py teacher \
  --packet .data/tts_strategy/extract_20260908/review_packet --limit 36
uv run --no-sync python scripts/audit_tts_data.py export \
  --packet .data/tts_strategy/extract_20260908/review_packet \
  --output .data/tts_strategy/extract_20260908/labeled.jsonl
uv run --no-sync python scripts/audit_tts_data.py combine \
  --inputs .data/tts_strategy/pool_20260908/labeled_pool.jsonl \
            .data/tts_strategy/extract_20260908/labeled.jsonl \
  --output .data/tts_strategy/pool_20260908_v2/labeled_pool.jsonl
uv run --no-sync python scripts/build_distillation_dataset.py balance \
  --input-files .data/tts_strategy/pool_20260908_v2/labeled_pool.jsonl \
  --pass-ratio 0.50 \
  --validation-recordings youtube:fwN5VT_QxkY youtube:Oa-mVxGS4cw \
  --train-out .data/tts_strategy/pool_20260908_v2/train.jsonl \
  --val-out .data/tts_strategy/pool_20260908_v2/calibration.jsonl
```

New sources (challenge `youtube:H0VpjeULCck` remains reserved): Vietcetera
`10.000 hours` EP5 dancer (`3Nll-JLzvvE`, 806 s), EP4 catwalk (`NI8JVXNWlN8`,
699 s), and Khanh Vy FPT campus vlog (`zK3qFnKZFRo`, 492 s). Extraction composed
`run_zero_contamination_pipeline` with Sortformer, context collar, the measured
no-gap-expansion word lock (PhoWhisper-small on `cuda:0`), and 2–15 s smart
segmentation. Consensus was off: `.venv-diarizen` is CPU torch on this host.
The project `.venv` still has CUDA wheels that report no GPU; extract must use
`.venv-sortformer` (PyTorch `2.13.0+rocm10.0.0`).

Funnel and fresh MEDIUM labels (36/36 completed, 0 unresolved; all
`gemini-3.8-flash`, `thinkingLevel=MEDIUM`, reviewer
`gemini-3.8-flash:MEDIUM`, rubric `tts-v1`, provenance
`user_accepted_gemini_ground_truth`, service tier `standard`, one config
hash). Wall clock: extract 178.840 s (funnel sum 175.06 s), teacher 195.123 s
(per-clip latency min 3.374 s / mean 5.383 s / max 8.696 s / sum 193.792 s).
Sources total 1,996.66 s (0.5546 h) at 16 kHz mono. Emitted clip durations
2.484–14.714 s (sum 238.846 s, mean 6.635 s). No teacher-cap subsample:
`max_clips_per_recording=25` and every recording emitted ≤19 clips.

| Recording | Source s | Extract s | ×RT | Diarizer turns (speech s) | After lock | Emitted | Pass / reject | Pass s | Acc. min / src h | Defects |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `3Nll-JLzvvE` | 805.82 | 75.82 | 10.628 | 93 (598.20) | 27 (184.69) | 19 (103.23) | 16 / 3 | 89.544 | 6.667 | music 1, reverb 1, clipped start 1 |
| `NI8JVXNWlN8` | 698.88 | 59.80 | 11.687 | 30 (633.08) | 7 (158.87) | 4 (21.46) | 2 / 2 | 15.484 | 1.329 | music 1, clipped end 1 |
| `zK3qFnKZFRo` | 491.96 | 39.44 | 12.474 | 30 (228.84) | 20 (162.55) | 13 (114.16) | 2 / 11 | 7.311 | 0.892 | music 8, secondary 6, clipped end 3, effects 2, clipped start 1, reverb 1 |
| **total** | **1996.66** | **175.06** | **11.405** | **153 (1460.12)** | **54 (506.11)** | **36 (238.85)** | **20 / 16** | **112.339** | **3.376** | music 10, secondary 6, clipped end 4, clipped start 2, reverb 2, effects 2 |

Lock dropped 66 + 23 + 10 = 99 / 153 diarizer turns (speech 413.51 + 474.21 +
66.29 = 954.01 s). Segmentation dropped 8 + 3 + 7 = 18 / 54 remaining turns
(speech 81.46 + 137.41 + 48.39 = 267.26 s). Pass rate among emitted clips:
20/36 = 55.6% (dancer 16/19 = 84.2%, catwalk 2/4 = 50.0%, FPT vlog 2/13 =
15.4%). Planning yield floor is 10 accepted minutes / source hour; dancer
reached 6.667, the three-source mix 3.376. Emitted speaker IDs: `spk_00` 24,
`spk_04` 5, `spk_03` 4, `spk_01` 3 (FPT vlog is the only multi-speaker
emit). Artifact: `.data/tts_strategy/extract_20260908/measurements.json`.

Usage: 14,480 prompt + 6,577 answer + 12,162 thinking = 33,219 tokens. No
prices assumed. The dancer interview is the current best first-tier source;
lock still drops most catwalk turns (30 → 4) and does not remove music.

Pool arithmetic (legacy 2026-09-08 files not overwritten):

- Legacy labeled: 97 clips / 7 recordings (42 pass / 55 reject; 3.537 pass min).
- New locked labels: 36 clips / 3 recordings (20 pass / 16 reject; 1.872 pass min).
- Combined `pool_20260908_v2/labeled_pool.jsonl`: 133 clips / 10 recordings
  (62 pass / 71 reject; 5.409 pass min + 7.316 reject min).
- Combined by recording: `j83rzAzRDAI` 22, `3Nll-JLzvvE` 19, `QBml8L3wS3Q` 18,
  `i0zYcXBjytE` 14, `Oa-mVxGS4cw` 13, `fwN5VT_QxkY` 13, `zK3qFnKZFRo` 13,
  `lfIbjICmfW0` 11, `PXEtB-CsvSw` 6, `NI8JVXNWlN8` 4.
- Calibration frozen: `youtube:fwN5VT_QxkY` 13 + `youtube:Oa-mVxGS4cw` 13 = 26
  (17 pass / 9 reject), same as v1.
- v1 train: 50 (25/25) from 71 eligible after discarding 21.
- v2 train: 90 (45/45) from 107 eligible (45 pass / 62 reject) after discarding
  17 rejects to keep a 0.50 pass ratio. By recording: `3Nll-JLzvvE` 19,
  `j83rzAzRDAI` 19, `QBml8L3wS3Q` 15, `i0zYcXBjytE` 12, `zK3qFnKZFRo` 10,
  `lfIbjICmfW0` 8, `PXEtB-CsvSw` 4, `NI8JVXNWlN8` 3.
- iid one-sided 95% upper at zero errors: 0.109 on n=26 calibration; 0.047 on
  n=62 teacher passes. Both are diagnostic only (recording-clustered, not a
  release sample). 26 calibration clips still cannot support a <10% bound.
  Not a production quality claim. No new training run.

### 8. Experiment-tab config sweep (Gemini 3.8 Flash MEDIUM judge)

Commands / artifacts under `.data/experiment_tab_sweep/` (`run_sweep.py`,
`measurements.json`, `scorecard.json`, `joined_labels.jsonl`,
`unique_review_packet/`).

Two 180 s slices (360.0 s source total): Vietcetera EP6 pilot
`youtube:PXEtB-CsvSw` and EP5 dancer `youtube:3Nll-JLzvvE`. Device
`cuda:0` via `.venv-sortformer`. Stage 5 Gemini/VibeVoice off so the teacher
was independent. Consensus off (DiariZen is CPU torch on this host). Challenge
`youtube:H0VpjeULCck` unused. Six configs:

| Config | Emitted (2–15 s) | Pass / reject | Pass s | Acc. min / src h | Clip S/E rejects | Notes |
|---|---:|---:|---:|---:|---:|---|
| `tab_defaults` | 37 | 30 / 7 | 167.140 | 27.857 | 0 / 0 | Library collar defaults; music 6, secondary 1 |
| `defaults_smartseg` | 39 | 28 / 11 | 157.490 | 26.248 | 1 / 1 | Smart seg alone; more music/effects children |
| **`recipe_nolock`** | **39** | **31 / 8** | **184.918** | **30.820** | **1 / 2** | Soft onset/collar + smart seg; **winner** |
| `extract_lock` | 10 | 9 / 1 | 42.496 | 7.083 | 0 / 0 | PhoWhisper-small lock; PXEtB → 1 clip / 0 pass s |
| `completeness_recipe` | 7 | 6 / 1 | 28.190 | 4.698 | 0 / 1 | Docs recipe with lock; yield collapse |
| `recipe_energy` | 5 | 5 / 0 | 18.174 | 3.029 | 0 / 0 | Energy+lock zeroed PXEtB |

Winner defects among rejects (not in the pass pool): music 4, clipped_word_end 2,
clipped_word_start 1, secondary_speaker 1, reverberation 2. Pass-pool clipping
was zero for every config (rejects are discarded). Word lock dropped interview
turns as `word_boundary_conflicts_with_safe_bounds` (PXEtB extract_lock:
20 → 1 emitted). Energy snapping plus lock zeroed PXEtB.

Teacher: 110 unique clips, all `gemini-3.8-flash`, `thinkingLevel=MEDIUM`,
reviewer `gemini-3.8-flash:MEDIUM`, rubric `tts-v1`, provenance
`user_accepted_gemini_ground_truth`, service tier `standard`. Usage:
42,143 prompt + 16,651 answer + 38,246 thinking = 97,040 tokens. Latency
min 2.764 s / mean 4.732 s / max 10.122 s / sum 520.543 s. Pipeline wall
sum across 12 runs: see `run_summaries.json`.

Studio Experiment tab Reset / `/api/experiment/status` defaults now match
`recipe_nolock`. Library `ZeroContaminationConfig` dataclass constants are
unchanged for non-UI callers. Docs: `04_zero_contamination_diarization.md`,
`08_model_parameters_and_tradeoffs.md`. Not a production quality claim; does
not cover music-backed vlogs; music leakage still needs a later teacher or
Stage 5 gate.

Durable reuse of the paid labels (do not discard): unique-by-SHA export
`.data/tts_strategy/experiment_tab_sweep_20260908/labeled.jsonl` (110 clips,
88 pass / 22 reject; audio copies under `.../audio/`; evaluation JSON under
`.../gemini_medium/`). Combined with v2 via `audit_tts_data.py combine` into
`.data/tts_strategy/pool_20260908_v3/labeled_pool.jsonl`: **241 clips /
10 recordings** (148 pass / 93 reject). SHA dedupe dropped 2 byte-identical
overlaps already in v2. Recording counts after merge: `3Nll-JLzvvE` 85,
`PXEtB-CsvSw` 48, `j83rzAzRDAI` 22, `QBml8L3wS3Q` 18, `i0zYcXBjytE` 14,
`Oa-mVxGS4cw` 13, `fwN5VT_QxkY` 13, `zK3qFnKZFRo` 13, `lfIbjICmfW0` 11,
`NI8JVXNWlN8` 4. Calibration freeze unchanged (`fwN5VT_QxkY` + `Oa-mVxGS4cw`).
v2 left intact. Still not a production quality claim.

### 9. Unified benchmark gold (challenge holdout merged)

User request: merge the former held-out Khanh Vy challenge into the larger
MEDIUM gold set because n=31 was too small for benchmarking.

Exported `.data/tts_strategy/khanhvy_challenge_20260908/labeled.jsonl`:
31 phase1 challenge originals + 16 boundary-relock children, all
`youtube:H0VpjeULCck`, Gemini 3.8 Flash MEDIUM (11 pass / 36 reject among the
47). `audit_tts_data.py combine --include-reserved-challenge` then built
`.data/tts_strategy/pool_20260908_v4/labeled_pool.jsonl` (also copied to
`.data/tts_strategy/gold_benchmark_20260908/labeled_pool.jsonl`):

- **288 clips / 11 recordings** (159 pass / 129 reject)
- Adds `youtube:H0VpjeULCck`: 47
- Prior v3 recordings unchanged in count except the new recording row
- v1/v2/v3 left intact

Trade-off: this file is a bigger gold/benchmark set, but it is **not** a
source-disjoint unseen test anymore. Reserve a fresh recording later when
unseen-source generalization must be measured. `combine` still refuses
challenge rows unless `--include-reserved-challenge` is passed. `extract`
still blocks harvesting that recording by default.

### 10. Local verifier benchmark (E2B V3 vs 288-clip MEDIUM gold)

Command (ROCm GPU via `.venv-sortformer`; transformers upgraded to `5.16.1`
plus CPU `torchvision==0.28.0` so Gemma 4 processor imports; weights resolved
through `HF_HOME=.data/huggingface` symlink to the existing
`~/.cache/huggingface/hub/models--google--gemma-4-E2B-it` cache):

```bash
.venv-sortformer/bin/python scripts/evaluate_verifier.py \
  --backend hf_local \
  --model google/gemma-4-E2B-it \
  --adapter-path .data/distillation/checkpoints_e2b_v3/best_adapter \
  --device cuda:0 \
  --torch-dtype bfloat16 \
  --input .data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl \
  --output-report .data/tts_strategy/gold_benchmark_20260908/reports/e2b_v3_vs_gemini38_medium.md \
  --output-json .data/tts_strategy/gold_benchmark_20260908/reports/e2b_v3_vs_gemini38_medium.json \
  --export-csv .data/tts_strategy/gold_benchmark_20260908/reports/e2b_v3_vs_gemini38_medium.csv
```

Ground truth: Gemini 3.8 Flash MEDIUM on `gold_benchmark_20260908` (159 pass /
129 reject). Student: Gemma 4 E2B + LoRA V3.

| Metric | Value |
|---|---:|
| Evaluated / success | 288 / 288 |
| Avg latency | 3.52 s |
| Model pass / reject | **288 / 0** |
| Agreement vs Gemini | **55.2%** (159/288) |
| Precision / recall / F1 | 55.2% / 100.0% / **71.1%** |
| True pass / true reject | 159 / 0 |
| Contamination leaks / false rejects | **129 / 0** |

Same all-pass failure mode as the 31-cut Khanh Vy probe (38.7% agreement there):
agreement on this gold set is exactly the Gemini pass rate because the student
never rejects. Defect codes among the 129 leaks (a clip may have several):
music 61, secondary_speaker 37, clipped_word_end 29, reverberation 17,
sound_effect 16, clipped_word_start 13, overlapping_speech 8, excessive_noise 3.
Largest leak source: `youtube:H0VpjeULCck` (36).

Reports also copied to
`.data/distillation/reports/finetuned_e2b_v3_vs_gold_benchmark_20260908.{md,json,csv}`.
Root `.venv` still has CUDA torch with `cuda=False` on this host; do not use it
for GPU Gemma inference until ROCm wheels are restored there.

### 11. Gemini 3.5 Flash-Lite vs 288-clip MEDIUM gold

Yes — reasoning effort is controllable via
`evaluate_verifier.py --reasoning-effort {none,low,medium,high}` → API
`thinkingConfig.thinkingLevel`. Probed on `gemini-3.5-flash-lite`: `none`,
`low`, and `medium` all succeed. This run used **medium** to match the gold
teacher effort.

```bash
uv run --no-sync python scripts/evaluate_verifier.py \
  --backend gemini \
  --model gemini-3.5-flash-lite \
  --reasoning-effort medium \
  --concurrency 8 \
  --input .data/tts_strategy/gold_benchmark_20260908/eval_input.jsonl \
  --output-report .data/tts_strategy/gold_benchmark_20260908/reports/gemini35_flash_lite_medium_vs_gemini38_medium.md \
  --output-json .data/tts_strategy/gold_benchmark_20260908/reports/gemini35_flash_lite_medium_vs_gemini38_medium.json \
  --export-csv .data/tts_strategy/gold_benchmark_20260908/reports/gemini35_flash_lite_medium_vs_gemini38_medium.csv
```

| Metric | Flash-Lite MEDIUM | E2B V3 LoRA (ckpt 10) |
|---|---:|---:|
| Agreement vs 3.8 Flash MEDIUM | **63.2%** (182/288) | 55.2% (159/288) |
| Precision / recall / F1 | 61.0% / 92.5% / **73.5%** | 55.2% / 100% / 71.1% |
| Model pass / reject | 241 / **47** | 288 / 0 |
| True pass / true reject | 147 / **35** | 159 / 0 |
| Contamination leaks / false rejects | 94 / **12** | 129 / 0 |
| Avg latency | 2.81 s | 3.52 s |

Lite can reject (unlike E2B V3), so agreement beats the all-pass baseline, but
94/129 Gemini rejects still leak. Dominant missed codes among leaks: music 40,
clipped_word_end 25, secondary_speaker 23. Historical 31-cut lite probe
(no thinkingConfig) was 51.6% agreement; this MEDIUM gold run is stronger but
not teacher-grade.

**Cost note:** That 288-clip Flash-Lite run completed before automatic cost
instrumentation. Per-call `_usage`/`_cost` is now logged by `GeminiVerifier`
(shared rate card `src/diarization/gemini_pricing.py`, as of 2026-09-04 paid
Standard). A post-fix smoke of 2 clips cost **$0.000805** (~$0.12 extrapolated
to 288 if thinking stays near zero). Re-runs should record exact totals in the
report JSON/Markdown.

### 12. Automatic Gemini cost logging + portable gold eval

`GeminiVerifier.verify` now always attaches `_usage` / `_cost`, accumulates a
thread-safe session total, and logs running USD. `evaluate_verifier.py` writes
usage/cost into summary JSON, Markdown, CSV (`cost_usd`), and the console.
`OverlapVerifier` reuses the same pricing helpers.

Portable full-gold runner for another machine (preflight + materialize audio
into `.data/tts_strategy/gold_benchmark_20260908/audio/` + evaluate):

```bash
export GEMINI_API_KEY=...
# sync repo + .data/ first (or rely on --materialize-audio after local paths resolve)
./scripts/run_gold_verifier_eval.sh gemini-3.5-flash-lite medium 8
# shard / resume:
LIMIT=100 OFFSET=0 ./scripts/run_gold_verifier_eval.sh
RESUME_JSON=path/to/partial.json ./scripts/run_gold_verifier_eval.sh
```

Also on `evaluate_verifier.py`: `--check-audio-only`, `--materialize-audio`,
`--limit`, `--offset`, `--resume-json`, `--allow-missing-audio`.

Reports also at
`.data/distillation/reports/gemini35_flash_lite_medium_vs_gold_benchmark_20260908.{md,json,csv}`.

## Continuation rules

Keep this section current in every meaningful checkpoint. Record exact commands,
**every measured number** (counts, durations, keep/drop fractions, rates, token
usage, latencies, wall times), caveats, pending human/hardware dependencies, and
next bounded tasks. Do not round a measured value away or replace it with a
qualitative summary. Persist the same figures under `.data/` (for example
`extract_*/measurements.json`) so a later run can diff them. Never claim
completion just because a script or plan exists. Preserve runtime artifacts
under `.data/`; pushes contain source/docs only. Do not infer human judgments
from Gemini. Follow AGENTS.md, including no test cases without request.
