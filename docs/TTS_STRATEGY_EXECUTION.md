# TTS strategy execution log

## Current handoff — 2026-09-08

- Strategy: [TTS_PRODUCTION_STRATEGY.md](TTS_PRODUCTION_STRATEGY.md).
- Active phase: 1, establish evidence and an executable audit workflow.
- Authorization: user requested execution with regular commits and pushes.
- Working tree was clean on `main`, tracking `origin/main`, before work began.
- User clarification: Gemini 3.8 Flash MEDIUM is accepted as human-quality ground
  truth; API hearing is explicitly authorized. Separate human review is not a gate.
- Plan commit `db025ad` is pushed to `origin/main`.
- Completed MEDIUM API audits: 31 legacy challenge clips and 97 source-resolved
  training-pool clips. Responses, labels, configuration/input hashes and usage are cached.
- No training or test cases have run. Boundary behavior is now stricter (see below);
  acoustic improvement has not yet been measured.
- Next: run real-audio boundary evaluation with local timestamp evidence, grow the
  source-disjoint dataset, then fit/calibrate a specialized acoustic baseline.

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

- Word locking only expands words intersecting the original edge; padding no
  longer propagates expansion into successive words. If speaker-safe clamping
  still cuts a recognized word, or no complete word remains, reject with an audit.
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

## Continuation rules

Keep this section current in every meaningful checkpoint. Record exact commands,
counts, caveats, pending human/hardware dependencies, and next bounded tasks.
Never claim completion just because a script or plan exists. Preserve runtime
artifacts under `.data/`; pushes contain source/docs only. Do not infer human
judgments from Gemini. Follow AGENTS.md, including no test cases without request.
