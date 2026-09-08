# TTS strategy execution log

## Current handoff — 2026-09-08

- Strategy: [TTS_PRODUCTION_STRATEGY.md](TTS_PRODUCTION_STRATEGY.md).
- Active phase: 1, establish evidence and an executable audit workflow.
- Authorization: user requested execution with regular commits and pushes.
- Working tree was clean on `main`, tracking `origin/main`, before work began.
- User clarification: Gemini 3.8 Flash MEDIUM is accepted as human-quality ground
  truth; API hearing is explicitly authorized. Separate human review is not a gate.
- Plan commit `db025ad` is pushed to `origin/main`.
- Offline inventory completed. A fresh 31-clip MEDIUM full-rubric API audit is
  running; each completed response/label is cached under the packet directory.
- No training or test cases have run. Pipeline behavior is unchanged so far.
- Next: finish teacher audit, record scores and defects, restore source lineage
  before replacement splits, then perform independent boundary engineering.

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

New script: `scripts/audit_tts_data.py` (`prepare`, `teacher`, `report`). Artifacts:
`.data/tts_strategy/phase1_20260908/`. No inference runs during inventory; no
pytest/unit cases were written or run. Production risk remains unqualified due
to challenge sampling, not due to a requirement for additional human review.

## Continuation rules

Keep this section current in every meaningful checkpoint. Record exact commands,
counts, caveats, pending human/hardware dependencies, and next bounded tasks.
Never claim completion just because a script or plan exists. Preserve runtime
artifacts under `.data/`; pushes contain source/docs only. Do not infer human
judgments from Gemini. Follow AGENTS.md, including no test cases without request.
