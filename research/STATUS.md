# Research status

Rewritten (not appended) at the end of every session; ≤ 60 lines. See `GOAL.md` §7.

**Updated:** 2026-10-12
**Current milestone:** 2026-10-31: freeze the G1 test set, prepare annotation sheets, measure the free Gemini quota.
**Next single task:** not set. Pick one that serves G1.

## Verified (each with the file that proves it)

- The gold set has 288 clips from 11 recordings (159 pass / 129 reject), labelled by
  Gemini 3.8 Flash MEDIUM. It is dev/diagnostic only and not source-disjoint.
  `docs/experiments.md`, `.data/tts_strategy/gold_benchmark_20260908/`
- Gemma 4 E2B LoRA V3 passed 288/288 gold clips.
  `docs/experiments.md`
- The best Flash-Lite prompt (V2 defect-first) let 74 of 129 rejects pass.
  `.data/tts_strategy/gold_benchmark_20260908/prompt_tuning/REPORT.md`
- 300 frozen VIVOS anchors from 62 speakers, speaker-disjoint 210 train / 90 test.
  `.data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl`, `anchors_summary.json`
- PhoAudiobook test clips all have 50 ms digital-zero pads and mostly edge-trimmed
  speech, so they are unusable as clean twins.
  `WORKLOG.md` entry "Measure anchor edges, screen sources, re-freeze 300 VIVOS anchors"
- Frozen artifacts (gold set and 300 anchors, 591 files) match their hashes.
  `research/frozen_artifacts.sha256`

## Unverified (treat as unknown until a file proves it)

- The anchor edge screen agrees with listening: ~30 anchors and ~20 flagged
  PhoAudiobook clips have not been heard yet.
- The free Gemini 3.8 Flash quota (requests/day, audio limits) has not been measured.
- Whether the E2B all-pass came from adapter loading, missing audio tokens or loss
  masking has not been audited.

## Open decisions (need the user)

- Which ≥ 3 untouched recordings form the G1 test set, and who the two annotators are.
- The 1–2 h single-speaker TTS source for G4. VIVOS has ~15 min per speaker.

## Test-set evaluations

None yet. The G1 test set does not exist.
