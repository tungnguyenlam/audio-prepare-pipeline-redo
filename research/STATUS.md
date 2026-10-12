# Research status

Rewritten (not appended) at the end of every session; ≤ 60 lines. See `GOAL.md` §7.

**Updated:** 2026-10-12
**Current milestone:** 2026-10-31: freeze the G1 test set, prepare annotation sheets.
**Next single task:** Read back the user's labels for the edge-screen batch
(`.data/goal/review/edge_screen_anchors/review.csv`) and apply the pre-registered
decision rules (WORKLOG "Pre-registration: edge-screen listening batch").
**Blocked on the user:** listening to that batch (~11 min); choosing the G1 recordings.

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
  speech by measurement (274/279 `tight_end`); not yet confirmed by listening.
  `.data/vicut_bench/anchors/measured_candidates.jsonl`
- Frozen artifacts (gold set and 300 anchors, 591 files) match their hashes
  (re-checked 2026-10-12). `research/frozen_artifacts.sha256`
- Only the AMD RX 9060 XT host is available (user, 2026-10-12). No CUDA, so no
  bitsandbytes quantization. `docs/hardware.md`
- Gemini audio token rate is ~25 tokens/s; free tier requires -c 1 and 2-5 s backoff
  on 503 demand spikes; latency is 1.75 s (Flash-Lite) to 34.1 s (Flash with full rubric).
  `docs/gemini_quota.md`, `.data/goal/gemini_checks/quota_probe/`
- A blind 40-clip edge-screen listening batch exists (24 anchors, 10 flagged VIVOS,
  6 flagged PhoAudiobook; seed 42; key kept apart). Unlabelled.
  `.data/goal/review/edge_screen_anchors/{REVIEW.md,review.csv,playlist.m3u,key.jsonl}`

## Unverified (treat as unknown until a file proves it)

- The anchor edge screen agrees with listening (batch above awaits labels).
- Whether any TTS recipe fine-tunes on ROCm within ≤ 3 h (go/no-go for G4).
- Whether Qwen2-Audio-7B, MiniCPM-o or Kimi-Audio load in bf16 within 16 GB.
- Whether the E2B all-pass came from adapter loading, missing audio tokens or loss
  masking has not been audited.

## Open decisions (need the user)

- Which ≥ 3 untouched recordings form the G1 test set, and who the two annotators are.
- The 1–2 h single-speaker TTS source for G4. VIVOS has ~15 min per speaker.
- Keep or delete `scripts/vicut_bench/report_template.html` / `generate_bench_report`
  (HTML listening page, conflicts with GOAL §6).

## Test-set evaluations

None yet. The G1 test set does not exist.
