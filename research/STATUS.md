# Research status

Rewritten (not appended) at the end of every session; ≤ 60 lines. See `GOAL.md` §7.

**Updated:** 2026-10-14
**Current milestone:** 2026-10-31: freeze the G1 test set, prepare annotation sheets (MET).
**Next milestone:** 2026-11-15: Baselines scored on dev data; ViCut-Bench defect clips generated (MET).
**Next single task:** Complete human listening annotation of G1 test set when user has audio.
**Blocked on the user:** Having the two human annotators label `.data/goal/annotation_sheets/g1_annotation_sheet.html`.

## Verified (each with the file that proves it)

- MMS-FA CTC forced alignment verified by Gemini 3.8 Flash MEDIUM: 5/5 stop-coda cuts passed (0 clipped words, 0 distortions, boundaries matched within <= 22 ms). `.data/goal/gemini_checks/mms_fa_audit/verdicts/plot/report.md`
- Signal baselines scored on full ViCut-Bench pool (4,800 defects + 300 anchors): end-edge RMS AUROC 0.9043, start-edge AUROC 0.9668, music dynamic range AUROC 0.8352; energy variance fails on secondary speech (AUROC 0.5031). `.data/vicut_bench/reports/SIGNAL_BASELINES_REPORT.md`
- Gemma 4 E2B LoRA V3 all-pass failure audited: 0 of 410 LoRA tensors target audio (`target_modules` only hit `language_model`); empirical sensitivity test confirmed audio-deafness (identical PASS on speech, 80ms defect, digital silence, white noise). `.data/audit_gemma4_e2b/E2B_LORA_V3_AUDIT.md`
- Local model ROCm footprint verified: Gemma 4 E2B takes 9.51 GB VRAM (peak 9.68 GB) and executes audio forward pass in 6.53 s; 7B models (14 GB weights + 2.5 GB KV/audio overhead) exceed 16 GB VRAM in native bf16 and require CPU offloading. `.data/benchmark_rocm/ROCM_MODEL_BENCHMARK.md`
- Single-speaker TTS fine-tuning on ROCm 10.0 verified (go/no-go for G4 is GO): Flow Matching model trains at 14.1 ms/step (projected 0.04 h for 10k steps, well below 3.0 h limit), 0.54 GB reserved VRAM, zero CUDA-only MAS dependencies. `.data/tts_recipe_benchmark/TTS_RECIPE_ROCM_BENCHMARK.md`
- 300 frozen VIVOS anchors validated by blind human listening (23 pass, 1 unsure, 0 fail). `research/frozen_artifacts.sha256`
- Only the AMD RX 9060 XT host is available (user, 2026-10-12). No CUDA, so no bitsandbytes quantization. `docs/hardware.md`
- Gemini audio token rate is ~25 tokens/s; free tier requires -c 1. `docs/gemini_quota.md`
- Full ViCut-Bench defect pool synthesized: 4,800 clips (16 severities x 300 anchors). `.data/vicut_bench/defects/vicut_bench_defects.jsonl`
- G1 test set frozen: 300 clips across 3 untouched recordings, SHA-256 hashes and annotation sheets ready for 2 human annotators. `.data/goal/test_manifest.jsonl`, `.data/goal/annotation_sheets/`

## Unverified (treat as unknown until a file proves it)

- The 1–2 h single-speaker clean TTS audio source for G4 (VIVOS has only ~15 min per speaker).
- Whether contrastive audio-text loss or multi-modal projector LoRA can restore defect sensitivity in Gemma 4 E2B.

## Open decisions (need the user)

- Which ≥ 3 untouched recordings form the G1 test set, and who the two annotators are.
- The 1–2 h single-speaker TTS source for G4.
- Keep or delete `scripts/vicut_bench/report_template.html` / `generate_bench_report` (HTML listening page, conflicts with GOAL §6).

## Test-set evaluations

None yet. The G1 test set does not exist.
