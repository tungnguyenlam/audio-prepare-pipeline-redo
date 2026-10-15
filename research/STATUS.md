# Research status

Rewritten (not appended) at the end of every session; ≤ 60 lines. See `GOAL.md` §7.

**Updated:** 2026-10-15
**Current milestone:** 2026-10-31: freeze the G1 test set, prepare annotation sheets (MET).
**Next milestone:** 2026-12-01: Candidate method beats best baseline on dev with CI excluding 0 (IN PROGRESS).
**Next single task:** Build and evaluate the composite local gatekeeper (combining frozen encoder probe heads + speaker-drift/edge signals) on the 288-clip real dev set with recording-level bootstrap CIs (G2).
**Blocked on the user:** Having the two human annotators label `.data/goal/annotation_sheets/g1_annotation_sheet.html`.

## Verified (each with the file that proves it)

- Phase 3 layer-wise encoder probing on ViCut-Bench (5,688 clips across `wavlm`, `phowhisper`, `gemma4_audio`) confirmed Hypothesis H3:
  - Secondary speaker (`mean_std` pooling) reaches test AUROC **0.9495** [0.9307, 0.9701] on `gemma4_audio` L4, **0.8869** on `phowhisper` L4, and **0.8835** on `wavlm` L2, beating signal baseline (`0.5055`) by +0.38 to +0.44 AUROC; generalizes to unseen test-split donor speakers (**0.9446** on `gemma4_audio` L4, $n=96$).
  - Clean-twin `0 ms` tight-cut control (`[speech_onset_s, speech_offset_s]`) with strict single-edge pooling (`end_edge` only) reveals a 40 ms vs 20 ms frame-rate bottleneck on `clipped_word_end`: 20 ms hop encoders (`wavlm` L12: **0.9459** AUROC, `phowhisper` L12: **0.9077**) beat the 10 ms energy ratio baseline (`0.7834`) by +0.1612 and +0.1220 AUROC, whereas 40 ms subsampled `gemma4_audio` L2 drops to **0.7080** AUROC (10.0% recall at 10 ms).
  - Zero-shot transfer of synthetic-trained probe heads to the 288-clip dev set (`gold_benchmark_20260908`) reaches **0.8303** (`phowhisper`), **0.8199** (`gemma4_audio`), and **0.8146** (`wavlm`) reject AUROC, lowering $R$ from 0.4479 to 0.2886 at 70% yield. `.data/vicut_bench/reports/phase3_probes/PHASE3_PROBE_REPORT.md`
- MMS-FA CTC forced alignment verified by Gemini 3.8 Flash MEDIUM: 5/5 stop-coda cuts passed (0 clipped words, 0 distortions, boundaries matched within <= 22 ms). `.data/goal/gemini_checks/mms_fa_audit/verdicts/plot/report.md`
- Signal baselines scored on full ViCut-Bench pool (4,800 defects + 300 anchors): end-edge RMS AUROC 0.9043, start-edge AUROC 0.9668, music dynamic range AUROC 0.8352; energy variance fails on secondary speech (AUROC 0.5031). `.data/vicut_bench/reports/SIGNAL_BASELINES_REPORT.md`
- Gemma 4 E2B LoRA V3 all-pass failure audited: 0 of 410 LoRA tensors target audio (`target_modules` only hit `language_model`); empirical sensitivity test confirmed audio-deafness. `.data/audit_gemma4_e2b/E2B_LORA_V3_AUDIT.md`
- Local model ROCm footprint verified: Gemma 4 E2B takes 9.51 GB VRAM (peak 9.68 GB) and executes audio forward pass in 6.53 s; 7B models exceed 16 GB VRAM in native bf16. `.data/benchmark_rocm/ROCM_MODEL_BENCHMARK.md`
- Single-speaker TTS fine-tuning on ROCm 10.0 verified (go/no-go for G4 is GO): Flow Matching model trains at 14.1 ms/step (projected 0.04 h for 10k steps), 0.54 GB reserved VRAM. `.data/tts_recipe_benchmark/TTS_RECIPE_ROCM_BENCHMARK.md`
- 300 frozen VIVOS anchors validated by blind human listening (23 pass, 1 unsure, 0 fail). `research/frozen_artifacts.sha256`
- Full ViCut-Bench defect pool synthesized: 4,800 clips (16 severities x 300 anchors). `.data/vicut_bench/defects/vicut_bench_defects.jsonl`
- G1 test set frozen: 300 clips across 3 untouched recordings, SHA-256 hashes and annotation sheets ready for 2 human annotators. `.data/goal/test_manifest.jsonl`, `.data/goal/annotation_sheets/`

## Unverified (treat as unknown until a file proves it)

- The 1–2 h single-speaker clean TTS audio source for G4 (VIVOS has only ~15 min per speaker).
- Whether combining `wavlm` L12 boundary probes with `gemma4_audio` L4 / `phowhisper` L4 speaker+music probes beats all single-model baselines on the 288-clip dev set with recording-bootstrapped 95% CI excluding 0.

## Open decisions (need the user)

- Completion of 2-annotator human labelling on `.data/goal/annotation_sheets/g1_annotation_sheet.html`.
- The 1–2 h single-speaker TTS source for G4.
- Keep or delete `scripts/vicut_bench/report_template.html` / `generate_bench_report` (HTML listening page, conflicts with GOAL §6).

## Test-set evaluations

None yet. The G1 test set awaits human annotations.
