# Research status

Rewritten (not appended) at the end of every session; ≤ 60 lines. See `GOAL.md` §7.

**Updated:** 2026-11-06
**Current milestone:** 2026-10-31: freeze G1 test set, prepare annotation sheets (IN PROGRESS; pool sampled, frozen in SHA-256, packaging complete, pending verifier audit).
**Next milestone:** 2026-12-01: Candidate method beats best baseline on dev with CI excluding 0 (IN PROGRESS; fallback to benchmark/audit paper if missed).
**Next single task:** Complete independent verification of Phase 0 pool by Claude Opus 5.5 subagent per PLAN.md.
**Blocked on the user:** Approve new G1 recording selection and frozen-hash commit (GOAL_CHANGE_APPROVED=1).

## Verified (each with the file that proves it)

- Regionally stratified G1 test pool sourced & frozen (PLAN.md Phase 0): 12 recordings (4 North, 4 Central, 4 South), 360 blind clips (30/rec), speaker disjointness verified with NeMo TitaNet-L on ROCm (centroid vs dev-288 max 0.6867 < 0.6944, ViCut-Bench max 0.6777 < 0.6944, archived max 0.6922 < 0.6944, TTS max 0.6681 < 0.6944). Blind clips in `.data/goal/g1_blind_audio/`, answer key in `.data/goal/g1_blind_audio/key.jsonl`, manifest in `.data/goal/test_manifest.jsonl`, 10 review batches (36 clips each) in `.data/goal/review/g1_test/batch_{01..10}/` strictly compliant with GOAL §6 (strictly 3 files: REVIEW.md, review.csv, playlist.m3u). Hashes frozen in `research/frozen_artifacts.sha256`. Comprehensive audit report in `.data/goal/DISJOINTNESS_REPORT.md`.
- Synthetic clean-twin (`0 ms`) boundary calibration resolved (Hypothesis A confirmed): replacing in-sample 50/50 mixture 95% thresholds with speaker-disjoint 5-fold OOF thresholds on `clean_tight_tr` (and per-anchor `max(anchor, tight)`) reduces held-out `clean_tight_twin` (n=90) FRR from 35.6% to **3.3%** (`composite`, raw-anchor FRR **1.1%**) and **4.4%**/**3.3%** (`boundary_only`, raw-anchor FRR **0.0%**) with zero weight changes. `.data/vicut_bench/reports/composite_probe_calibration_20261016/COMPOSITE_PROBE_GATE_REPORT.md`
- Real-dev (`dev-288`) probability-scale mismatch and fixed-70%-yield marginal-precision ceiling diagnosed (Hypothesis B confirmed): at 70% minute yield (`t=0.998703`), `composite` (`R=0.2872`) gives 0/93 rejections to `wavlm` boundary heads (`gemma4_audio:background_music` 73/93, `secondary_speaker` 20/93). Same report.
- Fixed composite dev experiment is negative (288 clips, 11 recordings): at 69.97% accepted-minute yield, R=0.2872 (56/195), versus PhoWhisper R=0.2792 at 69.65%; paired recording-bootstrap baseline-minus-composite CI [-0.0403, 0.0474], 1,000 replicates, seed 42 (seed 123 also crosses zero). `.data/vicut_bench/reports/composite_probe_gate/COMPOSITE_PROBE_GATE_REPORT.md`
- Phase 3 layer-wise encoder probing on ViCut-Bench (5,688 clips across `wavlm`, `phowhisper`, `gemma4_audio`) confirmed Hypothesis H3: Secondary speaker reaches test AUROC **0.9495** on `gemma4_audio` L4; 20 ms hop encoders (`wavlm` L12: **0.9459** AUROC) beat 40 ms on `clipped_word_end`. `.data/vicut_bench/reports/phase3_probes/PHASE3_PROBE_REPORT.md`
- Single-speaker TTS fine-tuning on ROCm 10.0 verified (go/no-go for G4 throughput is GO): Flow Matching model trains at 21.9 ms/step, 0.54 GB reserved VRAM. `.data/tts_recipe_benchmark/TTS_RECIPE_ROCM_BENCHMARK.md`
- 300 frozen VIVOS anchors validated by blind human listening (23 pass, 1 unsure, 0 fail). `research/frozen_artifacts.sha256`
- Full ViCut-Bench defect pool synthesized: 4,800 clips (16 severities x 300 anchors). `.data/vicut_bench/defects/vicut_bench_defects.jsonl`

## Unverified (treat as unknown until a file proves it)

- Independent verifier audit (Claude Opus 5.5) sign-off on Phase 0 pool integrity.
- The 1–2 h clean/defect-bearing single-speaker audio source for G4 (VIVOS has only ~15 min per speaker).
- Whether combining content probes with deterministic edge energy reduces between-recording variance on `dev-288`.
- Sensitivity of the Flow Matching TTS model to data curation quality (positive control run required).

## Open decisions (need the user)

- Human commit approval with `GOAL_CHANGE_APPROVED=1` for newly frozen G1 artifacts.
- Completion of 2-annotator G1 labelling after verifier PASS.
- The 1–2 h single-speaker TTS source for G4.

## Test-set evaluations

None yet. The G1 test set has been sampled and frozen, awaiting human listening labels.
