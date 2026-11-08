# Research status

Rewritten (not appended) at the end of every session; ≤ 60 lines. See `GOAL.md` §7.

**Updated:** 2026-11-08
**Current milestone:** 2026-12-01: Candidate method dev evaluation & benchmark characterization (Branch B Fallback triggered; candidate locked; proceed to Phase 2 G1 characterization & Phase 3 benchmark paper).
**Next milestone:** 2026-12-15: Complete blind 2-annotator human review on G1 test pool (`.data/goal/review/g1_test/batch_{01..10}/`).
**Next single task:** Complete 2-annotator blind human labeling on G1 review batches.
**Blocked on the user:** Completion of 2-annotator G1 human labelling (`.data/goal/review/g1_test/batch_{01..10}/`).

## Verified (each with the file that proves it)

- Phase 1 Single Candidate Hybrid Gate evaluated on dev-288 and ViCut-Bench (`.data/vicut_bench/reports/dev_hybrid_gate/DEV_HYBRID_GATE_REPORT.md`): Pre-gate (RMS < -50 dBFS), boundary arm (10 ms edge ratios @ 95% specificity on 5-fold OOF `clean_tight_tr`: $\tau_\text{start}=0.4235, \tau_\text{end}=0.2538$), content arm (PhoWhisper L4 `mean_std` pooling logistic regression, OOF AUROC 0.9340 secondary speaker, 0.9954 music). Continuous score $s = \text{flag\_edge} + p_\text{content}$, lexicographical tuple `(pre_gate, flag_edge, p_content, clip_id)`.
- Dev-288 competition: Candidate defect rate: vs `flash_lite` ($37.66\%$ vs $39.00\%$, $\Delta R = -1.34\%$ pt at ~83.8% yield), vs `signal_3feat` ($30.49\%$ vs $32.64\%$, $\Delta R = -2.15\%$ pt at 70% yield), vs `signal_only` ($30.49\%$ vs $29.10\%$, $\Delta R = +1.39\%$ pt), and vs `recipe_nolock` ($41.84\%$ vs $44.79\%$ at 90% yield). Paired recording-bootstrap 95% CIs cross 0 against `flash_lite` ([-0.0600, +0.0830], [-0.0607, +0.0812]) and `signal_3feat` ([-0.0636, +0.0843], [-0.0625, +0.0782]). Pre-registered **Branch B (Fallback)** triggered; candidate locked as baseline per PLAN.md line 99.
- ViCut-Bench G3 mechanism verified on held-out test split (90 anchors / 19 speakers): Clean-twin 0 ms cut FRR = 3.33% start, 2.22% end, 5.56% union, 0.00% raw anchors. Monotonic per-severity AUROC vs tight twins up to 0.9317 (`clipped_word_end` 80ms), 0.9389 (`clipped_word_start` 80ms), 0.9978 (`secondary_speaker` 6dB), 0.9999 (`background_music` 12dB).
- Real dev-288 component ablation documented across all 5 conditions: Full Hybrid (30.49%), Content only (27.55%), Boundary only (33.49%), Full w/o Pre-gate (30.49%), Pre-gate only (44.79%). Same report.
- Independent verifier audit (Claude Opus 5.5) sign-off on Phase 0 pool integrity: **VERDICT: PASS** (`.data/goal/DISJOINTNESS_REPORT.md`).
- Regionally stratified G1 test pool sourced & frozen (12 recs, 360 blind clips, 4:4:4 regional balance, speaker disjointness NeMo TitaNet max 0.6867 < 0.6944). Hashes frozen in `research/frozen_artifacts.sha256`.
- Single-speaker TTS fine-tuning on ROCm 10.0 verified (go/no-go for G4 throughput is GO): Flow Matching model trains at 21.9 ms/step. `.data/tts_recipe_benchmark/TTS_RECIPE_ROCM_BENCHMARK.md`
- 300 frozen VIVOS anchors validated by blind human listening. `research/frozen_artifacts.sha256`
- Full ViCut-Bench defect pool synthesized: 4,800 clips (16 severities x 300 anchors). `.data/vicut_bench/defects/vicut_bench_defects.jsonl`
- G4 1–2 h single-speaker TTS source sourced, profiled, disjointness-verified, edge-filtered into clean training split (1.00 h, 339 clips), and audited by Claude Opus 5.5 (`.data/downstream_experiments/G4_SPEAKER_PROFILING_REPORT.md`).

## Unverified (treat as unknown until a file proves it)

- G1 test-set defect rate and curator performance under blind human review (awaiting human labels).
- Downstream TTS sensitivity to data curation quality (positive control run required in Phase 4).

## Open decisions (need the user)

- Completion of 2-annotator G1 human labelling (`.data/goal/review/g1_test/batch_{01..10}/`).
- Final user sign-off on the 1–2 h single-speaker TTS source for G4 (`qhuy242` Southern Male primary, edge-filtered 1.00 h partition; `Diễm_Hân` PhoAudiobook secondary).

## Test-set evaluations

None yet. The G1 test set has been sampled and frozen, awaiting human listening labels.

