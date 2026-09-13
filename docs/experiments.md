# Experiments and governing decisions

[← Index](README.md)

Historical record, condensed. The scripts that produced these numbers
(`audit_tts_data.py`, `build_distillation_dataset.py`, `train_verifier.py`, the
root `evaluate_verifier.py`, the `src/` library and the Studio web UI) were removed
in commit `d805b62` when the repository became standalone commands. Their outputs
survive under `.data/` (tracked reports listed at the end) and in git history.
Nothing below is a production quality claim.

## Goal and decision rules

- **Target.** 2–15 s Vietnamese speech clips with exactly one speaker, complete
  first/last words, no overlap or secondary intrusion, no unacceptable
  music/effects/noise, preserved voice quality. Precision over recall.
- **Metric.** R = bad accepted clips / accepted clips (a clip counts once however
  many defects it has). Target < 10 %, preferably < 5 %, together with accepted
  minutes per source hour and runtime. Accepting nothing cannot win (R undefined).
  Teacher agreement, DER, accuracy and F1 are diagnostics, not R.
- **Reference annotator.** Gemini 3.8 Flash with `thinkingLevel=MEDIUM`, accepted
  by the user as ground truth (2026-09-08). Record model, reasoning setting, prompt,
  input hash, raw response and usage for every label; human listening is optional
  adjudication, never fabricated from teacher output.
- **Rubric.** Three orthogonal dimensions replaced the old catch-all
  `boundary_issue`: `speaker_purity` (pure | secondary_speaker | overlapping_speech),
  `word_completeness` (complete | clipped_word_start | clipped_word_end),
  `audio_quality` (studio_clean | music_bleed | noisy_reverberant | distorted).
  Pass requires all three clean. The teacher audits used a finer nine-defect
  rubric (`tts-v1`) whose codes map onto these.
- **Splits.** Split by recording before any augmentation; children, stems and
  synthetic derivatives inherit the parent's split. Unknown lineage is quarantined.
  Never tune on release data.
- **Frozen benchmark.** `.data/tts_strategy/gold_benchmark_20260908/` — 288 clips
  from 11 recordings, 159 pass / 129 reject under Gemini 3.8 Flash MEDIUM. It merges
  the earlier pools with the former Khanh Vy hold-out (`youtube:H0VpjeULCck`), so it
  is **not** source-disjoint from historical training data; reserve a fresh
  recording for any unseen-source claim. Do not relabel, augment or train on it.
- **Local production, offline teacher.** Gemini is the offline reference; a
  Gemini gate in production would mean cloud-assisted operation, not local-only.

## Timeline

| Date | Experiment | Result |
|---|---|---|
| — | 3-way diarizer benchmark on a 180 s vlog slice (Mel-RoFormer stem; DiariZen Large, Pyannote Community-1, 3D-Speaker; raw vs energy-valley "mitigated" cuts) | Snapping boundaries to ≤ −32 dBFS valleys cut word-incompleteness 37.5 → 25 % (DiariZen) and 75 → 62.5 % (Pyannote). Overall pass rate 0 % for every system because music bleed and hall reverb failed `audio_quality`. Report: `.data/benchmark_v2/BENCHMARK_REPORT.md` |
| — | Gemini 3.8 Flash MEDIUM vs shallow verifiers on 12 Khanh Vy cuts | 6 decision flips; shallow models accepted cuts whose final consonant closure was already chopped |
| — | Gemma 4 E2B LoRA V1 (120 cuts, bf16 on RX 9060 XT) | val loss 1.65 → 0.34; adapter `tungnguyenlam/gemma-4-e2b-acoustic-verifier` |
| — | E2B LoRA V2 (+ synthetic boundary hard negatives, 312 rows) / V3 (balanced crawl, 328 rows) | best val loss 0.276 / 0.329. Dataset repo `tungnguyenlam/gemma-4-e2b-acoustic-verifier-data` |
| — | Gemini 3.5 Flash-Lite (no thinking) on 31 Khanh Vy turns | 74.2 % pass, 51.6 % agreement with 3.8 Flash MEDIUM, 1.79 s/clip. Report: `.data/experiment_khanhvy/GEMINI_35_FLASH_LITE_REPORT.md` |
| 2026-09-08 | Legacy audit: 263 train + 65 val rows | No `recording_id`; 10 filename groups and 1 exact SHA duplicate cross the split; archived Khanh Vy labels were produced with LOW reasoning and omit `audio_quality` → relabelled with MEDIUM |
| 2026-09-08 | Fresh MEDIUM audit of the 31 Khanh Vy cuts | 21 reject / 10 pass; E2B V3 accepted all 31 (accepted error 67.7 %; 73.3 % on 2–15 s). 15 of the 31 were byte-identical to old training audio |
| 2026-09-08 | Word-lock boundary repair on the 31 located cuts (PhoWhisper-small) | Expanding into inter-turn gaps: 15 reject / 1 pass among 16 children (R = 93.8 %). Rejecting every ASR-overlapping edge dropped all 31. Final policy — never expand into gaps, reject only competitor/adjacent-turn word completions — emitted 8 crops with inherited R = 50 %. Word lock is a competitor-conflict gate, not a clip repairer |
| 2026-09-08 | Source-disjoint growth: 3 new recordings, Sortformer + measured lock | 1,996.66 s source → 36 clips (20 pass / 16 reject), 3.376 accepted min / source hour. Studio interview 16/19 pass (6.667 min/h); music-backed vlog 2/13 (0.892). Lock dropped 99/153 diarizer turns |
| 2026-09-08 | Config sweep on two 180 s slices, 6 pipeline configs, MEDIUM judge (110 clips) | Winner `recipe_nolock` (soft onset/collar + smart segmentation, no word lock): 39 clips, 31 pass, 30.8 accepted min / source hour. Every config with the word lock collapsed yield (≤ 10 clips) |
| 2026-09-08 | Unified gold benchmark | 288 clips / 11 recordings (159 / 129); see decision rules |
| 2026-09-08 | E2B LoRA V3 on the 288-clip gold | 288/288 pass — an all-pass classifier. Agreement 55.2 % (= teacher pass rate), F1 71.1 %, 129 leaks (music 61, secondary 37, clipped end 29), 3.52 s/clip |
| 2026-09-08 | Gemini 3.5 Flash-Lite MEDIUM on the gold | Agreement 63.2 %, F1 73.5 %, 47 rejects (35 true / 12 false), 94 leaks, 2.81 s/clip |
| 2026-09-09 | Flash-Lite prompt tuning on the gold (4 prompts) | **V2 defect-first: 71.5 % agreement, F1 78.6 %, 74 false pass / 8 false reject, $0.20 for 288 clips.** V1 checklist and V3 Vietnamese translation were worse than V0. Report: `.data/tts_strategy/gold_benchmark_20260908/prompt_tuning/REPORT.md` |

## What was learned

1. Deterministic boundary hygiene (energy-valley snapping, collars, no gap
   expansion) measurably reduces clipped words; ASR word overlap alone is not proof
   of clipping and PhoWhisper-small timestamps overlap nearly every edge.
2. Music bleed and in-interval secondary speech, not boundaries, dominate rejects
   on real Vietnamese vlogs; studio interviews are the productive source tier.
3. The Gemma 4 E2B LoRA student never learned to reject; low validation loss did
   not transfer. Diagnose adapter loading, audio input sensitivity, prompt masking
   and checkpoint selection on separate development audio before training again
   (the [scaffold command](agent_verifier.md#scaffold_experimentsh---name-name)
   creates the workspace for that).
4. A cheaper Gemini tier with a defect-first prompt is the best local-cost
   alternative measured so far, but still misses ~40 % of teacher rejects (quiet
   music, exact word-edge clipping).
5. Hardware constraints for local training are documented in [hardware.md](hardware.md).

## Published diarization DER (%) for the backends in this repo

Lower is better; `–` = no published number. Protocols differ across rows.

| Backend (checkpoint) | AISHELL-4 | AliMeeting | AMI SDM | DIHARD 3 | VoxConverse | CALLHOME p2 | CH109 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Pyannote Community-1 (`pyannote/speaker-diarization-community-1`, 0 s collar, overlap incl.) | 11.7 | 20.3 | 19.9 | 20.2 (full) | 11.2 | 26.7 | – |
| 3D-Speaker (FSMN VAD + CAM++ `speech_campplus_sv_zh_en_16k` + spectral clustering) | 10.30 | 19.73 | 21.76 | – | 11.75 | – | – |
| NeMo Sortformer (`nvidia/diar_sortformer_4spk-v1`) | – | – | – | 16.28 (≤ 4 spk eval subset) | – | 6.49 / 10.01 / 14.14 (2/3/4 spk) | 6.27 |
| NeMo clustering (MarbleNet VAD + TitaNet-large) | – | – | – | – | – | – | – |
| DiariZen (`BUT-FIT/diarizen-wavlm-large-s80-md-v2`, no collar, CC BY-NC 4.0) | 10.1 | 10.8 | 13.9 | 14.5 | 9.1 | – | – |

NVIDIA's published TitaNet clustering numbers use oracle VAD and are therefore
omitted. None of these checkpoints has a published Vietnamese number; the
100-file `tuanduy1612/ViYT-Diar` set is the natural next benchmark to run with
`scripts/evaluate/diarization.py`.

Sources: pyannote and NVIDIA model cards on Hugging Face, `modelscope/3D-Speaker`,
`BUTSpeechFIT/DiariZen`, `NVIDIA-NeMo/Speech` diarization README.

## Tracked artifacts under `.data/`

- `benchmark_v2/BENCHMARK_REPORT.md`, `audit_results.json`, `diar_turns.json`
- `experiment_khanhvy/*` — 31-cut probe results (LOW-reasoning labels; superseded)
- `distillation/{train,val}_e2b.jsonl`, `{train,val}_extended.jsonl`, `distillation_e2b/audit_manifest.json`
- `crawled/crawled_manifest.json` — 6 Tran Thanh / Khanh Vy tracks (4,817 s, 16 kHz mono)
- `tts_strategy/gold_benchmark_20260908/prompt_tuning/*` — prompts V0–V3 and per-clip results
