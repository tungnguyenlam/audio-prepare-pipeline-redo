# TTS cutting verification and integration decision

> **Superseded strategy (2026-09-17).** These results describe the `experimental-v2`
> planner, which required a pause strictly inside the aligned inter-word gap and
> optimized durations by dynamic programming. On this recording only 4 of 264
> Silero pauses fell inside an aligned gap, which explains finding 3 below. The
> current `experimental-v3` planner searches around sentence ends instead and
> merges fragments afterwards; see [tts_segmentation_testing.md](tts_segmentation_testing.md).
> The Gemini verdicts below have not been rerun on v3 output.

**Decision: native Silero JIT works on CPU and the local AMD GPU, but the current
cutting prototype should not replace the main pipeline.** Gemini found more
clipped boundaries in both prototype versions than in the existing cuts. The
experiment is complete, including retries; this is a negative quality result,
not an unfinished verification run.

## What was actually tested

- The downloaded Hana Lexis video `vn3KdmD0eCA`, in the `truyen-chem` playlist
  directory: 916.448875 seconds, mono, 48 kHz. Four playlist WAVs were available;
  this experiment used the entire named video, not the whole playlist.
- Source SHA-256:
  `0ee9b121e011bb4cc16248809b804a4391b0e14706beb8a2cf6ba1909d602057`.
- Native `silero_vad.jit`, SHA-256
  `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`.
  No ONNX inference or runtime was used. No packages were installed.
- AMD Radeon RX 9060 XT, `torch 2.13.0+rocm10.0.0`,
  `torchaudio 2.11.0.2+rocm10.0.0`, PyTorch-reported HIP `7.15.26333`.
  `cuda:0` is the PyTorch device name for this ROCm GPU.
- The existing `asr/phowhisper.sh`, cached `vinai/PhoWhisper-small`, ROCm,
  `--no-vad`, produced 31 ASR segments and 3,567 word entries from the full video.
  Its ASR VAD was disabled so the separate native JIT evidence remained explicit.
- Existing DiariZen cuts were the baseline. A fresh local DiariZen run recovered
  **85 unfiltered turns, two speaker IDs, and 14 turns over 15 seconds**.
- Gemini **3.8 Flash**, **MEDIUM** reasoning, standard inference, temperature 0,
  using the existing `prompts/acoustic_defect-3.txt` and production verifier.
  Every distinct candidate WAV was judged; byte-identical WAVs reused verdicts
  by full SHA-256. No judge was substituted.

**Not tested:** VibeVoice generation (complete local weights were unavailable),
NVIDIA CUDA execution (SSH to the configured server timed out), other PhoWhisper
sizes, a held-out recording, or human listening judgments. Standalone PhoWhisper
word alignment is not an end-to-end test of VibeVoice's supplied-text forced
alignment path. The cached PhoWhisper model did also load through the existing
VibeVoice alignment loader; that only establishes loading, not inference quality.

## Device results

Three repetitions per device, one CPU thread, one sequential recording stream,
32 ms frames, model state reset before each run. Timings include blockwise
probability transfer back to CPU but exclude preprocessing, loading, and initial
upload. This does not measure batched multi-recording GPU throughput.

| Audio | CPU median | ROCm median | Maximum probability difference | Frame disagreements at 0.35 |
| --- | ---: | ---: | ---: | ---: |
| Existing 89.629 s clip, 44.1 kHz input | 0.432 s | 0.848 s | 2.265e-6 | 0 |
| Existing 360 s interview, 16 kHz input | 1.775 s | 3.458 s | 1.311e-6 | 0 |
| Hana Lexis, 916.449 s, 48 kHz input | 4.603 s | 9.628 s | 1.371e-6 | 0 |

Repeated runs on the same device had identical probabilities in these checks.
CPU was roughly twice as fast for this single-stream implementation; both were
much faster than real time. Keep both device options. These results do not imply
that ROCm is unsupported, that all GPUs are slower, or that frame classifications
prove correct acoustic boundaries.

## Cutting and Gemini results

All outputs stayed within the inclusive 1.5–15 second limits. Counts below include
all produced clips, not a hand-picked subset. Each final clip has a successful,
schema-valid Gemini verdict after retries.

| Variant | Clips | Gemini pass / reject | Clipped boundaries | Clips in 7–10 s | Total / Gemini-pass seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing DiariZen cuts | 59 | 49 / 10 | 3 | 12 | 430.20 / 330.10 |
| Word collars, already-filtered speaker turns | 57 | 42 / 15 | 8 | 10 | 393.86 / 283.28 |
| Word collars, unfiltered speaker turns | 62 | 46 / 16 | 8 | 10 | 437.03 / 312.67 |
| VAD-supported outer edges, unfiltered turns | 62 | 48 / 14 | 9 | 11 | 433.22 / 316.19 |

The last variant had seven clipped-end and two clipped-start judgments. Gemini
also flagged music bleed in six clips; defects can co-occur. A higher total pass
count than the preceding trial did **not** establish better boundary safety.
The baseline remained better on both clipped-boundary count and passed audio
seconds. The variants differ in output intervals/populations, so these are
descriptive development results, not a paired statistical accuracy benchmark.

Total estimated Gemini cost: **$1.893008**, including failed attempts and retries.
There were 189 requests: 59 baseline, four baseline retries, 57 first-trial
clips, five additional distinct raw-turn clips, 62 corrected-trial clips, and
two corrected-trial retries. All six initial parse failures were retained and
retried with a larger output budget (4,096 → 8,192 or 8,192 → 16,384 tokens).
Successful verdicts were not retried to select a more favorable answer.

The judgment is about audio suitability. Gemini's independent pass transcript
is displayed next to ASR text, but its normalization rules differ; this experiment
does not establish ASR transcript accuracy or treat text differences as WER.

## Findings that change the implementation plan

1. **Use unfiltered speaker turns.** The legacy 59-clip manifest had already
   removed long turns. It left 1,758 ASR word entries unassigned. Fresh raw turns
   reduced that to 140; another 53 word entries had invalid/zero-length timing.
   Do not attempt to recover missing speech by treating gaps in a filtered
   manifest as silence.
2. **A word timestamp plus 40 ms is insufficient.** The first word-based trim
   increased clipping judgments. The correction protects the union of word
   support and nearby VAD speech, then bounds collars against speaker changes
   and omitted words. It still failed on quiet/uncertain edges. The current
   prototype records truncated collars instead of hiding them.
3. **The strict silence policy has poor recall on this monologue.** Only two
   internal boundaries met the current gap, collar, and sustained low-probability
   requirements. Only one became an internal cut. There were 22 rejected
   unsegmentable run/fragment intervals, including several continuous stretches
   of 16–44 seconds. A bounded edge correction cannot solve this internal-cut
   problem. Only 11/62 final clips reached the target band.
4. **VAD and alignment are evidence, not phoneme certification.** Silence frame
   classification agreement across CPU/GPU does not validate the final cut.
   Recurrent probabilities are coarse, while alignment can allocate pauses to
   adjacent words or place a boundary within an acoustic syllable.
5. **Music rejection is a separate quality gate.** Changing boundary scoring
   cannot make a music-backed passage suitable for clean TTS.

## What you can inspect now

All runtime artifacts are local and gitignored under `.data/evaluate/hana_tts/`.

| Artifact | What to inspect |
| --- | --- |
| `review.md` | Markdown review report for VS Code preview; click audio links to open WAV clips directly in VS Code; inspect word times, boundary evidence, ASR text, and Gemini verdicts |
| `summary.json` | Counts, passed seconds, costs, every clip hash and referenced verdict, all failed/successful attempts |
| `asr/phowhisper.json` | Actual production PhoWhisper output with all word entries |
| `vad/report.json` | Raw CPU/ROCm probabilities, hashes, timing repetitions, environment, numerical comparisons |
| `diarizen/<stem>/segments.raw.json` | Unfiltered speaker intervals used by the planner |
| `plan/`, `raw_plan/`, `protected_plan/` | Each trial's manifest, candidate scores, word lineage, and rejection ledger |
| `clips/`, `raw_clips/`, `protected_clips/` | Exported WAVs, sample-accurate manifests, and plots |
| `gemini_*/` | Exact raw response `.txt` plus validated verdict/error JSON; retries are separate |
| `checks/results.json` | Twelve successful logic/artifact checks |
| `checks/check_planner.py` | Rerunnable local check script; synthetic cases are clearly identified as logic checks |
| `make_review.py` | Rebuild the local Markdown/JSON review without model calls |

The report uses the local WAVs; it is not a hosted service. Do not move it alone
without preserving its relative paths. Audio links can be clicked directly in VS Code
Markdown Preview to open and play each clip in VS Code.

Checks covered sentence aggregation, long-run partitioning, isolated short
utterances, continuous speech with no admissible pause, a speaker interruption
with no aligned words, rejection of source-hash mismatches, exact WAV hashes and
sample counts, no overlapping output clips, word containment, and accounting for
all 3,567 input word entries exactly once as selected or rejected. Launchers and
CLI help were checked. These checks passed; acoustic quality did not pass the
promotion criterion.

## Recommended integration into ASR cutting

Keep the ASR worker and cut planner separate. The experimental commands now
demonstrate the file handoff, but leave the current pipeline's default unchanged.

1. **ASR/alignment writes evidence.** Preserve source identity, raw generation,
   unfiltered speaker turns, and source-relative word alignments. Add explicit
   alignment status/coverage, crop offsets, model identity, and token/character
   mappings. Long turns must be aligned in supported windows; a silently clipped
   30-second alignment input cannot support cutting the rest of a turn.
2. **The cut planner writes an inspectable plan.** Reuse a single native JIT
   probability track. Keep duration optimization independent of inference. Each
   boundary needs acoustic support, uncertainty, retained collar, selected text
   span, and competing candidate scores. Preserve all rejected spans. Input
   speaker IDs come from VibeVoice or an explicitly supplied unfiltered diarizer.
3. **Render only from that plan.** Reuse `_common/segments.py`; do not introduce
   another WAV cutting implementation. Extend its publication path to produce
   sibling JSONs and include them in completion/hash checks. The current renderer
   already keeps turn metadata and references the original plan by hash.
4. **Review before accepting training data.** Show waveform/VAD/word intervals
   around both ends, clip/source-context audio, text, and Gemini outcomes. Store
   manual adjustments as a new plan with lineage; do not edit original ASR.
   A renderer rerun should require neither ASR nor VAD inference.
5. **Treat continuous-speech fallback as a separate review class.** Add scored
   inter-word candidates where no silence is available, but mark absent collars
   and uncertain alignment. Verify them separately before admitting them to
   training. Do not relax the global threshold until a 7–10 second histogram
   looks good, or call such boundaries acoustically safe without evidence.

For the next quality iteration, preserve the original diarizer/ASR outer bounds
unless there is sustained, corroborated silence to trim. Examine the nine current
boundary failures in source context and distinguish missing transcript coverage,
incorrect alignment, and genuinely clipped source audio. Improve the evidence
used to authorize a trim before increasing aggregation complexity. Evaluate on
another downloaded playlist recording held out from this development loop.

Promotion should require no duration violations, no speaker-crossing merges,
complete text/word accounting, fewer boundary failures than the current method
without collapsing passed audio yield, and separate transcript-accuracy review.
Gemini can support that review; its labels are not human ground truth.

## Reproduce the independent stages

The actual existing CLI is intentionally smaller than the full design proposal.
Each command below runs one stage. Supply a **new output directory** for another
experiment, or explicitly use `--overwrite` where supported.

```bash
# Native model check; setup_worker_envs downloads the pinned JIT into ~/.cache/silero-vad.
bash scripts/evaluate/silero_jit.sh \
  --input-file .data/s1-download/truyen-chem/vn3KdmD0eCA_KHAU-NGHIE-48000.wav \
  --devices cpu cuda:0 --repeats 3 \
  --output-file .data/evaluate/hana_rerun/vad.json
```

```bash
# Reuse the saved ASR and unfiltered speakers; this invokes no model.
bash scripts/audio/segment_tts.sh \
  --input-manifest .data/evaluate/hana_tts/asr/phowhisper.json \
  --speaker-manifest .data/evaluate/hana_tts/diarizen/vn3KdmD0eCA_KHAU-NGHIE-48000/segments.raw.json \
  --vad-report .data/evaluate/hana_rerun/vad.json --vad-device cuda:0 \
  --output-file .data/evaluate/hana_rerun/plan/segments.json
```

```bash
bash scripts/audio/export_segments.sh \
  --input-manifest .data/evaluate/hana_rerun/plan/segments.json \
  --output-dir .data/evaluate/hana_rerun/clips \
  --min-duration-s 1.5 --max-duration-s 15
```

```bash
# Paid verifier; credentials come only from environment/local .env.
bash scripts/s4-agent/verifier/gemini.sh \
  --input-dir .data/evaluate/hana_rerun/clips \
  --output-dir .data/evaluate/hana_rerun/gemini \
  --model gemini-3.8-flash --reasoning-effort medium \
  --inference-mode standard --max-tokens 16384 --concurrency 4 --batch-size 4
```

The planner does not yet provide corpus batch mode, source-text character-span
fidelity, a voiced-word fallback, or an automatic acceptance gate. Keep it
experimental until those contracts and the quality regressions are resolved.
