# Testing the sentence-end TTS cutter

This is the hands-on recipe for the `experimental-v3` strategy in
`scripts/audio/segment_tts.py`: cut at every sentence end that Silero confirms as
a pause, split anything over 15 s at a word pause, then merge fragments to a
7–10 s mean. Every step is an independent command; nothing runs upstream models
implicitly. All artifacts stay under `.data/`.

## What you need

| Input | Produced by | Notes |
| --- | --- | --- |
| Source WAV | `s1-download` or any local file | Any rate; the same bytes must be used for every step |
| ASR JSON with `turns[].words[]` | `asr/vibevoice.sh` (preferred) or `asr/phowhisper.sh` | VibeVoice punctuates every sentence; PhoWhisper emitted one sentence end per ~44 s on the Hana test, so its plans rely on the word-pause fallback |
| Silero JIT report | `evaluate/silero_jit.sh` | Needs `silero_vad.jit` on disk, e.g. `.data/models/silero/silero_vad.jit`; CPU is fine and faster than ROCm |
| Optional unfiltered diarization | any `s3-diarize` backend's `segments.raw.json` | Only when ASR speaker labels are not trusted; never pass an exported, filtered manifest |

## Step by step

VibeVoice full precision needs ~17 GB VRAM. On the AMD host use the NVIDIA
server for VibeVoice; INT8/NF4 checkpoints are bitsandbytes and CUDA-only.

```bash
# 0. Pick a recording
src=.data/s1-download/truyen-chem/vn3KdmD0eCA_KHAU-NGHIE-48000.wav
out=.data/tts/vn3KdmD0eCA

# 1. Transcribe with word alignment (VibeVoice on the NVIDIA server, or PhoWhisper locally)
bash scripts/asr/vibevoice.sh --input-file "$src" --output-dir "$out/asr"
#   or: bash scripts/asr/phowhisper.sh --input-file "$src" --output-dir "$out/asr" --no-vad

# 2. Cache Silero probabilities once per recording (reusable for every cut setting)
bash scripts/evaluate/silero_jit.sh --input-file "$src" \
  --model-file .data/models/silero/silero_vad.jit \
  --devices cpu --output-file "$out/vad.json"

# 3. Plan cuts (no model inference; fast)
bash scripts/audio/segment_tts.sh \
  --input-manifest "$out/asr/$(basename "${src%.wav}")_vibevoice.json" \
  --vad-report "$out/vad.json" \
  --output-file "$out/plan/segments.json" --overwrite

# 4. Render clips for listening
bash scripts/audio/export_segments.sh --input-manifest "$out/plan/segments.json" \
  --output-dir "$out/clips" --min-duration-s 1.5 --max-duration-s 15
```

The planner prints one summary line to stderr, for example:

```
TTS_PLAN : 83 speaker runs; 5/21 sentence ends passed the VAD gate; 104 fragments -> 90 candidates (mean 8.09s, 20 in target band); 205 rejection records
```

Read it as: how many punctuation marks the ASR produced, how many of those had a
real pause, how fragmented the raw cut was, and what the merge produced.

## Knobs worth trying

| Flag | Default | Effect |
| --- | --- | --- |
| `--silence-threshold` | 0.1 | Raise toward 0.2 if too few sentence ends pass; lower if clips end on breath noise |
| `--min-silence-ms` | 64 | Two Silero frames; 96–128 is stricter |
| `--cut-search-ms` | 400 | How far into the neighbouring words the pause search may reach; 0 recreates the old "pause inside aligned gap" behaviour |
| `--merge false` | merge on | Emit raw fragments to inspect where the cuts land before any merging |
| `--target-min/--target-max` | 7 / 10 | Merge band; `--hard-max` 15 is never exceeded |
| `--speaker-manifest` | none | Use a diarizer's `segments.raw.json` instead of ASR speaker IDs |

Every run writes `parameters` into the plan, so two plans can be diffed.

## Checking the result

1. **Counts.** In the plan JSON, `rejected[].reason` tells where speech was lost:
   `no_pause_within_hard_max` means a continuous stretch with no usable pause,
   `too_short` a leftover under 1.5 s, and `unassigned_or_overlapping_speaker`
   speech the speaker labels could not own.
2. **Where cuts landed.** `audit[].boundaries[]` lists every accepted candidate
   with `pause_start_s`/`pause_end_s`, `max_speech_probability`, and
   `inside_aligned_word` (true when the pause was inside a word's aligned extent,
   which is the normal case with forced alignment). `audit[].fragments[]` shows
   the pre-merge pieces.
3. **Listening.** Open the exported clips; the manifest keeps `text` and
   `_words` with clip-relative times for each candidate.
4. **Model judgement (paid).** The Gemini acoustic-defect verifier from the
   earlier experiment can be run on the new clips; see
   [tts_segmentation_verification.md](tts_segmentation_verification.md) for the
   prompt and procedure. It has not yet been rerun on v3 output.

## Hana baseline for comparison

Cached inputs from the earlier experiment live in `.data/evaluate/hana_tts/`
(PhoWhisper ASR, CPU/ROCm Silero report, unfiltered DiariZen turns). Rerun the
planner on them with:

```bash
bash scripts/audio/segment_tts.sh \
  --input-manifest .data/evaluate/hana_tts/asr/phowhisper.json \
  --vad-report .data/evaluate/hana_tts/vad/report.json \
  --speaker-manifest .data/evaluate/hana_tts/diarizen/*/segments.raw.json \
  --output-file .data/evaluate/hana_tts/v3_plan/segments.json --overwrite
```

| Planner | Candidates | Mean | In 7–10 s | Retained seconds | Unsplittable spans |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 (pause inside aligned gap, DP) | 62 | – | 11 | 433 | 22 |
| v3 (sentence end + windowed pause, merge) | 90 | 8.09 s | 20 | 728 | 2 |

The v3 numbers are structural only; no listening or Gemini pass has judged
their boundary quality yet. PhoWhisper supplied only 21 sentence ends, so
almost all v3 cuts here came from the word-pause fallback. Rerunning with a
VibeVoice transcript is the real test of the sentence-end rule.
