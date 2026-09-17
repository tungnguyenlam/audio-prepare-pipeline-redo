# Recursive VAD-only segmentation baseline

## Scope

This records one real-video run of `audio/segment_vad.sh`. The command is a
literal baseline: every interval over 15 seconds is cut at its globally lowest
Silero speech-probability frame, recursively. It does not use ASR, diarization,
sentence boundaries, or a minimum child duration.

The example is the 89.629-second mono excerpt from YouTube video `a9aVQCWi9MY`,
“Movie cut | Trích đoạn KHÁCH MỜI HẢI LAN: MỘT LẦN NÓI HẾT - THỎ ƠI!! - Phim
Tết TRẤN THÀNH 2026”. Runtime artifacts remain under `.data/` and are not
committed.

## Reproduction

The run reused the existing Silero report, so it made no new VAD inference call:

```bash
bash scripts/audio/segment_vad.sh \
  --input-file .data/quick_save/a9aVQCWi9MY__89.63s_44100Hz_1ch.wav \
  --vad-report .data/evaluate/silero_jit/quick_save/report.json \
  --vad-device cpu \
  --output-file .data/audio/segment_vad/a9aVQCWi9MY/segments.json
```

The planner produced 97 intervals from 96 cuts. All intervals were at most 15
seconds (maximum 14.976 seconds), but 86 were shorter than one second. The long
quiet tail caused the literal recursion to peel off adjacent 32 ms VAD frames.
All selected cut probabilities were below 0.1, ranging from 0.000084 to 0.021893.

Only the 11 intervals at least one second long were rendered for model review.
This avoids spending verifier calls on 86 tiny silence shards while retaining the
substantive speech-bearing sides of the boundaries:

```bash
bash scripts/audio/export_segments.sh \
  --input-manifest .data/audio/segment_vad/a9aVQCWi9MY/segments.json \
  --output-dir .data/audio/segment_vad/a9aVQCWi9MY/clips \
  --min-duration-s 1 --max-duration-s 15
```

## Gemini 3.8 Flash review

Existing Gemini 3.8 Flash artifacts were searched by exact clip SHA-256 before
submission. None matched the new clips. The 11 clips were therefore submitted in
one Gemini Batch job using the existing production verifier and the default
`acoustic_defect-3.txt` prompt:

```bash
bash scripts/s4-agent/verifier/gemini.sh \
  --input-dir .data/audio/segment_vad/a9aVQCWi9MY/clips \
  --output-dir .data/s4-agent/verifier/gemini_vad_baseline_a9aVQCWi9MY \
  --model gemini-3.8-flash --reasoning-effort medium \
  --inference-mode batch --batch-size 11
```

All 11 generations completed. Gemini classified word completeness as `complete`
for all 11 clips: zero `clipped_word_start` and zero `clipped_word_end`. Four
clips passed overall; seven were rejected for unrelated music, singing, or
secondary-speaker defects. Estimated Batch cost was $0.02435775.

The existing `gemini_vad_protected_relaxed` result set uses the same Gemini model
and prompt on a different video and a different, VAD-protected segmentation
strategy. It contains 54/62 complete judgments, two clipped starts, and six
clipped ends. Those results cannot be attributed to this exact recursive
algorithm, but they are reusable evidence that a low VAD score does not certify
phoneme completeness.

## Conclusion

This example did not exhibit word clipping in its 11 substantive intervals, so
it is a successful negative observation rather than proof of safety. The method
can still cut a word when the lowest probability in an oversized interval occurs
inside quiet speech: it has no silence threshold, word alignment, or phoneme
check. More immediately, literal global-minimum recursion is unsuitable as a
standalone dataset segmenter because clustered minima create many tiny fragments.
A production variant should treat a silence valley as one candidate and enforce
a minimum child duration before comparing candidates; that would be a different
algorithm and is not implemented by this baseline.
