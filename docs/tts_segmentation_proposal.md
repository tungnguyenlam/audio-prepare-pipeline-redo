# Acoustic-aware TTS segmentation proposal

Status: design only; the proposed command and flags below are not implemented.

Recommend an independent `scripts/audio/segment_tts.py` command after VibeVoice
and PhoWhisper alignment, with native Silero JIT inference on CPU. Optimize a
whole contiguous speaker run before exporting any of its chunks. Sentence
punctuation proposes boundaries; it does not trigger an immediate cut.

## Placement and reuse

| Placement | Advantages | Costs |
| --- | --- | --- |
| Inside ASR/alignment worker | Reuses decoded audio and word objects; avoids intermediate reads | Couples segmentation policy to expensive inference; keeps ASR VRAM occupied longer; complicates independent retries and tuning |
| Standalone downstream command (recommended) | Reuses immutable sidecars and VAD cache; CPU workers leave VRAM to ASR; independently resumable and tunable | Reads audio again; requires explicit source identity and timestamp validation |

Keep ASR, alignment, segmentation, and later dataset export independently
invocable. The segment command may plan and render its own clips; it must not
launch upstream models or a pipeline orchestrator. Keep the inference function
separate from thresholding, candidate selection, and export. There is no current
second consumer requiring a new shared VAD abstraction.

Repository integration points inspected:

- `scripts/asr/vibevoice.py` already emits turns with `start_time`, `end_time`,
  `speaker_id`, `text`, and nested `words`; its forced-alignment offsets are
  already source-relative. Do not add the turn offset again.
- `scripts/asr/phowhisper.py` emits the same word shape but speaker `"0"`.
  Standalone ASR words must be explicitly associated with VibeVoice turns;
  do not use PhoWhisper's placeholder as diarization identity. If the transcripts
  differ, require a recorded text mapping or realignment, not a time-only join.
- `scripts/purity/segment.py` splits individual turns at word gaps using RMS;
  it neither aggregates turns nor runs Silero. It is not this proposed policy.
  Its valley helper is also used by `purity/snap.py`; preserve that consumer.
- `_common/files.py` supplies identity, digest, atomic JSON, progress, and batch
  utilities. `_common/segments.py` supplies source verification, normalization,
  integer-sample duration filtering, naming, and WAV rendering. Reuse those
  paths instead of adding another audio writer or hash implementation.
- The existing exporter preserves per-turn custom fields but does not write a
  JSON sidecar per WAV. `audio/export_segments.py` also rebuilds top-level
  metadata. Implementation must preserve segmentation lineage explicitly and
  add sibling sidecar publication/integrity checks. Extend tracked cleanup to
  this operation and its sidecars; current cleanup recognizes only `diarize`
  and `export_segments`. Preserve existing consumers' behavior.
- The audio environment intentionally has no model dependencies. Use a
  same-name `segment_tts.sh` selecting `SEGMENT_TTS_PYTHON`, then the existing
  alignment environment (as `purity/align.sh` does), forwarding `"$@"` unchanged.
  Default Silero to CPU; do not silently install packages or use ONNX.

## Input normalization and speaker isolation

Validate finite, monotonic, source-relative timestamps; source hashes; text/word
coverage; and bounds against actual audio frames. Preserve raw values and input
indices before normalization. Normalize `word`/`text` once at this command's
boundary. Use VibeVoice speaker IDs scoped to the source recording; equal IDs
from different recordings are not necessarily the same person.

Split each turn's text into sentence spans using punctuation attached to aligned
tokens. Vietnamese whitespace units are not necessarily lexical words: retain
original character spans and punctuation, rather than rebuilding all text with
blind whitespace joining. Missing punctuation still permits acoustic word gaps.
Sentence boundaries must map to aligned tokens; missing or inconsistent mappings
are audited, not filled with invented timestamps.

Build runs in chronological order. A different speaker, unknown identity,
unresolved overlap, or a long/unexplained gap ends a run. Never gather all turns
with the same ID globally: A–B–A is three runs. Reject overlapping speech regions
by default; VAD cannot establish which speaker is active. Same-speaker overlaps
need deduplication or review before aggregation.

Propose `--max-join-gap 1.0`: longer gaps end a run, and shorter gaps can be
bridged only when acoustic evidence supports silence and no other speaker
intersects the union. Do not silently omit untranscribed speech in a gap. Keep
internal pauses when aggregating; dead-air trimming applies to exposed clip
edges. Removing internal silence would require an explicit splice contract.

Long-turn alignment coverage needs special care: the current VibeVoice worker
uses Whisper `pad_or_trim` and a 3,000-frame limit, and skips alignment above its
token limit. A long turn can therefore have empty or incomplete words. Require
complete coverage before using its internal word boundaries; route failures to
an independently invoked alignment repair step, not implicit ASR inside cutting.

## VAD evidence and protected boundaries

Run native JIT Silero once on a continuously resampled 16 kHz mono analysis
stream. Use `torch.jit.load(path, map_location='cpu').eval()` with a pinned model
hash, or a pinned Torch Hub revision with `onnx=False`. Set Torch Hub caches under
`.data/` if that loading path is used. There is no ONNX fallback.

The upstream implementation resets recurrent state once per recording and
processes 512-sample frames at 16 kHz, padding the final frame. Follow that
sequence, saving raw probabilities before thresholding. A frame is 32 ms; it is
not a phoneme boundary measurement. See [Silero inference source](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py)
and [JIT loading path](https://github.com/snakers4/silero-vad/blob/master/hubconf.py).

Query this cached track for each inter-sentence interval `[end_k, start_k+1]`.
Do not reset and run the model on each tiny gap: context and recurrent state
would differ, and many gaps are shorter than a frame. Empty or overlapping gaps
offer no silence evidence. Frames straddling speech are uncertain; inspect local
RMS and alignment too, and do not treat final zero padding as real silence.

Starting parameters (to calibrate on the corpus, not phoneme guarantees):

- Speech threshold 0.35, exit threshold 0.20, with hysteresis.
- At least 64 ms sustained low probability for an ordinary silence candidate.
- Safety collar 40 ms on both sides; maximum retained external silence 150 ms
  per edge. The 150 ms limit is not a cap on internal pauses.
- Local RMS measured over 5–10 ms windows can refine a VAD valley. An RMS minimum
  or zero crossing alone never certifies absence of speech.

For a candidate silence interval, estimate conservative left speech offset `L`
and right speech onset `R` from both alignment and acoustic evidence. Protect the
union of aligned word support and plausible adjacent speech, rather than
shortening a word because VAD misses a soft ending. Search for a cut `c` only in
the intersection of verified silence and `[L + collar, R - collar]`.
Choose the minimum window-mean speech probability, then normalized RMS; use
distance from the silence center and source-sample order for deterministic ties.
One isolated low-probability frame is insufficient. No admissible interval means
there is no protected silence cut at that gap.

For an accepted separating boundary, use two export edges:

```text
left_end    = min(c, L + 0.150)
right_start = max(c, R - 0.150)
```

Because `c >= L + collar` and `c <= R - collar`, both collars remain intact.
The discarded interval between the two edges contains only verified silence.
For example, `L=8.00`, `R=8.60`, `c=8.30` gives edges 8.15 and 8.45 seconds.
Both retain 150 ms, instead of giving each clip 300 ms of dead air.

At the beginning/end of a run, trim to protected onset minus collar / offset
plus collar, removing only verified external silence. Stay within recording and
speaker-safe bounds. Never extend a collar into a competing speaker; record a
truncated collar or reject an unsafe boundary. When evidence is uncertain,
preserve speech or reject rather than enforce a silence cap by clipping it.

Whisper `probability` averages text-token probabilities; it is not calibrated
timestamp confidence. Preserve it as evidence but assess coverage, ordering,
word duration, and acoustic agreement separately. See [Whisper alignment source](https://github.com/openai/whisper/blob/main/whisper/timing.py).

## Aggregation by a shortest-path dynamic program

Use sentence boundaries first, with word-boundary candidates inside long
sentences as fallbacks. Each boundary node stores the ending edge of the left
clip, starting edge of the right clip, method, and acoustic cost. Retain multiple
nondominated boundary variants where needed; one locally deepest valley may
otherwise make the next clip infeasible.

For nodes `i < j`, an edge means one chunk containing every sentence/token
between them. Its duration `d(i,j)` uses the actual trimmed and collared export
edges, including internal silence. It is admissible only if all speech belongs
to the same run, text is fully covered, and:

```text
ceil(1.5 * source_rate) <= end_sample - start_sample
                       <= floor(15.0 * source_rate)
```

Use decimal duration conversion as the current exporter does. Validate again
after any output resampling. Do not silently truncate the WAV to repair an
overrun; select an earlier valid boundary. Preserving source rate by default
keeps sample constraints simple.

One explicit starting cost is:

```text
D(d) = max(0, 7-d)^2 + max(0, d-10)^2 + 0.02*(d-8.5)^2
B(j) = 2*mean_speech_probability(j)
     + 0.5*normalized_RMS(j)
     + 0.5*(1-min(silence_duration(j)/0.20, 1))
     + 2*is_non_sentence_boundary(j)
C(i,j) = 1 + D(d(i,j)) + B(j)
DP[j] = min_i(DP[i] + C(i,j))
```

Set `DP[start]=0`, set infeasible edges to infinity, and backtrack from run end.
No cut penalty is needed at a mandatory run endpoint. Costs are dimensionless
heuristics with duration expressed in seconds; version and record all weights.
For a concrete RMS normalization, use `clip((rms_dbfs + 80)/60, 0, 1)`
(digital silence maps to zero); these endpoints also require corpus calibration.
Prefer safe complete coverage first, then minimize cost. Unsafe boundaries
cannot buy admission through a favorable duration score. The 15-second limit is
a feasibility condition, never a finite penalty.

This naturally accumulates short sentences into 7–10 second chunks; a sentence
ending at 3 seconds is not finalized merely because punctuation exists. It also
considers the suffix: if a greedy 9-second cut leaves a 1-second tail, select an
earlier boundary or a legal 10-second combined chunk. Do not commit a clip until
the run's path is solved. Candidate predecessors outside the 15-second span can
be pruned; work scales with candidates in the local duration window rather than
all pairs in the recording.

### Fallbacks and infeasible cases

1. Seek a protected sentence pause near 7–10 seconds. A safe cut between 10 and
   15 seconds is preferable to an unsafe cut at the target duration.
2. When a sentence/run exceeds 15 seconds, add aligned internal word-gap nodes,
   including breaths or micro-pauses that meet the acoustic/collar requirements.
   Use the same optimization, so the final remainder is considered.
3. If there is no silence-supported path, an optional `--unsafe-cut-policy review`
   can propose the lowest-risk aligned inter-word boundary before the ceiling,
   using VAD/RMS and endpoint uncertainty. It must not cut through a recognized
   word or pretend the collar fits. These chunks are `needs_review` and excluded
   from the accepted training manifest. With default `reject`, retain only the
   rejection audit. Never export a chunk longer than 15 seconds.
4. If even an inter-word boundary is unavailable (including pathological words
   spanning over 15 seconds), reject the affected interval for alignment repair.
   A zero crossing inside continuous speech does not solve phoneme preservation.

An arbitrarily long continuous utterance cannot be guaranteed both lossless and
cut into <=15-second clips. The honest policy is accepted safe clips plus an
audit of unsegmentable regions, not an unqualified promise of safe forced cuts.

At a speaker change, reoptimize the preceding same-speaker run to absorb a short
tail. If impossible, reject that residual interval. A complete isolated 0.9-second
utterance has no legal neighbor: never merge it across speakers or pad it with
silence to manufacture the minimum. When a full path is impossible, use explicit
rejection edges at token boundaries, minimizing rejected speech duration first
and then segmentation cost; do not discard all recoverable speech in a long run.
Every input speech token must appear once in accepted/review output or in an
explicit rejection record. Silence-only omissions have separate trim audits.

## Proposed file contract

Use `.data/audio/segment_tts/<family>/<safe-stem>-<source-hash12>/<request-hash12>/`.
The request digest covers source/ASR/alignment identities, JIT hash, algorithm
version, all segmentation parameters, and rendering settings. Compare full
digests in metadata; short path digests are labels, not identity checks.

Reuse the exporter's naming convention:

```text
segments.json
<stem>_<model>_<speaker>_<start_ms:09d>-<end_ms:09d>_<index:04d>.wav
<stem>_<model>_<speaker>_<start_ms:09d>-<end_ms:09d>_<index:04d>.json
```

Here `model` can remain the inherited ASR model; `operation=segment_tts` and
explicit VAD metadata identify the cutting implementation. Milliseconds are
display-only; integer source samples determine identity and cuts. Speaker names
are sanitized for filenames while their original values remain in JSON.

Illustrative accepted manifest entry (hashes and array contents abbreviated):

```json
{
  "schema_version": 1,
  "operation": "segment_tts",
  "model": "vibevoice",
  "source": {"path": ".data/audio/source.wav", "sha256": "..."},
  "parameters": {
    "input_manifest": {"path": ".data/asr/source.json", "sha256": "..."},
    "target_min": 7.0, "target_max": 10.0,
    "hard_min": 1.5, "hard_max": 15.0,
    "vad_threshold": 0.35, "collar_ms": 40,
    "algorithm_version": "tts-dp-v1"
  },
  "vad": {"backend": "silero_jit", "model_sha256": "...", "cache_key": "..."},
  "timestamp_origin": "diarized_input",
  "source_sample_rate": 48000,
  "sample_rate": 48000,
  "channels": 1,
  "speaker_ids": ["spk00"],
  "turns": [{
    "segment_id": "sha256-of-request-speaker-sample-interval",
    "speaker_id": "spk00",
    "confidence": null,
    "start_sample": 576000, "end_sample": 984000,
    "start_s": 12.0, "end_s": 20.5,
    "duration_s": 8.5,
    "text": "Xin chào. ...",
    "_transcript": "Xin chào. ...",
    "_words": [{
      "word": "Xin", "text": "Xin", "start": 12.04, "end": 12.20,
      "probability": 0.94,
      "source_turn_index": 3, "source_word_index": 0,
      "clip_start_s": 0.04, "clip_end_s": 0.20
    }],
    "lineage": {
      "input_turn_indices": [3, 4],
      "sentences": [{"turn_index": 3, "char_start": 0, "char_end": 9}],
      "alignment_sha256": "..."
    },
    "boundary": {
      "start": {"method": "run_start", "collar_ms": 40},
      "end": {"method": "sentence_silence", "collar_ms": 40,
              "gap_start_s": 20.35, "gap_end_s": 20.80,
              "cut_s": 20.57, "mean_speech_probability": 0.06}
    },
    "quality": {"status": "accepted", "flags": []},
    "overlap": false, "overlap_with": [],
    "clip": "source_vibevoice_spk00_000012000-000020500_0001.wav",
    "clip_sha256": "...", "clip_frames": 408000,
    "sidecar": "source_vibevoice_spk00_000012000-000020500_0001.json",
    "sidecar_sha256": "..."
  }],
  "rejected": [],
  "complete": true
}
```

The full parameters object must include omitted defaults/weights and output
settings. External alignment files get their own path/hash identity; embedded
words reference the ASR sidecar hash. `_words` keeps source-relative times for
existing purity consumers; `clip_*` times are explicitly relative to the slice.
`text` preserves selected original text spans; `_transcript` is the existing
consumer field, written identically. Preserve raw word objects in the immutable
parent sidecar. Word/text aliases are an explicit local contract conversion.

Each sibling JSON uses the existing audio-sidecar envelope: `schema_version`,
`source`, `operation`, `model`, `parameters`, and `output` containing sample rate,
channels, frames, duration, format, and WAV SHA-256. A `segment` field contains
the manifest turn metadata except the sidecar's own path/hash. This avoids a
self-hash cycle. Rejections record source interval, speaker, turn/word references,
reason, and attempted boundary evidence. Review-only clips live in a separate
review manifest/directory, never in accepted `turns`.

Publish an incomplete manifest and sidecars (`output: {}`) first. Stage each WAV
on its destination filesystem and rename atomically, finalize its JSON, then
write `complete: true` last. Resume checks request identity plus WAV and sidecar
digests, not only existence. An optional corpus JSONL is a derived index of
completed manifests: one row per segment with audio/sidecar paths, speaker,
text, duration, lineage, and hashes. Atomically replace it; avoid concurrent
append and duplicate segment IDs.

## Proposed CLI and operation

Design example only; `segment_tts.sh` does not exist yet:

```bash
bash scripts/audio/segment_tts.sh \
  --input-manifest .data/asr/source.json \
  --input-file .data/audio/source.wav \
  --output-file .data/audio/segment_tts/example/segments.json \
  --target-min 7.0 --target-max 10.0 \
  --hard-min 1.5 --hard-max 15.0 \
  --vad-model-path .data/models/silero/silero_vad.jit \
  --vad-threshold 0.35 --vad-exit-threshold 0.20 \
  --min-silence-ms 64 --collar-ms 40 --max-edge-silence-ms 150 \
  --max-join-gap 1.0 --unsafe-cut-policy reject \
  --vad-cache-dir .data/cache/silero-jit \
  --device cpu --concurrency 2 --batch-size 8
```

`--input-manifest` selects one ASR/aligned sidecar; `--manifest-dir` is its batch
alternative. `--input-file` is an exact audio override; `--input-dir` is the
audio root for batch lookup, with file precedence. Associate batch inputs by
recorded source identity/relative path, never by ambiguous basename alone.
`--words-file` can override embedded words with a validated alignment artifact.
Omitting audio overrides resolves and verifies the recorded source; explicit
overrides follow the repository's same-timeline responsibility and record the
new hash. Keep original ASR source identity in lineage.

`--output-file` is the exact manifest destination and is single-input only;
clips go beside it. Batch mode uses `--output-dir` and the per-source layout
above. Reject ambiguous multiple-input/output-file combinations. Default all
runtime paths beneath `.data/`. Support `--overwrite`; do not silently replace
conflicting requests. Validate `1.5 <= hard_min <= target_min <= target_max <=
hard_max <= 15.0` for this training profile, positive collars/silence settings,
`collar_ms <= max_edge_silence_ms <= 150`, and
`0 <= exit_threshold < threshold <= 1`.

Emit progress and quality counts to stderr; print successful manifest paths to
stdout. Include duration distribution, percent within target, accepted/rejected
speech duration, sentence versus internal-word cuts, truncated collars,
alignment failures, overlap rejections, and cache hits. Never log secrets.

## Efficient caching and concurrency

Use a content-addressed raw-probability cache keyed by source SHA-256, channel
selection/downmix policy, resampler implementation/settings, JIT SHA-256,
inference version/device/dtype, 16 kHz rate, 512-sample hop, and state policy.
Store probabilities in a compact array with atomic metadata (hash, frame count,
true unpadded sample count, time origin). Thresholds, collars, and target durations
belong to the derived segmentation key, not this raw inference key.

One hour at this hop is 112,500 float32 probabilities, about 0.45 MB before
metadata. Preserve continuous resampler and model state across I/O blocks.
Prefix sums and indexed low-probability runs make repeated gap queries cheap.
Never cache only thresholded speech spans: they lose the depth needed for cuts.

Parallelize across recordings with one model instance/state per process and
bounded Torch thread counts. `--concurrency` limits recording workers and
`--batch-size` bounds queued recordings; neither batches neighboring temporal
frames as independent examples. Multiple recordings can be tensor-batched only
with separate stable recurrent-state lanes and careful reset/padding semantics;
defer that complexity until profiling warrants it. Serialize duplicate cache
builders with a per-key lock, recheck after acquiring it, and publish complete
metadata last. Reuse decoded audio within one worker for energy refinement;
export the original-quality source, not the VAD's 16 kHz analysis copy.

## Completion scope

This change records the algorithm and integration contract only. No command,
launcher, dependency, or production behavior changes. Existing launchers and
helpers were inspected statically; the proposed CLI cannot be invoked yet.
No tests, model inference, audio experiments, or package installations were run.
Tests were not requested. Before implementation release, corpus listening and
boundary-quality evaluation are still required to calibrate the heuristic
weights, collars, and VAD thresholds for soft Vietnamese endings.
