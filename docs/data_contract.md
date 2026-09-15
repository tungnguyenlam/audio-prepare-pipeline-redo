# Data and file contract

[← Overview](../README.md) · [CLI cookbook](commands.md)

Commands exchange data only through WAV files, sibling JSON sidecars, and JSON
manifests. Every JSON artifact is written atomically (temp file + rename).

## Crawl manifest (`download/crawl`)

The default destination is `.data/download/<collection>/crawl.json`. Downloaded
crawl audio is grouped under `.data/download/<resolved-source-name>/` (or under
`<output-dir>/<resolved-source-name>/` when `--output-dir` is supplied). `videos`
contains accepted unique YouTube videos in source-priority order; duplicate
occurrences are represented by multiple objects in a video's `sources` array.
`rejected` is an occurrence-level audit because the same video can be rejected by
one source-specific filter and accepted through another source. Each accepted
video has `download.status` equal to `pending`, `not_requested`, `complete`, or
`failed`; completed items record the WAV path and failures record the error
text. `sources` records per-source listing status and the resolved
playlist/channel name, while `summary` exposes source, filter, truncation, and
download counts.

```json
{
  "schema_version": 1,
  "operation": "crawl",
  "collection": "vi-en-natural-codeswitch",
  "filters": {"exclude_title_regex": ["podcast"], "exclude_live": true},
  "summary": {"accepted_unique": 42, "duplicate_occurrences": 7, "rejected_occurrences": 3, "metadata_only": true},
  "sources": [{"name": "Example", "url": "https://www.youtube.com/@example/videos", "resolved_name": "Example", "status": "complete"}],
  "videos": [{"video_id": "abcdefghijk", "title": "Example", "url": "https://www.youtube.com/watch?v=abcdefghijk", "sources": [], "download": {"status": "not_requested"}}],
  "rejected": [{"video_id": "lmnopqrstuv", "source": "Example", "reason": "title_matched_exclude_regex"}]
}
```

## 1. Audio sidecar (`recording.wav` → `recording.json`)

Written by `download/*`, `audio/{convert,cut}`, `separate/*` (and, with a
non-audio `output`, by `compare_*`, `evaluate/*`, `mix`). An incomplete record
(`"output": {}`) is installed before the audio and replaced afterwards, so an
interrupted write is recognizable and retried.

```json
{
  "schema_version": 1,
  "source": {"path": "/abs/source.wav", "sha256": "…", "origin": {"video_id": "…", "title": "…", "url": "…"}},
  "operation": "separate",
  "model": "htdemucs_ft",
  "parameters": {"stem": "vocals", "device": "cpu", "sample_rate": 48000, "channels": 1},
  "output": {"sample_rate": 48000, "channels": 1, "frames": 240000, "duration_s": 5.0, "format": "WAV", "sha256": "…"}
}
```

- Download sources carry `video_id`, `title`, `url` directly in `source`.
- Local sources carry absolute `path` + `sha256`; when a verified prior sidecar
  exists its `source` is copied into `source.origin` (this is how the audio family
  propagates).
- Skip / retry / conflict decisions compare `source`, `operation`, `model`,
  `parameters` and the output hash.

## 2. Diarization manifest (`<stem>/segments.json`)

```text
<output-dir>/<safe stem>/
  segments.json
  segments.raw.json
  plot/
    timeline.png
    timeline_duration.png
    timeline_cutoff.png
  <stem>_<model>_<speaker>_<start_ms:09d>-<end_ms:09d>_<index:04d>.wav
```

```json
{
  "schema_version": 1,
  "source": {"path": "/abs/input.wav", "sha256": "…"},
  "operation": "diarize",
  "model": "sortformer",
  "parameters": {"device": "cuda:0", "min_duration_s": 1.0, "max_duration_s": 15.0, "…": "…"},
  "timestamp_origin": "diarized_input",
  "source_sample_rate": 48000,
  "sample_rate": 48000,
  "channels": 1,
  "speaker_ids": ["spk00", "spk01"],
  "turns": [
    {"speaker_id": "spk00", "start_s": 12.34, "end_s": 18.92,
     "start_sample": 592320, "end_sample": 908160,
     "overlap": false, "overlap_with": [],
     "clip": "input_sortformer_spk00_000012340-000018920_0001.wav",
     "clip_sha256": "…", "clip_frames": 315840}
  ],
  "complete": true
}
```

- `model` is one of `sortformer`, `pyannote_community1`, `pyannote_31`,
  `clustering`, `threed_speaker`, `diarizen`.
- Sample intervals are half-open and expressed at `source_sample_rate`; `start_s`
  / `end_s` are derived from them. Millisecond labels in filenames are display only.
- Export duration limits are inclusive and checked against integer source-sample
  lengths. The minimum is rounded up to a whole sample and the maximum down,
  using decimal seconds so exact limits are not lost to timestamp subtraction.
- Commands resolving a manifest's source audio verify its recorded `source.sha256`
  before using its turns. A missing or mismatched hash is an error. An explicit
  `--input-file` override bypasses this check; the caller must preserve the original
  timeline. Downstream manifests record the override's current identity.
- `overlap_with` holds zero-based indices into `turns`. Turns are sorted by start.
- Clip paths are relative to the manifest. An empty `turns` array means no speech
  survived the duration filter. `complete: true` is written last; a manifest is
  reused only when every clip exists with a matching `clip_sha256`.
- `--overwrite` forces diarizers and `audio/export_segments` to rebuild even a
  matching, complete output. Before replacing the old manifest, export removes
  obsolete sibling WAVs referenced by that manifest. Unrelated files are retained.
  The incomplete manifest lists all planned clip names, so retries can also clean
  up clips from an interrupted export. Files already orphaned by older versions
  cannot be attributed safely; use a fresh output directory in that case.
- Export reads audio inside each rendering worker, keeping at most one clip's
  waveform per active worker in memory. `--batch-size` limits submitted clips;
  increasing it does not preload the batch's waveforms.
- All diarization backends also write `segments.raw.json` before optional merging
  and the export duration filter. It contains normalized backend turns, including
  short/long turns, with `merge_applied: false`, `duration_filter_applied: false`,
  `clips_valid: false`, and no clip
  references. Model-internal segmentation/VAD rules still apply. Its parameters
  retain the export request for provenance; neither merge nor duration limits
  have been applied to these raw turns. The processed `segments.json` records
  `raw_manifest_sha256`.
  Skip/resume requires that raw file and matching hash as well as valid clips.
  Rerunning an older output without the raw file reruns inference to recover it;
  filtered clips alone cannot recover discarded turns.
- With merging enabled, `parameters.merge` stores `max_gap_s`,
  `silence_threshold_dbfs`, `frame_ms`, and `adjust_mean`. The final `segments.json` keeps
  `operation: diarize` and adds `merge_applied: true`, `merge_statistics`, and
  `merge_audit` (see [silence-aware merge](#silence-aware-merge)). Mean adjustment
  details are recorded in `merge_mean_adjustment`. Statistics describe the merge
  before duration filtering; exported turn counts may be smaller. Each surviving
  turn's `merge_source_indices` and the audit indices
  refer to `segments.raw.json` turns. Plots describe the final exported turns.
  With `--merge false`, `parameters.merge` is omitted and export behavior is
  unchanged.
- Each diarize command also writes `plot/timeline.png` (speaker Gantt),
  `plot/timeline_duration.png` (segment-length histogram), and
  `plot/timeline_cutoff.png` (remaining count/percent and remaining audio if
  segments shorter than T are dropped). Missing plots are filled in on a
  skipped rerun; a new export overwrites them. `evaluate/plot_diarization`
  writes the same three figures beside `--output-file` for single-manifest
  mode, or under its aggregate `--output-dir` in folder mode.
- `audio/export_segments.py` re-renders any manifest with this shape and rewrites
  it (with fresh `clip*` fields) in the chosen output directory. It also writes
  `plot/timeline.png`, `plot/timeline_duration.png`, and
  `plot/timeline_cutoff.png` beside the exported `segments.json`; these plots
  describe the post-filter turns.

## 3. Purity and speaker manifests

`purity/{consensus,cleanup,merge,collar,snap,align,segment}` and
`speaker/{score,filter,purity}` read a manifest and write a new one (defaults:
`.data/purity/<stage>/<family>/segments.json`, `.data/speaker/<stage>/<family>/segments.json`).
They keep the diarization shape with these differences:

- `operation` names the stage (`consensus`, `cleanup`, `collar`, …); `model` is
  inherited; `parameters` gains `input_manifest` identity plus the stage flags.
- Boundaries are re-normalized against the source audio; `clip`, `clip_sha256`,
  `clip_frames` are dropped and `clips_valid: false` is set. Render clips with
  `audio/export_segments.py`.
- Each turn keeps `confidence` (may be `null`) and any stage-specific fields
  (e.g. similarity scores from `speaker/score`, decisions from `speaker/purity`).

### Silence-aware merge

`purity/merge` reads raw diarization turns and the source waveform, and writes an
unfiltered manifest. Diarizers with merging enabled reuse this same implementation
before filtering and rendering clips in their normal output directory.
Same-speaker, nonoverlapping turns can merge across a gap
of 0–`max_gap_s` inclusive (default 1 second), only if no different speaker
intersects the proposed union. All channels and all RMS frames in the gap must
be at or below `silence_threshold_dbfs` (default -40 dBFS). Frames are 20 ms by
default, including the final partial frame; a zero-length gap needs no acoustic
check. Nonfinite audio blocks a merge. This is an energy-based silence criterion,
not a speech classifier; tune the threshold on your recordings.

Speaker labels and original outer boundaries remain unchanged. The merged span
includes the intervening silence. `merge_source_indices` references the normalized,
time-sorted input turns; `merge_audit` records candidate decisions, gaps, and the
maximum per-channel frame RMS in dBFS (`null` when unmeasured or digitally silent).
`merge_statistics` records input/output turn counts, the number of merged gaps,
maximum-duration rejections, and the final maximum gap used.
Merged turns drop clip-specific scores/transcripts; confidence is the minimum of
component confidences when all are known, otherwise null. Overlap indices are
recomputed and clip references invalidated.

Standalone `purity/merge` applies no duration limit. Integrated diarization passes
its inclusive maximum duration (default 15 seconds) into the merge: a candidate
union that would exceed the limit is rejected before the acoustic silence check,
and that candidate starts a new merge chain. The subsequent duration filter still
removes individual overlong turns, and there is no automatic splitting. Duration
includes silence. For standalone `purity/merge`, use `audio/export_segments`
afterwards; its merge manifest retains overlong chains, while integrated
diarization retains original component turns in `segments.raw.json` and merge
decisions (including duration rejections) in the final audit.

When `adjust_mean` is enabled, integrated diarization measures the mean of the
duration-filtered turns for each input video. It retries from the original raw
turns, moving `max_gap_s` by 0.1 seconds toward the 7–10 second target, and
records every attempt and its stop reason in `merge_mean_adjustment`. The final
gap and mean are also included in the terminal progress output. A bare
`--merge` or `--adjust-mean` enables the corresponding default-true option;
pass `false` to disable it.

## 4. Speaker profile (`.data/speaker_profiles/<slug>/profile.json`)

```json
{"schema_version": "2.0", "name": "khanh_vy",
 "created_at": "2026-09-09T00:00:00Z", "updated_at": "2026-09-09T00:00:00Z",
 "clips": ["clip_00.wav", "clip_01.wav"],
 "channel_id": null, "channel_name": null, "channel_url": null}
```

Reference clips are copied to `clips/clip_NN.wav`; `--add` appends, `--overwrite` replaces.

## 5. Agent response pair (`scripts/agent/*`)

Each input yields `<stem>_<backend>.txt` (exact model text, may be empty) and
`<stem>_<backend>.json` (source identity, prompts, generation settings, provider
response details, latency, usage/cost when available, and the text file's path,
byte count and SHA-256). The text is written first. A cached pair is reused only
when both files exist and metadata + text digest match. `.json` is reserved for
metadata, so `--output-file result.json` is rejected.

Defaults: `.data/agent/<backend>/<family>/`;
Gemini: `.data/agent/gemini/<model>/<reasoning-effort>/<family>/`.
Gemini Batch state lives under the variant's `work/batch_jobs/` for resume.

## 6. Verifier verdict (`scripts/agent/verifier/*`)

Same pair layout under `.data/agent/verifier/…`. The JSON adds `status` and
either a validated `verdict` or an `error`:

```json
{
  "schema_version": 1,
  "source": {"path": "/abs/clip.wav", "sha256": "…"},
  "operation": "verify",
  "model": "hf",
  "parameters": {"model_id": "google/gemma-4-E2B-it", "device": "cuda:0", "prompt": "…"},
  "status": "success",
  "response": {"path": "/abs/clip_hf.txt", "format": "utf-8 text", "kind": "text", "bytes": 194, "sha256": "…"},
  "verdict": {
    "speaker_purity": "pure",
    "word_completeness": "clipped_word_end",
    "audio_quality": "studio_clean",
    "decision": "reject",
    "failure_codes": ["clipped_word_end"],
    "reason": "…",
    "_latency_s": 0.32, "_inference_mode": "batch", "_batch_job": "batches/123",
    "_response_id": "…", "_usage": {"prompt_tokens": 318}, "_cost": {"pricing_tier": "paid_batch", "total_usd": 0.00042}
  }
}
```

For a pass, the public verdict fields use the same order, include a nonempty
`"emotion": "…"`, and end with a nonempty `"transcript": "…"`. Reject verdicts
omit `transcript` entirely and may omit `emotion`. Runtime schema errors include
`missing_emotion`, `invalid_emotion`, `missing_transcript`,
`unexpected_transcript`, and `transcript_not_last`; the exact raw model response
remains in the sibling text artifact for diagnosis.

Failure records include a stable machine-readable code and an explanation:

```json
{
  "status": "fail",
  "error": {
    "stage": "generation|parse|schema",
    "code": "missing_transcript",
    "message": "The model returned 'pass' without a non-empty transcript."
  },
  "response": {"path": "/abs/clip_hf.txt", "format": "utf-8 text", "kind": "text", "bytes": 194, "sha256": "…"},
  "invalid_verdict": {"decision": "pass", "failure_codes": []}
}
```

Generation errors additionally include the safe exception class as
`error.exception_type`; arbitrary exception text is not persisted because it may
contain credentials. Parse and schema failures preserve exact model text in the
sibling response file, and schema failures preserve the parsed object as
`invalid_verdict`. A generation failure with no model text writes only the JSON.
Legacy verdict-only JSON (no `status`/`response`) remains readable by the analyzer
and comparator.

Validation profile is selected by the prompt text (`scripts/agent/verifier/_verdicts.py`):

| Profile | Selected when prompt equals | Required fields and consistency |
|---|---|---|
| `acoustic_defect_v3` | `prompts/acoustic_defect-3.txt` (default) | Three acoustic dimensions as below; `failure_codes` contains exactly their non-clean values plus optional `unsupported_language` / `singing`; `decision` = pass iff no codes; pass requires nonempty `emotion` and final `transcript`; reject forbids the transcript field |
| `speaker_purity_v1` | `prompts/speaker_purity.txt` | `speaker_purity`; pass iff `pure` |
| `word_boundary_v1` | `prompts/word_boundary.txt` | `boundary_start`, `boundary_end` ∈ clean/clipped; pass iff both clean |
| `vibevoice_v1` | VibeVoice backend (no prompt) | `decision` ∈ pass/reject/uncertain, `num_speakers`, `secondary_speech_s`, `dominant_speaker_id`; `uncertain` is excluded from pass/reject metrics |
| `custom` | any other prompt | only `decision` ∈ pass/reject |

## 7. Verifier analysis (`analysis.sh --input-dir DIR` → `DIR/plot/`)

```text
plot/
  analysis.json           schema_version 2; coverage, decisions, transcripts, emotions, durations, model/prompt groups, failure codes, model_stats, error_stats, CSV digests, plot list
  all_samples.csv         every expected turn (from manifests) + every artifact, even unmatched/invalid
  successful_samples.csv  subset with a schema-valid pass/reject
  report.md               model configuration, statistics, linked error cases
  error_cases.csv         model_id, model, kind, category, family, audio_path, verdict_file, decision, reason, emotion, transcript, failure_stage, assistant_raw_response
  error_stats.csv         model_id, model, kind, category, count, denominator, rate
  coverage.png decisions.png defects.png transcripts.png emotions.png
  dimensions.png measurements.png processing_errors.png by_model.png by_speaker.png emotions_by_speaker.png timeline*.png   (when applicable)
```

Both CSVs share a column order beginning `audio_path, final_verdict,
assistant_raw_response, transcript, transcript_chars, transcript_words, emotion`; nested
values remain in `*_json` columns. Invalid or
missing results have a blank `final_verdict` and are never counted as rejects.
`kind` separates `acoustic`, `eligibility`, and `processing` failures; label rates
use valid artifacts, while processing rates use all artifacts of that model group. Rerunning
refreshes `plot/` and deletes PNGs it previously recorded.

## 8. Verifier comparison (`compare.sh` → `.data/agent/verifier/comparisons/<utc>-<hash>/`)

| File | Contents |
|---|---|
| `summary.json` | `schema_version: 3`; `reference` / `candidates` inventories (incl. `unmatched_artifacts`, `extra_clips`), `families`, per-family `summaries`, `global_summaries` with verifier, transcript, and emotion metrics |
| `pairs.csv` | one row per reference clip per candidate: verifier status/decisions/codes, schema profiles, transcript and emotion status/text, reasons, JSON and audio paths |
| `conflicts.csv`, `conflicts.md` | rows with status `bad_accept`, `false_reject`, `code_mismatch` |
| `transcript_differences.csv` | reference transcripts that differ from or are missing in the candidate |
| `report.md` | aggregate metrics and caveats |
| `plots/candidate_<i>_{defects,transcripts,emotions}.png` | defect recall, transcript comparison, and emotion agreement (when applicable, unless `--no-plots`) |

`status` ∈ `agree`, `bad_accept`, `false_reject`, `code_mismatch`,
`invalid_reference`, `missing_candidate`, `invalid_candidate`, `audio_mismatch`;
only the first four enter metrics. Zero denominators are `null`.

## 9. Evaluation metrics

```json
{"schema_version": 1, "operation": "separation_metrics", "source": {"…": "…"}, "parameters": {"…": "…"},
 "metrics": {"si_sdr_db": 14.82, "scored_samples": 4800000, "mixture_si_sdr_db": 3.1, "si_sdri_db": 11.72}}
```

`evaluate/diarization` (`operation: "diarization_metrics"`) metrics: `der_pct`, `jer_pct`, `missed_speech_s`,
`false_alarm_s`, `speaker_confusion_s`, `correct_speaker_s`, `reference_speaker_s`,
`hypothesis_speaker_s`, `scored_audio_s`, `collar_s`, `skip_overlap`,
`speaker_mapping` (Hungarian assignment).

`mix/mix` writes `mixture.wav`, `speech_reference.wav`, `music_reference.wav`
plus sidecars into `--output-dir`.
