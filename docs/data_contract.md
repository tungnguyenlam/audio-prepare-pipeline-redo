# Data and file contract

[← Index](README.md) · [CLI contract](cli_contract.md) · [Cookbook](commands.md)

Commands exchange data only through WAV files, sibling JSON sidecars, and JSON
manifests. Every JSON artifact is written atomically (temp file + rename).

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
  `parameters` and the output hash (see [CLI contract](cli_contract.md)).

## 2. Diarization manifest (`<stem>/segments.json`)

```text
<output-dir>/<safe stem>/
  segments.json
  <stem>_<model>_<speaker>_<start_ms:09d>-<end_ms:09d>_<index:04d>.wav
```

```json
{
  "schema_version": 1,
  "source": {"path": "/abs/input.wav", "sha256": "…"},
  "operation": "diarize",
  "model": "sortformer",
  "parameters": {"device": "cuda:0", "min_duration_s": 2.0, "max_duration_s": 15.0, "…": "…"},
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
- `overlap_with` holds zero-based indices into `turns`. Turns are sorted by start.
- Clip paths are relative to the manifest. An empty `turns` array means no speech
  survived the duration filter. `complete: true` is written last; a manifest is
  reused only when every clip exists with a matching `clip_sha256`.
- `audio/export_segments.py` re-renders any manifest with this shape and rewrites
  it (with fresh `clip*` fields) in the chosen output directory.

## 3. Purity and speaker manifests

`purity/{consensus,cleanup,collar,snap,align,segment}` and
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
  "model": "google/gemma-4-E2B-it",
  "parameters": {"device": "cuda:0", "prompt": "…"},
  "status": "success",
  "response": {"path": "/abs/clip_hf.txt", "format": "utf-8 text", "kind": "text", "bytes": 194, "sha256": "…"},
  "verdict": {
    "decision": "reject",
    "speaker_purity": "pure",
    "word_completeness": "clipped_word_end",
    "audio_quality": "studio_clean",
    "failure_codes": ["clipped_word_end"],
    "reason": "…",
    "_latency_s": 0.32, "_inference_mode": "batch", "_batch_job": "batches/123",
    "_response_id": "…", "_usage": {"prompt_tokens": 318}, "_cost": {"pricing_tier": "paid_batch", "total_usd": 0.00042}
  }
}
```

Failure: `"status": "fail"`, `"error": {"stage": "generation|parse|schema", "code": "…"}`;
a request failure with no model text writes only the JSON. Legacy verdict-only
JSON (no `status`/`response`) is still readable by the analyzer and comparator.

Validation profile is selected by the prompt text (`scripts/agent/verifier/_verdicts.py`):

| Profile | Selected when prompt equals | Required fields and consistency |
|---|---|---|
| `acoustic_defect_v1` | `prompts/acoustic_defect.txt` | `speaker_purity` ∈ pure/secondary_speaker/overlapping_speech; `word_completeness` ∈ complete/clipped_word_start/clipped_word_end; `audio_quality` ∈ studio_clean/music_bleed/noisy_reverberant/distorted; `failure_codes` = exactly the non-clean values; `decision` = pass iff no codes |
| `speaker_purity_v1` | `prompts/speaker_purity.txt` | `speaker_purity`; pass iff `pure` |
| `word_boundary_v1` | `prompts/word_boundary.txt` | `boundary_start`, `boundary_end` ∈ clean/clipped; pass iff both clean |
| `vibevoice_v1` | VibeVoice backend (no prompt) | `decision` ∈ pass/reject/uncertain, `num_speakers`, `secondary_speech_s`, `dominant_speaker_id`; `uncertain` is excluded from pass/reject metrics |
| `custom` | any other prompt | only `decision` ∈ pass/reject |

## 7. Verifier analysis (`analysis.sh --input-dir DIR` → `DIR/plot/`)

```text
plot/
  analysis.json           coverage, decisions, durations, model/prompt groups, failure codes, model_stats, error_stats, CSV digests, plot list
  all_samples.csv         every expected turn (from manifests) + every artifact, even unmatched/invalid
  successful_samples.csv  subset with a schema-valid pass/reject
  report.md               model configuration, statistics, linked error cases
  error_cases.csv         model_id, model, kind, category, family, audio_path, verdict_file, decision, reason, failure_stage, assistant_raw_response
  error_stats.csv         model_id, model, kind, category, count, denominator, rate
  coverage.png decisions.png defects.png
  dimensions.png measurements.png processing_errors.png by_model.png by_speaker.png timeline*.png   (when applicable)
```

Both CSVs share a column order beginning `audio_path, final_verdict,
assistant_raw_response`; nested values remain in `*_json` columns. Invalid or
missing results have a blank `final_verdict` and are never counted as rejects.
`kind` separates `acoustic` labels from `processing` failures; acoustic rates use
valid artifacts, processing rates use all artifacts of that model group. Rerunning
refreshes `plot/` and deletes PNGs it previously recorded.

## 8. Verifier comparison (`compare.sh` → `.data/agent/verifier/comparisons/<utc>-<hash>/`)

| File | Contents |
|---|---|
| `summary.json` | `schema_version: 2`; `reference` / `candidates` inventories (incl. `unmatched_artifacts`, `extra_clips`), `families`, per-family `summaries`, `global_summaries` with `matched_clips`, `reference_valid`, `coverage`, status `counts`, `overall`, `confusion_matrix`, `latency`, `per_criterion` |
| `pairs.csv` | one row per reference clip per candidate: `candidate`, `family`, `key`, `status`, decisions, defect codes, reasons, JSON and audio paths |
| `conflicts.csv`, `conflicts.md` | rows with status `bad_accept`, `false_reject`, `code_mismatch` |
| `report.md` | aggregate metrics and caveats |
| `plots/candidate_<i>_defects.png` | caught/missed defect counts (unless `--no-plots`) |

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
