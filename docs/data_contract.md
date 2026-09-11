# Data & File Contract (Master Gateway)

[← Docs Index](README.md) | [CLI Contract Gateway →](api_contract.md) | [Command Cookbook →](../scripts/COMMANDS.md)

All public pipeline data is file-backed. Objects in memory are not passed across
commands. Commands communicate exclusively through WAV files, sibling JSON sidecars,
and segment manifests.

## 0. Audio Family & Standard Naming Contract

Audio files ingested via download or pipeline inception establish a canonical **Audio Family**:
- **Downloaded audio filename:** `{video_id}_{safe_title_10}-{sample_rate}.wav` (e.g. `dQw4w9WgXcQ_Never-Gonn-48000.wav`)
- **Audio Family Key:** `{video_id}_{safe_title_10}`
- **Dynamic per-family output routing:** Downstream pipeline stages infer this family key from the input filename or its sibling `.json` sidecar, dynamically directing default outputs into `.data/<operation>/[<model>/]<family>/`:
  - Download: `.data/download/<family>/`
  - Separation: `.data/separate/<model>/<family>/`
  - Diarization: `.data/diarize/<model>/<family>/segments.json` (and turn clips)
  - Speaker / Purity: `.data/<operation>/<stage>/<family>/segments.json`
  - Agent exploration: `.data/agent/<backend>/<family>/`
  - Hardened verification: `.data/agent/verifier/<backend>/<family>/`

## 1. Audio Sidecar Contract (`recording.json`)

Commands producing single audio files (`youtube.py`, `convert.py`, `cut.py`,
`htdemucs.py`, `bs_roformer.py`, `mel_roformer.py`, `mvsep_mdx23.py`) write a
sibling `.json` file (`recording.wav` -> `recording.json`):

```json
{
  "schema_version": 1,
  "source": {
    "path": "/absolute/path/to/source.wav",
    "sha256": "abcdef...",
    "video_id": "optional_yt_id",
    "title": "optional_yt_title",
    "origin": {}
  },
  "operation": "separate",
  "model": "htdemucs_ft",
  "parameters": {
    "sample_rate": 48000,
    "channels": 1,
    "stem": "vocals",
    "device": "cpu"
  },
  "output": {
    "sample_rate": 48000,
    "channels": 1,
    "frames": 240000,
    "duration_s": 5.0,
    "format": "wav",
    "sha256": "123456..."
  }
}
```

## 2. Diarization Manifest Contract (`segments.json`)

Diarization engines produce a directory containing `segments.json` and turn WAV clips:

```text
example_htdemucs_ft/
  segments.json
  example_htdemucs_ft_sortformer_spk00_000012340-000018920_0001.wav
```

### Manifest Schema:

```json
{
  "schema_version": 1,
  "source": {
    "path": "/absolute/path/to/example.wav",
    "sha256": "abcdef..."
  },
  "timestamp_origin": "diarized_input",
  "model": "sortformer",
  "parameters": {
    "sample_rate": 48000,
    "channels": 1
  },
  "turns": [
    {
      "speaker_id": "spk00",
      "start_s": 12.34,
      "end_s": 18.92,
      "start_sample": 544194,
      "end_sample": 834372,
      "overlap": false,
      "overlap_with": [],
      "clip": "example_sortformer_spk00_000012340-000018920_0001.wav",
      "clip_sha256": "fedcba...",
      "clip_frames": 290178
    }
  ],
  "speaker_ids": ["spk00"],
  "source_sample_rate": 48000,
  "complete": true
}
```

- Sample intervals are half-open (`[start_sample, end_sample)`).
- Timestamps in filenames are millisecond labels for display; the manifest records precise sample indices.
- Clip paths are relative to `segments.json`.
- `complete: true` is written atomically last.

## 3. Purity Manifest Contract

Purity stages (`consensus.py`, `cleanup.py`, `collar.py`, `snap.py`, `align.py`, `segment.py`)
consume an input manifest and write a new output manifest. When boundaries are altered,
previous clip references are invalidated (`clips_valid: false`). `scripts/audio/export_segments.py`
renders the updated clips into a specified directory.

## 4. Speaker Profile Contract (`profile.json`)

Enrolled speakers live under `.data/speaker_profiles/<name>/`:

```text
.data/speaker_profiles/khanh_vy/
  profile.json
  clips/
    clip_00.wav
    clip_01.wav
```

### Profile Schema:

```json
{
  "schema_version": "2.0",
  "name": "khanh_vy",
  "created_at": "2026-09-09T00:00:00Z",
  "updated_at": "2026-09-09T00:00:00Z",
  "clips": [
    "clip_00.wav",
    "clip_01.wav"
  ],
  "channel_id": null,
  "channel_name": null,
  "channel_url": null
}
```

## 5. Agent Response Pair Contract

Raw agent commands (`scripts/agent/{gemini,endpoint,hf}.sh`) produce two sibling
files for every input. `<stem>_<backend>.txt` contains the exact unparsed model
text. `<stem>_<backend>.json` records source identity, prompts, generation
parameters, provider response details, and the text artifact path, byte count,
and SHA-256 digest. Non-Gemini pairs default under
`.data/agent/<backend>/<family>/`; Gemini pairs default under
`.data/agent/gemini/<model>/<reasoning-effort>/<family>/`. Both files must
exist for artifact cache reuse.

## 6. Verification Verdict Contract

Verifier commands under `scripts/agent/verifier/` produce a sibling response
text and verdict JSON pair for each model response. Non-Gemini defaults live
under `.data/agent/verifier/<backend>/<family>/`; Gemini separates variants at
`.data/agent/verifier/gemini/<model>/<reasoning-effort>/<family>/`. The `.txt` file is the exact
unparsed assistant response. The JSON records its path, byte count, and digest
along with either `status: "success"` plus a verdict or `status: "fail"` plus a
stable error stage/code. A request failure with no model response writes only
the failed JSON artifact. Older verdict JSON without `status` or `response`
remains readable as a legacy result.

```json
{
  "schema_version": 1,
  "source": {
    "path": "/path/to/clip.wav",
    "sha256": "..."
  },
  "operation": "verify",
  "model": "google/gemma-4-E2B-it",
  "parameters": {
    "device": "cuda:0"
  },
  "status": "success",
  "response": {
    "path": "/path/to/clip_hf.txt",
    "format": "utf-8 text",
    "kind": "text",
    "bytes": 194,
    "sha256": "..."
  },
  "verdict": {
    "decision": "pass",
    "reason": "single_speaker_clean",
    "confidence": 0.95,
    "_latency_s": 0.32,
    "_inference_mode": "batch",
    "_batch_job": "batches/123456",
    "_response_id": "provider-response-id",
    "_usage": {"prompt_tokens": 318, "cached_input_tokens": 0},
    "_cost": {"pricing_tier": "paid_batch", "total_usd": 0.00042}
  }
}
```

Failed response example:

```json
{
  "schema_version": 1,
  "source": {"path": "/path/to/clip.wav", "sha256": "..."},
  "operation": "verify",
  "model": "gemini",
  "parameters": {"prompt": "..."},
  "status": "fail",
  "response": {"path": "/path/to/clip_gemini.txt", "sha256": "..."},
  "error": {"stage": "parse", "code": "invalid_json"}
}
```

## 7. Verifier Analysis Contract

`scripts/agent/verifier/analyze.sh` reads verifier JSON artifacts and optional
diarization manifests. Its default destination is `<verdict-dir>/plot/`:

```text
plot/
  analysis.json
  all_samples.csv
  successful_samples.csv
  coverage.png
  decisions.png
  defects.png
  dimensions.png              # when dimension columns are available
  by_speaker.png              # when a diarization manifest is available
  timeline.png                # one or more, when turn timestamps are available
```

`all_samples.csv` contains every expected diarization turn and any unmatched
verifier artifact. `successful_samples.csv` is the subset with a schema-valid
`pass` or `reject` result; "successful" describes verification completion, not
acceptance. Both files use the same column order, beginning with `audio_path`,
`final_verdict`, and `assistant_raw_response`. Known verdict fields, scalar
status flags, timing/provenance fields, and one column per standard failure code
are promoted for direct dataframe use. Nested or custom values remain lossless
in `*_json` columns. Invalid and missing results have a blank final verdict and
are never counted as rejects.

Plots are generated by reading these two CSV files back from disk. Individual
malformed, schema-invalid, unmatched, or missing verifier results produce a
partial report instead of aborting analysis. `analysis.json` records coverage,
decision and duration summaries, prompt/backend/model groups, stable failure
codes, CSV digests, and plot paths.

### Verifier comparison artifacts

`compare.sh --reference-dir REF --candidates-dir CAND` compares saved JSON
artifacts recursively, with one run per candidate argument. It supports all verifier
backend suffixes and legacy verdict-only JSON. See
[the comparator guide](../scripts/agent/verifier/README.md#compare-saved-verifier-runs)
for directory matching and migration from implicit global discovery.

The default destination is a fresh directory under
`.data/agent/verifier/comparisons/<UTC-timestamp>-<selection-hash>/`.
`summary.json` has `schema_version: 2`, `reference` and `candidates` inventories,
`families`, per-family `summaries`, and aggregate `global_summaries`. Each summary
records `matched_clips`, `reference_valid`, `coverage`, status `counts`, `overall`
metrics, `confusion_matrix`, `latency` (including sample count) and `per_criterion`.
Zero-denominator rates and unavailable latency are JSON `null`.

`pairs.csv` has one row per reference clip per candidate, with `candidate`,
`family`, `key`, `status`, decisions, defect codes, reasons, JSON paths and audio
path. Status is one of `agree`, `bad_accept`, `false_reject`, `code_mismatch`,
`invalid_reference`, `missing_candidate`, `invalid_candidate`, or `audio_mismatch`.
Only the first four enter decision metrics. When both source hashes are recorded,
different hashes exclude a pair. Candidate-only keys and their paths are listed
in each candidate inventory's `unmatched_artifacts`, with `extra_clips` counts.

`conflicts.csv` shares the pair columns and includes the three disagreement
statuses. `conflicts.md` adds readable reasons and audio links; `report.md` gives
aggregate metrics and caveats. Optional `plots/candidate_<index>_defects.png`
shows observed caught/missed counts in candidate argument order. Comparisons do
not require audio or response TXT files to be present. Reference and candidates
must be disjoint; output must be outside their trees and new or empty. Every
candidate must have at least one valid pair before report creation. Missing or
invalid results reduce coverage and are not interpreted as model rejects.

## 8. Evaluation Metrics Contract

Evaluation commands output metrics JSON with complete source provenance:

```json
{
  "schema_version": 1,
  "operation": "separation_metrics",
  "source": [...],
  "parameters": {"sample_rate": 48000},
  "metrics": {
    "si_sdr_db": 14.82,
    "sdr_db": 15.11
  }
}
```
