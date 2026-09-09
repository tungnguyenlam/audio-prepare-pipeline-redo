# Data & File Contract (Master Gateway)

[← Docs Index](README.md) | [CLI Contract Gateway →](api_contract.md) | [Command Cookbook →](../scripts/COMMANDS.md)

All public pipeline data is file-backed. Objects in memory are not passed across
commands. Commands communicate exclusively through WAV files, sibling JSON sidecars,
and segment manifests.

## 0. Audio Family & Standard Naming Contract

Audio files ingested via download or pipeline inception establish a canonical **Audio Family**:
- **Downloaded audio filename:** `{video_id}_{safe_title_10}-{sample_rate}.wav` (e.g. `dQw4w9WgXcQ_Never-Gonn-16000.wav`)
- **Audio Family Key:** `{video_id}_{safe_title_10}`
- **Dynamic per-family output routing:** Downstream pipeline stages infer this family key from the input filename or its sibling `.json` sidecar, dynamically directing default outputs into `.data/<operation>/[<model>/]<family>/`:
  - Download: `.data/download/<family>/`
  - Separation: `.data/separate/<model>/<family>/`
  - Diarization: `.data/diarize/<model>/<family>/segments.json` (and turn clips)
  - Speaker / Purity: `.data/<operation>/<stage>/<family>/segments.json`
  - Verification: `.data/verify/<model>/<family>/`

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
    "sample_rate": 44100,
    "channels": 1,
    "stem": "vocals",
    "device": "cpu"
  },
  "output": {
    "sample_rate": 44100,
    "channels": 1,
    "frames": 220500,
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
    "sample_rate": 44100,
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
  "source_sample_rate": 44100,
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

## 5. Verification Verdict Contract

Verifier commands (`hf.sh`, `gemini.sh`, `vibevoice.sh`, etc.) produce a JSON file for each analyzed audio file:

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
  "verdict": {
    "decision": "pass",
    "reason": "single_speaker_clean",
    "confidence": 0.95,
    "_latency_s": 0.32
  }
}
```

## 6. Evaluation Metrics Contract

Evaluation commands output metrics JSON with complete source provenance:

```json
{
  "schema_version": 1,
  "operation": "separation_metrics",
  "source": [...],
  "parameters": {"sample_rate": 44100},
  "metrics": {
    "si_sdr_db": 14.82,
    "sdr_db": 15.11
  }
}
```
