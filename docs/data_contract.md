# Data contract

## `Audio`

`Audio` is file-backed. Every derived file keeps the source video and YouTube
channel identity unless a caller explicitly replaces it.

| Field | Type | Meaning |
|---|---|---|
| `path` | `Path` | Current audio artifact on disk. |
| `source_id` | `str` | Video/file identity. A YouTube video's ID remains stable across derivatives. |
| `title` | `str \| None` | Video/file display title. |
| `source_url` | `str \| None` | Original video/source URL. |
| `channel_id` | `str \| None` | Stable YouTube channel or uploader ID. |
| `channel_name` | `str \| None` | Human-readable YouTube channel name. |
| `channel_url` | `str \| None` | Canonical YouTube channel URL. |
| `sample_rate` | `int \| None` | Current file rate. |
| `native_sample_rate` | `int \| None` | Original capture rate. |
| `duration_s` | `float \| None` | Current file duration. |
| `channels` | `int \| None` | Current channel count. |
| `format` | `str` | Current file extension/format. |
| `history` | `tuple[str, ...]` | Ordered transformation fingerprints. |

The adjacent `audio.sidecar` JSON stores the same identity and rate fields.
`Audio.from_file()` restores them, and `Audio.with_file()` preserves them for
separation, cutting, resampling, and other derived artifacts.

## Diarization and target-speaker results

`DiarizationResult` and `TargetSpeakerResult` expose optional `channel_id`,
`channel_name`, and `channel_url` alongside their video-level `audio_id`.
Diarizer backends copy these fields from the input `Audio`; target-speaker
filtering preserves them from the scored result.

`Speaker.global_speaker_id` names a globally enrolled identity that was
injected into a supporting diarization pipeline. Turns continue to reference
the result-local `speaker_id`; clients render `global_speaker_id` as the
speaker name when present.

## `SpeakerProfile`

A profile stores `name`, reference `clip_paths`, `created_at`, `updated_at`, and
optional channel provenance. Profiles are global identities reusable across
channels. Reference clips are the portable source of truth; each supporting
diarization pipeline builds its own enrollment representation before inference.

## `SpeakerPurityResult`

Each result covers one diarization-turn candidate and records:

- candidate identity (`audio_id`, `speaker_id`, `start_s`, `end_s`,
  `profile_name`);
- `decision` (`pass`, `reject`, or `error`) plus a stable `reason`;
- union overlap duration and ratio from other-speaker turns;
- every successfully computed `SpeakerSimilarityWindow` and embedding model
  metadata;
- an operational error message only when `decision == "error"`.

`passed` and `min_target_similarity` are derived properties rather than stored
state. Only `pass` enters the dataset. Both `reject` and `error` fail closed,
while `error` remains distinguishable so callers may retry it.

## `OverlapVerificationResult`

Every direct-audio overlap verifier returns the same two-field mapping:

| Field | Type | Meaning |
|---|---|---|
| `overlap` | `bool` | Whether speech from at least two speakers occurs simultaneously. |
| `reason` | `str` | A short, non-empty explanation of the decision. |

Malformed or incomplete model output raises `OverlapVerifierError` instead of
being coerced into a decision.

SonicStudio purity reports may attach a `direct_overlap` object to each
serialized `SpeakerPurityResult`. It is `null` when the optional verifier was
enabled but the candidate already failed stage one. Otherwise it records the
backend, model, normalized `overlap` decision, model reason, and any request
error. This web-report evidence does not change the core
`SpeakerPurityResult` dataclass.

## Pipeline `AudioItem`

Pipeline registry items expose the `Audio` source/channel fields directly in
addition to dataset, tags, stems, diarization, and arbitrary metadata. Target
speaker summaries are stored per profile under
`metadata["target_speakers"][profile_name]`; `metadata["target_speaker"]`
contains the most recent result for compatibility.

Each target-speaker summary includes segment and duration denominators plus
`qualified_segment_percent` and `qualified_duration_percent`.
