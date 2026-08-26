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

`DiarizationResult` schema 2.0 is the canonical handoff from diarization to
verification. Every newly produced result contains:

- a stable `result_id`, creation timestamp, and `audio_id`;
- the complete file-backed source `Audio` snapshot under `source_audio`;
- all declared `Speaker` and `SpeakerTurn` records, including confidence and
  normalized `overlaps_other_speaker` evidence;
- `DiarizationModelInfo` and source/channel identity; and
- a derived `summary` with speaker/turn counts, total speech duration, and
  duration per speaker.

`to_dict()` is the only serialized representation. `from_dict()` round-trips
it, `save()` writes it atomically, and `load()` restores it. Turn `duration_s`
and `summary` are serialized conveniences and are recomputed from canonical
fields on load. Constructors reject duplicate speakers, turns referencing
undeclared speakers, mismatched source identity, and turns beyond a known
source duration. Schema-1.0 payloads can be migrated by passing their source
`Audio` to `from_dict()`.

The derived Python properties are `speaker_count`, `turn_count`,
`total_speech_duration_s`, `duration_per_speaker_s`, and
`turns_by_speaker`. Durable web results live under
`.data/diarization/results/`; verification reports live under
`.data/diarization/verifications/`. Studio session audio IDs persist in
`.data/studio/audio_registry.json` so history can reattach the same source
file after a backend restart. Lazy audition cuts use
`.data/diarization/preview/` and are not registered as audio assets.

Clean turns are a derived output policy, not a second diarization result.
`clean_speaker_turns()` returns new `SpeakerTurn` values and never mutates the
canonical result. The Studio toggle may use those derived boundaries for
preview, verification, and speaker-stem extraction, while durable result JSON
and RTTM export retain all raw turns and overlap evidence. Extracted audio is
tagged `turns:clean` or `turns:raw` so downstream dataset selection can tell
which boundary policy produced it. Speaker-stem, purity-stem, and target-speaker
wav export apply optional pre-roll and post-roll only at cut time. Canonical
`start_s`/`end_s` values are not rewritten. Padded windows that overlap after
this expansion are merged before writing.

Studio listen/export cleanup defaults `boundary_collar_s` to `0` so close
speaker boundaries are not trimmed. The library default of 40 ms remains the
high-purity identity-clip policy when a caller requests it.

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
addition to dataset, `custom_tags`, `system_tags`, stems, canonical
`diarization`, and arbitrary metadata. System tags are pipeline-owned and use
the namespaces `type:`, `stage:`, `speaker:`, `profile:`, and `verification:`.
Only `custom_tags` are user editable. Legacy unnamespaced tags are migrated as
custom tags except known processing markers, which migrate to system tags.
Target
speaker summaries are stored per profile under
`metadata["target_speakers"][profile_name]`; `metadata["target_speaker"]`
contains the most recent result for compatibility.

Each target-speaker summary includes segment and duration denominators plus
`qualified_segment_percent` and `qualified_duration_percent`.
