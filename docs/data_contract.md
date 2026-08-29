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
fields on load.         Load is more tolerant than direct construction so Studio history can
        reopen persisted files: unknown viewer keys are ignored, last-frame
        timestamps that overshoot ``duration_s`` are clamped, and speakers
        referenced only by turns are added. Constructors still reject invalid
        newly produced results.

The derived Python properties are `speaker_count`, `turn_count`,
`total_speech_duration_s`, `duration_per_speaker_s`, and
`turns_by_speaker`. Durable web results live under
`.data/diarization/results/`; verification reports live under
`.data/diarization/verifications/`. Studio session audio IDs persist in
`.data/studio/audio_registry.json` so history can reattach the same source
file after a backend restart. Lazy audition cuts use
`.data/diarization/preview/` and are not registered as audio assets.

All default crawler, cutter, and separator output/work directories are anchored
to the repository-root `.data/` directory, independent of the process working
directory. Persisted registries, diarization results, annotations, evaluations,
and verification reports store repository-relative paths (for example,
`.data/pipeline/ingest/example.wav`) and resolve them against the current
checkout when loaded. Legacy absolute paths containing a `.data` component are
remapped to the current checkout automatically. Pipeline registration copies
files from outside `.data/` into
`.data/pipeline/imports/`, so its durable registry never depends on an external
machine-local source path.

The scripts under `scripts/sync/` synchronize this canonical `.data/` tree and
fold the legacy `src/notebooks/.data/` tree into it before transfer. Host-local
model/package caches, virtual environments, backend work directories, cloned
model repositories, and interrupted queue snapshots are excluded by
`scripts/sync/data_excludes.txt`; they are rebuilt on each server. Remote hosts
and checkout paths can be overridden with `SYNC_SERVER_HOST`,
`SYNC_SERVER_REPO`, `SYNC_ANHNCT_HOST`, and `SYNC_ANHNCT_REPO`.

Clean turns are a derived output policy, not a second diarization result.
`clean_speaker_turns()` returns new `SpeakerTurn` values and never mutates the
canonical result. The Studio toggle may use those derived boundaries for
preview, verification, and speaker-stem extraction, while durable result JSON
and RTTM export retain all raw turns and overlap evidence. Extracted audio is
tagged `turns:clean` or `turns:raw` so downstream dataset selection can tell
which boundary policy produced it. Speaker-stem, purity-stem, and target-speaker
wav export apply optional pre-roll and post-roll only at cut time, and only
when the caller enables `add_extra`. Default stem export is the raw labeled
`start_s`/`end_s` windows. When `stop_at_other_speakers` is also on, extra
stops at neighboring other-speaker turns instead of mixing them into the
stem. Canonical `start_s`/`end_s` values are not rewritten. Padded windows
that overlap after this expansion are merged before writing.

Studio listen/export cleanup defaults `boundary_collar_s` to `0` so close
speaker boundaries are not trimmed. The library default of 40 ms remains the
high-purity identity-clip policy when a caller requests it.

`Speaker.global_speaker_id` names a globally enrolled identity that was
injected into a supporting diarization pipeline. Turns continue to reference
the result-local `speaker_id`; clients render `global_speaker_id` as the
speaker name when present.

## Manual diarization annotations and evaluations

Manual ground truth uses `kind: "diarization.annotation"` and schema version
`1.0`. Durable annotation JSON lives under
`.data/diarization/annotations/` and contains:

- `annotation_id`, `revision`, `created_at`, `updated_at`, and a display `name`;
- the stable source `audio_id`, current `session_audio_id`, and a
  `source_audio` snapshot with path, fingerprint, title, duration, sample rate,
  channel count, and format;
- speakers with stable local `speaker_id`, display name, color, and optional
  `global_speaker_id` linking an enrolled speaker profile; and
- turns with stable `turn_id`, `speaker_id`, and second-based `start_s` / `end_s`
  values preserved to microsecond JSON precision.

Writes are atomic and revision checked. A stale client receives HTTP 409 rather
than overwriting a newer revision. Turn validation requires finite in-bounds
timestamps and rejects same-speaker overlap; simultaneous speech is represented
by turns on different speaker lanes. JSON and NIST RTTM are exchange formats.

Manual evaluation compares one annotation with one or more compatible durable
`DiarizationResult` values. Compatibility prefers an exact audio fingerprint,
then an exact resolved path, then stable source identity plus duration. Reports
contain DER, JER, missed speech, false alarm, speaker confusion, scored duration,
optimal one-to-one hypothesis/reference speaker mapping, and per-reference-
speaker coverage. The boundary collar is excluded on each side of every
reference boundary. Reference-overlap regions may optionally be excluded.

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
- union overlap duration and ratio from other-speaker turns (recorded for
  reporting; they do not veto on the Studio Speaker Purity tab);
- embedding `windows` / `min_target_similarity` only when a caller uses
  `SpeakerVerifier.verify_purity` directly (Studio's Speaker Purity tab
  leaves them empty / null);
- an operational error message when the LLM request failed (always stored;
  `decision` is `error` under fail-closed, or still `pass` under fail-open).

`passed` and `min_target_similarity` are derived properties rather than stored
state. Only `pass` enters the dataset. Both `reject` and `error` fail closed,
while `error` remains distinguishable so callers may retry it.

## `VibeVoicePurityResult`

Speaker-count purity from VibeVoice-ASR structured diarization. The transcript
is not part of the decision.

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `str` | Currently `"1.0"`. |
| `audio_id` | `str` | Candidate identity (`Audio.source_id`). |
| `decision` | `pass` / `reject` / `uncertain` | Admit, exclude, or hold for review. |
| `reason` | `str` | `single_speaker`, `multiple_speakers`, `tiny_secondary_speaker`, `empty_output`, `no_speaker_labels`, or `inference_error`. |
| `num_speakers` | `int` | Distinct speaker IDs with positive duration. |
| `secondary_speech_s` | `float` | Total duration of non-dominant speakers. |
| `speaker_turns` | `tuple[VibeVoiceSpeakerTurn, ...]` | Timestamped speaker intervals (no text). |
| `dominant_speaker_id` | `int \| None` | Speaker with the longest total duration. |
| `model` | `DiarizationModelInfo \| None` | Backend metadata (`vibevoice-asr`). |
| `error` | `str \| None` | Present only for `inference_error`. |

`passed` is true only for `decision == "pass"`. A short secondary blip below
`min_secondary_speech_s` is `uncertain`, not `pass` and not `reject`.

SonicStudio purity rows attach a `vibevoice` object when this verifier ran:
`num_speakers`, `dominant_speaker_id`, `secondary_speech_s`, `reason`, and
`speaker_turns`. Those web rows may use `decision: "uncertain"`; that does not
change the embedding `SpeakerPurityResult` dataclass, which remains
`pass` / `reject` / `error`.

## `OverlapVerificationResult`

Every direct-audio overlap verifier returns the same two-field mapping:

| Field | Type | Meaning |
|---|---|---|
| `overlap` | `bool` | Whether speech from at least two speakers occurs simultaneously. |
| `reason` | `str` | A short, non-empty explanation of the decision. |

Malformed or incomplete model output raises `OverlapVerifierError` instead of
being coerced into a decision.

SonicStudio purity reports attach a `direct_overlap` object to each serialized
row whenever the Gemma or Gemini (3.1 Pro or 3.1 Flash-Lite) overlap verifier
ran — including
**Verify All Eligible Turns** (`POST /api/diarization/results/verify`) and
chosen session/library audio (`POST /api/purity/verify`). VibeVoice-ASR runs attach
`vibevoice` instead (see `VibeVoicePurityResult`). In both modes embeddings
do not run, so `windows` is empty and `min_target_similarity` is null.
Diarization overlap duration/ratio remain on the row but do not decide. The
`direct_overlap` object records backend, model, normalized `overlap`
decision, model reason, and any request error. The Speaker Purity tab does
not run an identity filter. This web-report evidence does not
change the core `SpeakerPurityResult` dataclass.

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
