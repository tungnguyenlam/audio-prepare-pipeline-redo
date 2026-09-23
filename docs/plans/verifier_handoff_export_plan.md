# Plan: export a verifier run for a nontechnical recipient

Status: proposed; no exporter implemented by this task.

## Outcome

Deliver one versioned ZIP. The recipient extracts it, opens START_HERE.html,
and can read the instructions, search transcripts, filter available labels, and
listen to accepted clips offline. No Python, terminal, account, server, or network
is required. Assume this is an accepted-audio delivery, not a request to review
rejected samples. Retain the full original verifier run internally.

## Existing behavior and gap

Verifier folders contain verdict JSON and raw response TXT pairs; audio remains
at the recorded source path. Existing plot/report.md and sample CSVs support
technical analysis. scripts/s5-export/bundle.py packages manifest-listed audio
and manifest.json; index.py indexes audio without selecting verifier passes.
Neither provides a verifier-aware, recipient-friendly handoff. Do not index the
whole source folder and assume its contents passed verification.

## Proposed delivered structure

```text
speech_dataset_v1_2026-09-23.zip
└── speech_dataset_v1_2026-09-23/
    ├── START_HERE.html
    ├── READ_ME.txt
    ├── clip_catalog.csv
    ├── audio/
    │   ├── clip_000001.wav
    │   └── clip_000002.wav
    └── supporting_files/
        ├── delivery_summary.txt
        ├── excluded_clips.csv
        ├── release_manifest.json
        └── checksums.sha256
```

- START_HERE.html: title/version, accepted clip count and hours, scope of verifier
  checks, search/filter controls, transcript/emotion when available, and audio
  players using relative paths. Embed catalog data; do not fetch local JSON or
  use a CDN. Escape model text, use lazy audio loading, and paginate large runs.
- READ_ME.txt: three steps (extract ZIP, open start page, play clips), catalog
  fallback, explanation of folders/columns, and instructions for reporting a
  problem by clip ID. Explain that labels/transcripts are model-produced and
  identify any separately performed human review without implying it occurred.
- clip_catalog.csv: Excel-friendly UTF-8 with BOM; clip ID, relative audio path,
  duration seconds, transcript, emotion, source recording ID, speaker ID and
  source start/end times where reliably available. Leave unavailable fields
  blank; do not invent transcripts or globally consistent speaker identities.
  Quote correctly and neutralize spreadsheet formulas in text cells; preserve
  exact original text in the JSON manifest.
- delivery_summary.txt: release date/version, intended use supplied by owner,
  counts and duration, verifier/model/profile, input coverage status, accepted,
  rejected, uncertain, failed and missing counts, audio formats and known limits.
- excluded_clips.csv: IDs, disposition and reason for excluded inputs; excluded
  audio is not shipped by default. Distinguish verifier rejection from processing
  failure. This is an audit list, not a second audio collection.
- release_manifest.json: schema/version, deterministic ID-to-source mapping,
  relative delivered paths, audio hashes, verdict provenance/settings hashes,
  metadata availability, counts and completeness evidence. Use portable source
  identifiers rather than leaking machine-specific absolute paths.
- checksums.sha256: hashes of delivered files, excluding itself. Keep raw model
  responses, provider logs, costs and original JSON internally by default.

Copy original accepted audio bytes and preserve their extension. WAV examples
above assume WAV inputs; export is not conversion or normalization. Use actual
files, never symlinks. Resolve duplicate source basenames with deterministic clip
IDs and preserve traceability in the manifest.

## Selection and completion rules

1. Select one final run/model/prompt configuration. Detect mixed configurations
   and duplicate/conflicting verdicts; require explicit selection rather than
   guessing the newest result or silently aggregating variants.
2. Recursively discover production verifier artifacts, excluding plots, work,
   comparison and experiment directories using the existing discovery behavior.
   Reuse production verdict validation and file/path/hash helpers; inspect and
   extract stable shared behavior only where both commands actually need it.
3. Include only successfully processed, schema-valid pass verdicts with matching
   source audio and intact required response artifacts. Use the production
   compatibility rules for legacy completion metadata. Reject, uncertain,
   malformed, missing and failed outcomes never become accepted audio.
4. Reconcile against an expected input manifest, or a persisted full input list
   where present. A folder of verdicts alone cannot prove all inputs finished.
   If no full input inventory exists, report coverage as unknown and require an
   explicitly labeled partial delivery instead of calling the run complete.
5. A complete run may contain valid rejections. Unresolved failures/missing
   verdicts block a finished release; an explicit partial option must label every
   summary accordingly. Missing/changed accepted audio blocks packaging. Empty
   accepted output should produce a clear error instead of a misleading release.
6. Apply to every verifier backend through its artifact contract. Acoustic v3
   provides transcript/emotion; narrower/custom profiles and prompt-free
   VibeVoice may not. Surface the actual profile and field availability rather
   than claiming every pass satisfies the full acoustic/transcript rubric.

## Implementation steps

1. Document the exact release contract and map existing verifier discovery,
   completion validation, source resolution, segment joins and bundle behavior.
2. Add standalone scripts/s5-export/export_verifier_handoff.py and same-name
   Bash launcher using the audio environment. Proposed inputs: --input-dir,
   --input-manifest (expected inventory), --output-file (exact ZIP path),
   --dataset-name, --version, and explicit --allow-partial. Defaults should
   support one selected run without requiring recipient-facing customization.
   Do not chain pipeline stages or invoke models.
3. Build and validate the selected inventory first. Generate recipient documents,
   embedded offline browser and portable metadata from that same inventory so
   counts and filenames cannot drift. Keep provider payload logic untouched.
4. Stage under .data/exports/, build a ZIP with one top-level folder, verify
   references/hashes/counts, and atomically publish only after success. Refuse
   overwriting existing releases unless explicitly requested; prefer a new
   version for corrected deliveries. Use stderr for progress and stdout for the
   final artifact path. Keep all runtime artifacts out of Git.
5. Update docs/commands.md, docs/agent_verifier.md and docs/data_contract.md with
   the export command, completion rules, sample layout and recipient instructions.

Proposed invocation (not available yet):

```bash
bash scripts/s5-export/export_verifier_handoff.sh \
  --input-dir .data/s4-agent/verifier/<backend>/<selected-run> \
  --input-manifest .data/manifests/expected_verifier_inputs.json \
  --output-file .data/exports/speech_dataset_v1_2026-09-23.zip \
  --dataset-name speech_dataset --version v1
```

## Acceptance and validation plan

A recipient should be able to extract the ZIP in a different directory/machine,
open the start page without a network connection, find a transcript and play its
clip. Catalog and manifest must match delivered audio count/duration and hashes;
no link may rely on the original repository. The release must account for every
expected input and identify any partial coverage. Check Unicode, filename
collisions, mixed profiles and absent optional metadata when tests are authorized.

During implementation, inspect the diff, validate syntax and launcher --help
without inference, and review generated document/link structure. Do not write or
run tests unless explicitly requested; browser/runtime acceptance remains pending
that authorization. This planning task ran no tests, model calls or export.

## Scope and priority

First delivery: accepted audio, offline listening page, CSV, plain instructions,
summary, exclusions and portable integrity/provenance files in one ZIP. Optional
later additions: XLSX if specifically needed, approved plots, per-clip transcript
TXT files, or a separate rejected-audio review package. Avoid adding these before
there is a recipient need. Recipient language defaults to English and can be
changed to Vietnamese when requested.
