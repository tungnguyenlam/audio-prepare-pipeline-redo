# Verifier audio/transcript review handoff plan

Status: implemented 2026-09-23. Supersedes the initial accepted-only export proposal.

## Recipient workflow

Deliver one versioned ZIP. Extract everything, then either open START_HERE.html
for offline listening/search/review, or open clip_catalog.xlsx / clip_catalog.csv.
Both spreadsheet formats contain clip IDs, relative audio paths, transcripts,
verifier outcomes, missing-transcript status and human correction/review columns.
XLSX has clickable relative audio links, a frozen header, filters, wrapped text
and a review-status dropdown. No new package installation is needed.

HTML supports pagination, search, outcome/review filters, audio players, editable
replacement transcripts and notes, downloadable review CSV, and save/load progress
JSON. Progress is in memory until saved; warn before leaving with unsaved edits.
Use one editing method per review. Spreadsheet edits are not imported by HTML.
Original verifier decisions and transcripts are never overwritten by review edits.

## Completion and rejected/missing transcripts

Keep verifier completion and human review separate. A complete verifier run has a
complete supplied expected inventory and valid pass/reject results for every input,
with intact response artifacts and available hash-matching audio. Valid rejection
is a finished decision, not a processing failure. Human review always starts pending.

Include rejected audio in a separate needs_attention folder so the recipient can
listen and check exclusions. Missing transcripts explicitly require transcription;
provide a blank corrected_transcript field for the reviewer. Missing emotion means
not provided. Do not invent text or call a model as part of export. Failed/invalid/
uncertain/missing results require sender remediation; --allow-partial is an explicit
escape hatch with PARTIAL in the top-level folder and visible partial status in the
HTML, README, summary and manifest. Without an expected inventory, coverage is unknown.
Missing/changed passed audio always blocks export. Partial packages omit unavailable
or changed non-pass audio and retain its issue row. Human opinions do not repair
verifier failures or promote rejected clips into passed audio.

## Deliverable layout

```text
speech_dataset_v1/                    # suffix _PARTIAL for a partial export
├── START_HERE.html
├── READ_ME.txt
├── clip_catalog.csv
├── clip_catalog.xlsx
├── audio/
│   ├── passed/                       # original bytes, stable clip IDs
│   └── needs_attention/              # rejected/unresolved; not approved data
└── supporting_files/
    ├── delivery_summary.txt
    ├── needs_attention.csv
    ├── release_manifest.json
    └── checksums.sha256
```

The release manifest maps IDs to source filenames/hashes and relative verdict
filenames/hashes; it carries configuration hashes, backend/model, profiles,
inventory hashes and exact original transcripts. No absolute local source paths,
provider payloads or raw response logs are delivered. Internal source path identity
is hashed into the clip ID to distinguish equal filenames/content from different
recordings. IDs persist across exports on the same source paths; relocation changes
IDs. The source run stays untouched and supplies full internal provenance.

## Implementation sequence and decisions

1. Rename commands and both Python/Bash entrypoints:
   index -> index_audio_manifest; filter -> filter_audio_manifest;
   export -> export_manifest_table; bundle -> bundle_manifest_audio.
   Update active docs; remove old names without aliases. Existing flags/behavior stay.
2. Extract only currently shared verifier discovery and response integrity helpers
   for production/analysis/export; reuse the production verdict validator across
   all backends. Preserve profile differences, including prompt-free VibeVoice.
3. Add export_verifier_handoff.py/.sh in s5-export with an optional output path
   defaulting to .data/s5-export/<manifest-family>_v1.zip. Select expected clips before checking settings;
   retain differing configurations across unique clips and offer explicit hash/folder
   selection when competing verdicts exist. Reconcile repeatable --input-manifest
   arguments (indexed audio entries or exported segment turns). Never infer full
   coverage from discovered verdicts alone. Reject malformed/unidentifiable JSON.
4. Build all catalogs/views from one inventory. Copy audio unchanged; generate
   CSV safely for Excel and XLSX with stdlib ZIP/XML. Keep exact original text in
   XLSX/JSON; reject text exceeding Excel limits rather than truncate it silently.
5. Use existing Python without provisioning audio packages; use inventory duration
   metadata and explicitly count unavailable durations. Stage and atomically publish under .data/; checksum delivered files; prevent
   accidental overwrite. Document the independent command and recipient workflow.

## Validation and limits

Review final diff; compile changed Python; check Bash and JavaScript syntax;
validate command --help through the launchers without model inference. Do not write
or run tests without explicit user instruction. Browser/Excel playback, save/load,
archive relocation and end-to-end exports are runtime acceptance work and are not
claimed as verified by syntax/help checks. Original codecs are retained; browsers
may need the external player link for formats they do not support. No human-review
reimport/final training-dataset publication command is included in this scope.
