# AGENTS.md

Instructions for coding agents working in this repository.

## Hard rules

- Do not write or run test cases unless the user explicitly instructed.
- Do not commit, push, or amend unless the user asked.
- Do not add orchestration that chains crawl → separate → diarize → mix. Callers compose the standalone commands.
- Keep runtime artifacts out of git. Write downloads, stems, cuts, and plots under `.data/` (gitignored). Do not commit `.wav` / `.mp3` / similar media.
- Constantly write and maintain documentation. Proactively update or archive stale docs so that documentation accurately and truthfully reflects the current state of the pipeline, APIs, models, and interfaces. Never let documentation drift behind code changes.

## Engineering ideology

Complexity must justify itself. Existing complexity is not a reason to preserve it.

- Prefer simple, direct implementations over architectural abstraction. Prefer deleting unnecessary code over adding new layers.
- Do not introduce abstractions for hypothetical future reuse, and avoid speculative extensibility. Before adding a new abstraction, justify it with multiple real current usages.
- Do not create managers, factories, registries, adapters, wrappers, services, generic frameworks, or extra state layers unless they solve a concrete current problem.
- Prefer functions and straightforward classes over elaborate design patterns. A small amount of duplication is acceptable when it keeps code easier to understand.
- Keep modules cohesive, but do not split code into tiny files merely for architectural purity.
- Keep data flow obvious and traceable. Avoid storing the same state in multiple places; derived state should usually be computed rather than synchronized through effects or duplicated variables.
- Do not fix complexity-induced bugs by adding more complexity. When debugging, simplify first and fix the root cause.
- Remove dead code, stale compatibility layers, abandoned experiments, unused dependencies, and unused APIs. Git history is the archive; dead implementations do not need to remain in the active codebase.
- Optimize primarily for one developer being able to open the code and understand the execution path quickly.
- Backward compatibility is not automatically valuable for internal or unused APIs.
- Avoid large test suites for trivial implementation details; test behavior where failure would actually matter (and only when the user asked for tests).
- For refactors that leave functionality unchanged, prefer `deleted LOC > added LOC`.
- If two designs satisfy the same requirement, choose the one with fewer concepts, fewer layers, fewer dependencies, and less state.

## What this repo is

A Python 3.13 audio-prepare pipeline: ingest YouTube (or local files), separate stems, diarize, verify speaker purity, evaluate separation and diarization, and prepare speech datasets. Every operation is an independent CLI command under `scripts/`.

## Environment roles

- `tungnl5@VF-TUNGNL5-L` is the primary development machine equipped with an **AMD Radeon RX 9060 XT (16 GB VRAM, ROCm 10.0 / HIP)**.
- `vsf@vsf-242` (`10.148.21.12`) is the model server for running NVIDIA GPU separation and diarization queues.
- `loi` (`loinh8@10.148.1.176`) is an auxiliary compute node.
- Synchronize source code and runtime data across machines using the dedicated utilities under `scripts/sync/` (`*_server.sh`, `*_loi.sh`, `*_anhnct.sh`). Keep credentials and runtime artifacts machine-local.

## Layout

| Path | Role |
|---|---|
| `scripts/download/` | YouTube single video, playlist, and channel download commands |
| `scripts/separate/` | Stem separation commands and launchers (`htdemucs`, `bs_roformer`, `mel_roformer`, `mvsep_mdx23`) |
| `scripts/diarize/` | Speaker diarization commands and launchers (`sortformer`, `pyannote`, `clustering`, `threed_speaker`, `diarizen`) |
| `scripts/speaker/` | Target speaker enrollment, turn scoring, threshold filtering, and candidate purity verification |
| `scripts/purity/` | Purity refinement pipeline stages (`consensus`, `cleanup`, `collar`, `snap`, `align`, `segment`) |
| `scripts/verify/` | Candidate audio verifiers (`hf`, `gemini`, `endpoint`, `unsloth`, `moss`, `minicpm`, `kimi`, `vibevoice`) |
| `scripts/audio/` | Core audio utilities (`info`, `convert`, `cut`, `export_segments`, `compare_waveforms`, `compare_spectrograms`) |
| `scripts/mix/` | Calibrated speech + music mixing with controlled SMR |
| `scripts/evaluate/` | Separation SI-SDR metrics, diarization DER metrics, Gantt timeline and comparison plots |
| `scripts/dataset/` | File-based directory indexing, manifest filtering, JSONL/CSV export, and ZIP bundling |
| `scripts/_common/` | Shared private file and segment helpers (`files.py`, `segments.py`) |
| `scripts/sync/` | Multi-machine synchronization utilities |
| `docs/` | System architecture, CLI and file contracts, strategy documentation |

## CLI & File Conventions

- Each command is standalone, accepts paths and flags, and writes file artifacts consumed downstream.
- `--input-file` takes precedence over `--input-dir`.
- For single-output operations, `--output-file` is the exact destination.
- Audio-producing commands write sibling `.json` metadata sidecars (`recording.wav` → `recording.json`).
- Diarization commands write `<stem>/segments.json` plus turn clips.
- Commands log progress to stderr and emit successful output paths to stdout.
- Model commands have bash launchers (`.sh`) that select their isolated Python environment and forward arguments unchanged.

## Out of scope unless asked

- Tests under `tests/` (existing pytest is not a license to add more).
- Installing the project as a package, new orchestration entrypoints, or CI.
- Committing credentials, cookies, or downloaded media.
