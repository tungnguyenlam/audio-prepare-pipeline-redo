# Coding-agent instructions

## Non-negotiable rules

- Before any research or experiment task, read `GOAL.md`: acceptance criteria,
  what does not count, the critical-work protocol, and Gemini audio-check limits.
- Agent direct audio inspection: Antigravity can directly read and hear audio files using `view_file` (attaches raw `audio/wav` into native multimodal context; zero ASR proxy). Use for development/diagnostics, but never to replace blind human ground truth (GOAL.md §1–2).
- Do not write or run tests unless the user explicitly asks.
- Append decisions and results to the bottom of `WORKLOG.md` with `cat >> WORKLOG.md`
  (oldest entry first). Start each entry with `## YYYY-MM-DD - title`, keep it under
  ~20 lines, and put long detail in a `.data/` report that the entry links to.
- Commit and push after every completed task.
- File and script names must be explicit about what the file does; never use vague or overly general names (e.g. use `plot_verifier_analysis.py` instead of generic names like `analyze.py`).
- Keep every pipeline stage as an independent command. Never add orchestration
  chaining crawl → separate → diarize → mix.
- Put downloads, stems, cuts, plots, and other runtime artifacts under `.data/`
  (gitignored). Never commit media files.
- Update documentation affected by code changes. Delete or archive stale material
  in the area being changed, but do not broaden focused work into unrelated docs.
- Never log, persist, or echo credentials. Because parsed CLI options are logged,
  secrets must come from environment variables unless explicitly redacted.

## Reading the worklog and large files

Never read `WORKLOG.md` or any file over ~300 lines in full. This keeps context and
quota small. Read the newest entries from the bottom up:

```bash
grep -n '^## ' WORKLOG.md | tail -15                                  # index of recent entries
tac WORKLOG.md | awk '/^## /{if(++n==2){print; exit}} {print}' | tac  # last 2 entries in full
sed -n '120,150p' WORKLOG.md                                         # one entry by line range
grep -n -i 'anchor' WORKLOG.md docs/worklog_archive/*.md | tail      # older history by keyword
```

For research work, read `research/STATUS.md` first; it is the short summary of
current state. Read older entries only when the task needs a specific decision.
At the start of a new month, move the previous month's entries to
`docs/worklog_archive/YYYY-MM.md`. Use the same `grep -n` then `sed -n` approach for
large docs such as `docs/commands.md`.

## Implementation and completion

- Before editing, inspect the command, adjacent backends, shared helpers, launcher,
  and documented file contract. Search for an existing inference, transport,
  model-loading, retry, or artifact path before adding another implementation.
- Honor the grammatical scope of requests about generic “models”, “backends”, or
  capabilities: cover every relevant family or explicitly identify exceptions.
- Keep raw model generation separate from task parsing/validation. Experiments and
  verifiers must reuse production generation paths and retain unparsed responses.
- Share a helper only when at least two current commands use the same stable
  behavior. Keep provider payload construction inside its provider runner.
- Prefer direct functions and simple data flow. Do not add speculative managers,
  factories, registries, adapters, services, state layers, or compatibility shims.
  Simplify root causes; remove dead code and unused dependencies. For behavior-
  preserving refactors, prefer fewer lines and concepts.
- Before completion, inspect the final diff, include every intended new file,
  validate CLI/launcher wiring without invoking paid models, state validations
  omitted because tests were not requested, and append decisions and results to
  `WORKLOG.md` using bash.

## Repository

Python 3.13 standalone commands for preparing speech datasets: YouTube/local
ingest, stem separation, diarization, target-speaker filtering, purity refinement,
audio-model verification, evaluation, mixing, and dataset export.

| Path | Purpose |
|---|---|
| `scripts/s1-download/`, `audio/`, `mix/` | ingest and deterministic audio operations |
| `scripts/s2-separate/` | HTDemucs, BS/Mel-RoFormer, MVSEP-MDX23 |
| `scripts/s3-diarize/` | Sortformer, Pyannote, clustering, 3D-Speaker, DiariZen |
| `scripts/asr/` | VibeVoice-ASR / PhoWhisper transcription with word-level alignment |
| `scripts/speaker/`, `purity/` | enrollment, scoring, filtering, boundary refinement |
| `scripts/s4-agent/` | free-form raw audio-model generation |
| `scripts/s4-agent/verifier/` | validated pass/reject behaviors using shared generation |
| `scripts/evaluate/` | metrics and plots |
| `scripts/vicut_bench/` | ViCut-Bench clean anchors and controlled defects |
| `scripts/s5-export/` | dataset indexing, filtering, export, bundles |
| `scripts/_common/` | private shared file/segment behavior |
| `scripts/sync/` | code/data synchronization between machines |
| `envs/`, `prompts/`, `docs/` | environments, prompts, focused documentation |
| `GOAL.md`, `research/` | research acceptance criteria, plan, papers |
| `archive/` | unmaintained reference code; never run or import |

## CLI and artifact contracts

- Commands accept paths/flags and communicate through files; callers compose them.
- Standard flag shorthands are supported across all pipeline tools:
  `-i`/`-if` (`--input-file`), `-id` (`--input-dir`), `-o`/`-of` (`--output-file`),
  `-od` (`--output-dir`), `-wd` (`--work-dir`), `-w`/`-ow` (`--overwrite`),
  `-c` (`--concurrency`), `-b`/`-bs` (`--batch-size`), `-sr` (`--sample-rate`),
  `-ch` (`--channels`), `-m` (`--model`/`--model-id`), `-d` (`--device`),
  `-v` (`--verbose`), `-p`/`-pf` (`--prompt-file`), `-im` (`--input-manifest`),
  `-om` (`--output-manifest`).
- `--input-file` takes precedence over `--input-dir`; `--output-file` is exact.
- Audio commands write sibling JSON sidecars. Diarizers write
  `<stem>/segments.json` plus clips.
- Progress goes to stderr and successful artifact paths to stdout.
- Every public Python command has a same-name Bash launcher selecting its virtual
  environment and forwarding arguments unchanged. See `docs/commands.md`.
- File and command names must explicitly reflect what the file does rather than being generic (e.g., `plot_verifier_analysis.py` / `.sh` instead of `analyze.py` or `analysis.sh`).

## Machines

- `tungnl5@VF-TUNGNL5-L`: primary AMD Radeon RX 9060 XT development host
  (16 GB VRAM, ROCm 10.0/HIP).
- `CURRENT_MACHINE.md` (gitignored): summarizes local machine-specific hardware specs,
  ROCm/HIP quirks, memory ceilings, and backend constraints.
- **Only the AMD host is available (since 2026-10-12).** Run everything there; do not
  plan work on CUDA-only features (bitsandbytes 4/8-bit, `onnxruntime-gpu`).
- Unavailable for now: `vsf@vsf-242` (`10.148.21.12`, NVIDIA), `loi`
  (`loinh8@10.148.1.176`) and `anhnct@10.148.21.113`. Their `scripts/sync/` scripts
  stay for when they return.

## Out of scope unless requested

- Tests or a new `tests/` directory.
- Package installation, orchestration entrypoints, or CI.
- Credentials, cookies, downloaded media, or other runtime artifacts in git.
