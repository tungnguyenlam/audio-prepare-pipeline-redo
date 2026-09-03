# AGENTS.md

Instructions for coding agents working in this repository.

## Hard rules

- Do not write or run test cases unless the user explicitly instructed.
- Do not commit, push, or amend unless the user asked.
- Do not add orchestration that chains crawl → separate → diarize → mix. Callers (notebooks / web processors) compose the public APIs.
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

A Python 3.13 audio-prepare pipeline: ingest YouTube (or local files), separate stems, optionally diarize, and mix speech+music for separation benchmarks. There is no single pipeline class.

## Environment roles

- `tungnl5@VF-TUNGNL5-L` is the development machine. Use it to write and
  review code, documentation, and configuration; do not run model inference
  here.
- `vsf@vsf-242` is the model server. Run the web backend and model inference
  there, using its server-local virtual environment and runtime `.data/`
  directory.
- Synchronize source code between the two machines with the scripts under
  `scripts/sync/`. Keep credentials and runtime artifacts machine-local.
- The web server loads the repository-root `.env` at startup. `HF_TOKEN` is
  used for Hugging Face authentication, and `HF_HOME` defaults to
  `.data/huggingface` unless explicitly set.

Operational flow (see `docs/data_contract.md`):

```text
YouTube URL
  -> YtCrawler.download()/ingest()
  -> Audio
  -> BaseSeparator.separate()
  -> Audio
  -> optional BaseDiarizer.diarize()
  -> DiarizationResult
```

Benchmark mix flow:

```text
BenchmarkDefinition + speech Audio + music Audio
  -> AudioMixer.mix()
  -> AudioMixResult
```

## Layout

| Path | Role |
|---|---|
| `src/utils/` | File-backed `Audio` plus notebook helpers (`AudioCutter`, comparers) |
| `src/yt_crawler/` | `YtCrawler` ingest/download |
| `src/separation/` | `BaseSeparator` and backends (`HTDemucs`, `BSRoFormer`, `MelRoFormer`, `MVSepMDX23`) |
| `src/diarization/` | `BaseDiarizer`, schemas, Pyannote/Sortformer/Clustering/3D-Speaker backends, `zero_contamination.py` |
| `src/benchmark/separation/` | `AudioMixer` and mix/benchmark schemas |
| `src/base/model.py` | `ManagedModel` load/unload lifecycle |
| `src/notebooks/` | Interactive callers (`pipeline1.ipynb`, mixer, benchmark) |
| `src/web_backend/` | Shared REST backend and frontend mounts (port `8765`) |
| `src/web_studio/` | **SonicStudio** — Interactive audio exploration API domain, Experiment tab, and frontend |
| `src/web_pipeline/` | **SonicPipeline** — Large-scale task queue, dataset management API domain and frontend |
| `scripts/` | Runner scripts (`start_pipeline.sh`, `start_studio.sh`, `sync/` scripts) |
| `docs/api_contract.md` | Public method behavior |
| `docs/data_contract.md` | Return-object field contracts |

Imports are `from src....`. The repo is not installed as a package (`tool.uv.package = false`); notebooks and server scripts put the repo root on `sys.path`.

## Commands

Use `uv`. Do not add pytest/dev-test commands unless the user asked to run tests.

```bash
uv sync
```

### Starting the Web Applications

- **Start the shared backend (both frontends, port 8765):**
  ```bash
  ./scripts/start_web.sh [port] [host]
  # or
  uv run python scripts/start_web.py --host 127.0.0.1 --port 8765
  ```

  SonicStudio is at `/studio/` and SonicPipeline is at `/pipeline/`.
  The `start_studio.*` and `start_pipeline.*` launchers are compatibility
  aliases for the same backend.

Notebooks run from `src/notebooks/` so `os.getcwd()` ends with `notebooks`. Keep that assumption if you edit notebook setup cells.

## Architecture conventions

- **`Audio` is file-backed.** Identity is a path plus metadata (`source_id`, `title`, rates, duration, channels, format). Do not pass in-memory waveforms through the public pipeline APIs.
- **Preserve identity across steps.** Separators keep `source_id`, `title`, and `native_sample_rate`. `native_sample_rate` is the original capture rate; `sample_rate` is the current file rate. Use `Audio.resample_action(target_sample_rate)` against the current file rate.
- **Return new `Audio` for derived files** (`separate`, `cut`, mixer outputs). `Audio.save_to()` is the exception: it copies and mutates `self.path`.
- **`ManagedModel`** (`BSRoFormer`, `MelRoFormer`, diarizers): call `load()` first or use `with model:`. `close()` on separators that wrap managed models should unload.
- **Default audio target** is WAV, 44_100 Hz, mono (`DEFAULT_SAMPLE_RATE` in `AudioClass`). Runtime dirs default to `.data/<component>/{out,work,downloads}`.
- **Custom errors** per backend (`DownloadError`, `DemucsError`, `AudioCutterError`, …), not generic `Exception`.

## Web Architectures

- **`src/web_backend/`:** One aiohttp application owns both API domains and
  serves the independent frontends at `/studio/` and `/pipeline/`.

- **`src/web_studio/` (SonicStudio):** Designed for single-sample inspection, visual waveform/spectrogram comparer, manual cutting, fast preview, A/B audition, and the **Experiment tab** (zero-contamination single-speaker diarization with Gemma 4 direct-audio UI and funnel visualizer). Frontend is flat `static/{index.html,app.js,style.css,experiment.js,experiment.css}`.
- **`src/web_pipeline/` (SonicPipeline):** Designed for large-scale operations:
 - Frontend is flat `static/{index.html,app.js,style.css}`; job updates use SSE (`GET /api/events`).
 - Queues are **per GPU** (plus a `cpu`/`mps` lane): jobs for `cuda:0` and `cuda:1` never share a worker slot.
 - `queue_manager.py`: Async task queue with configurable workers-per-device (1–8), cooperative cancellation, and Server-Sent Events (SSE).
 - `dataset_manager.py`: Persistent audio registry (`.data/pipeline/dataset_registry.json`), dataset groupings, tag filtering, and ML manifest exports (JSONL / CSV / ZIP bundles).
 - `batch_processors.py`: High-throughput batch handlers for playlist ingestion, directory scans, bulk separation, batch diarization, and benchmark matrix generation.
 - `hardware_monitor.py`: Real-time telemetry (GPU VRAM, CPU utilization, RAM, disk space, and `x Realtime` speedup metrics).

## Code style

Match neighboring files rather than introducing a new style.

- `from __future__ import annotations` at the top of modules.
- Google-style docstrings on public classes and methods (Args / Returns / Raises).
- Type hints on public signatures. Prefer `str | Path`, `X | None`, and `Optional` only where the file already uses it.
- File names: class modules are PascalCase (`AudioCutter.py`, `HTDemucs.py`); shared helpers are snake_case (`audio_utils.py`, `schemas.py`).
- Lazy-import heavy optional-at-call-time deps (`librosa`, `matplotlib`, IPython) inside the method that needs them.
- Notebook display helpers must not return objects that Jupyter will render a second time. If `show=True`, call `plt.show()`, close the figure, and return `None`. Same for `Audio.notebook_display`: `display(...)` only.
- Keep changes scoped. Do not refactor unrelated modules or add files the user did not ask for.
- If you change a **public** method or return shape, update `docs/api_contract.md` and/or `docs/data_contract.md` in the same change.

## Adding a backend or util

- New separator: subclass `BaseSeparator`, implement `separate(audio: Audio) -> Audio`, export from `src/separation/__init__.py`.
- New diarizer: subclass `BaseDiarizer`, return `DiarizationResult` from `src/diarization/schemas.py` (do not leak backend-specific result types).
- New notebook util: take/return `Audio`, write outputs under `.data/`, follow `AudioCutter` / comparer patterns.
- Prefer small, independently usable classes over a shared mega-helper.

## Out of scope unless asked

- Tests under `tests/` (existing pytest is not a license to add more).
- Installing the project as a package, new orchestration entrypoints, or CI.
- Committing credentials, cookies, or downloaded media.
