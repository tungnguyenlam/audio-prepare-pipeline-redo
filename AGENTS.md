# AGENTS.md

Instructions for coding agents working in this repository.

## Hard rules

- Do not write or run test cases unless the user explicitly instructed.
- Do not commit, push, or amend unless the user asked.
- Do not add orchestration that chains crawl → separate → diarize → mix. Callers (notebooks) compose the public APIs.
- Keep runtime artifacts out of git. Write downloads, stems, cuts, and plots under `.data/` (gitignored). Do not commit `.wav` / `.mp3` / similar media.

## What this repo is

A Python 3.13 audio-prepare pipeline: ingest YouTube (or local files), separate stems, optionally diarize, and mix speech+music for separation benchmarks. There is no single pipeline class.

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
| `src/utils/` | File-backed `Audio` plus notebook helpers (`AudioCutter`, comparers, `display_audio`) |
| `src/yt_crawler/` | `YtCrawler` ingest/download |
| `src/separation/` | `BaseSeparator` and backends (`HTDemucs`, `BSRoFormer`, `MelRoFormer`, `MVSepMDX23`) |
| `src/diarization/` | `BaseDiarizer`, schemas, Pyannote/Sortformer backends |
| `src/benchmark/separation/` | `AudioMixer` and mix/benchmark schemas |
| `src/base/model.py` | `ManagedModel` load/unload lifecycle |
| `src/notebooks/` | Interactive callers (`pipeline1.ipynb`, mixer, benchmark) |
| `docs/api_contract.md` | Public method behavior |
| `docs/data_contract.md` | Return-object field contracts |

Imports are `from src....`. The repo is not installed as a package (`tool.uv.package = false`); notebooks put the repo root on `sys.path`.

## Commands

Use `uv`. Do not add pytest/dev-test commands unless the user asked to run tests.

```bash
uv sync
```

Notebooks run from `src/notebooks/` so `os.getcwd()` ends with `notebooks`. Keep that assumption if you edit notebook setup cells.

## Architecture conventions

- **`Audio` is file-backed.** Identity is a path plus metadata (`source_id`, `title`, rates, duration, channels, format). Do not pass in-memory waveforms through the public pipeline APIs.
- **Preserve identity across steps.** Separators keep `source_id`, `title`, and `native_sample_rate`. `native_sample_rate` is the original capture rate; `sample_rate` is the current file rate. Use `Audio.resample_action(target_sample_rate)` against the current file rate.
- **Return new `Audio` for derived files** (`separate`, `cut`, mixer outputs). `Audio.save_to()` is the exception: it copies and mutates `self.path`.
- **`ManagedModel`** (`BSRoFormer`, `MelRoFormer`, diarizers): call `load()` first or use `with model:`. `close()` on separators that wrap managed models should unload.
- **Default audio target** is WAV, 44_100 Hz, mono (`DEFAULT_SAMPLE_RATE` in `AudioClass`). Runtime dirs default to `.data/<component>/{out,work,downloads}`.
- **Custom errors** per backend (`DownloadError`, `DemucsError`, `AudioCutterError`, …), not generic `Exception`.

## Code style

Match neighboring files rather than introducing a new style.

- `from __future__ import annotations` at the top of modules.
- Google-style docstrings on public classes and methods (Args / Returns / Raises).
- Type hints on public signatures. Prefer `str | Path`, `X | None`, and `Optional` only where the file already uses it.
- File names: class modules are PascalCase (`AudioCutter.py`, `HTDemucs.py`); shared helpers are snake_case (`audio_utils.py`, `display_audio.py`, `schemas.py`).
- Lazy-import heavy optional-at-call-time deps (`librosa`, `matplotlib`, IPython) inside the method that needs them.
- Notebook display helpers must not return objects that Jupyter will render a second time. If `show=True`, call `plt.show()`, close the figure, and return `None`. Same for `display_audio`: `display(...)` only.
- Keep changes scoped. Do not refactor unrelated modules or add files the user did not ask for.
- Do not write markdown docs the user did not ask for. If you change a **public** method or return shape, update `docs/api_contract.md` and/or `docs/data_contract.md` in the same change.

## Adding a backend or util

- New separator: subclass `BaseSeparator`, implement `separate(audio: Audio) -> Audio`, export from `src/separation/__init__.py`.
- New diarizer: subclass `BaseDiarizer`, return `DiarizationResult` from `src/diarization/schemas.py` (do not leak backend-specific result types).
- New notebook util: take/return `Audio`, write outputs under `.data/`, follow `AudioCutter` / comparer patterns.
- Prefer small, independently usable classes over a shared mega-helper.

## Out of scope unless asked

- Tests under `tests/` (existing pytest is not a license to add more).
- Installing the project as a package, new orchestration entrypoints, or CI.
- Committing credentials, cookies, or downloaded media.
