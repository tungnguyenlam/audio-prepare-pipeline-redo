# Audio Prepare Pipeline Redo

High-throughput audio preparation, YouTube crawling, stem separation, speaker diarization, and benchmark evaluation suite.

## Overview

This repository provides modular, file-backed audio processing utilities, one
shared web backend, and two dedicated frontends:

1. **⚡ SonicPipeline (Large-Scale Processing Engine):** High-throughput batch audio ingestion (playlists, multi-URLs, folder scans), asynchronous worker task queue, dataset management, bulk stem separation (`BS-RoFormer`, `Mel-RoFormer`, `HTDemucs`, `MVSep-MDX23`), batch speaker diarization (`Sortformer`, `3D-Speaker`, `Pyannote`), separation benchmark matrix evaluation, hardware telemetry monitoring, and JSONL/CSV manifest exports.
2. **🎙️ SonicStudio (Interactive Exploration Studio):** Single-track audio workstation for manual audio cutting, waveform and spectrogram side-by-side visual comparison, model stem auditioning, and quick parameter exploration.

## Setup guide

Follow these steps on the machine that will run the web backend and model
inference (typically the model server). Development-only machines can stop after
installing the primary environment and editing code; see
[Development and model-server roles](#development-and-model-server-roles).

### Prerequisites

- Python >= 3.13
- [`uv`](https://docs.astral.sh/uv/) package manager
- `ffmpeg` on `PATH` (YouTube ingest and audio conversion)
- `git` on `PATH` (optional isolated backends clone toolkits on first use)
- NVIDIA GPU + CUDA drivers recommended for separation and diarization

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone and install the primary environment

```bash
git clone <repo-url> audio-prepare-pipeline-redo
cd audio-prepare-pipeline-redo
uv sync
```

This creates `.venv` and installs the locked application dependencies. Always
run the shared web backend from this primary environment.

### 2. Configure environment variables

Create a repository-root `.env` (loaded automatically at server startup):

```env
HF_TOKEN=hf_...
```

`HF_TOKEN` is required for Pyannote (and for 3D-Speaker overlap refinement).
Hugging Face caches default to `.data/huggingface`. Set `HF_HOME` in `.env`
only when a different writable cache location is needed.

Runtime artifacts (downloads, stems, plots, model checkouts) go under `.data/`
and are gitignored — do not commit media or caches.

### 3. Optional isolated diarizer environments

Sortformer/Clustering and 3D-Speaker pin packages that conflict with the
primary stack. Create these only on the model server, and only if you need
those backends.

**Sortformer / Clustering (NeMo)** — from the repo root, after `uv sync`:

```bash
uv venv --python .venv/bin/python .venv-sortformer
UV_PROJECT_ENVIRONMENT=.venv-sortformer uv sync --frozen --no-dev
uv pip install --python .venv-sortformer/bin/python -r requirements-sortformer.txt
```

**3D-Speaker** ([modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker);
`speakerlab` is not on PyPI). Toolkit sources are shallow-cloned into
`.data/3d-speaker` automatically on first use:

```bash
uv venv --python .venv/bin/python .venv-3dspeaker
UV_PROJECT_ENVIRONMENT=.venv-3dspeaker uv sync --frozen --no-dev
uv pip install --python .venv-3dspeaker/bin/python -r requirements-3dspeaker.txt
```

When Sortformer is selected, the backend starts a persistent worker with
`.venv-sortformer/bin/python`. When 3D-Speaker is selected, it uses
`.venv-3dspeaker/bin/python` the same way. Other models stay in the primary
`.venv`; you do not need to restart the server when switching models.

Optional overrides:

| Variable | Purpose |
|---|---|
| `SORTFORMER_PYTHON` | Path to the Sortformer/Clustering interpreter if not `.venv-sortformer` |
| `THREEDSPEAKER_PYTHON` | Path to the 3D-Speaker interpreter if not `.venv-3dspeaker` |
| `THREEDSPEAKER_ROOT` | 3D-Speaker checkout path if not `.data/3d-speaker` |

### 4. Run the web applications

Start the single backend that serves both frontends (default port `8765`):

```bash
./scripts/start_web.sh
# or with host/port overrides:
./scripts/start_web.sh 8765 127.0.0.1
# or:
uv run python scripts/start_web.py --host 127.0.0.1 --port 8765
```

Then open:

- SonicStudio: `http://127.0.0.1:8765/studio/`
- SonicPipeline: `http://127.0.0.1:8765/pipeline/`

Both frontends call the same `/api/*` backend. `start_studio.*` and
`start_pipeline.*` remain compatibility aliases for this unified service.

Long-running Studio work is serialized by default to avoid overlapping model
loads. Raise the bounded worker count with `STUDIO_QUEUE_CONCURRENCY=2`
(supported range: 1–4) if needed.

### 5. Notebooks (optional)

Interactive callers live under `src/notebooks/`. Open them from that
directory so `os.getcwd()` ends with `notebooks`, and select the project
`.venv` kernel (`ipykernel` is installed by `uv sync`).

---

## Development and model-server roles

`tungnl5@VF-TUNGNL5-L` is the development machine: write and review code there;
do not run model inference on it. The web backend and model inference run on
`vsf@vsf-242`.

Synchronize source changes with the scripts in `scripts/sync/` (credentials and
runtime `.data/` stay machine-local). Install and maintain the model-serving
environments (primary `.venv` plus optional `.venv-sortformer` /
`.venv-3dspeaker`) on the server rather than on the development machine.

---

## Repository Layout

```text
.
├── src/
│   ├── base/               # ManagedModel lifecycle (load/unload)
│   ├── benchmark/          # AudioMixer and separation benchmark schemas
│   ├── diarization/        # BaseDiarizer, Sortformer, Clustering, 3D-Speaker, Pyannote backends
│   ├── notebooks/          # Interactive Jupyter callers (pipeline1, benchmark, mixer)
│   ├── separation/         # BaseSeparator, BSRoFormer, MelRoFormer, HTDemucs, MVSepMDX23
│   ├── utils/              # File-backed Audio class, AudioCutter, Comparers
│   ├── web_backend/        # Shared API backend and frontend mounts
│   ├── web_pipeline/       # SonicPipeline API domain, queue, dataset manager, frontend
│   ├── web_studio/         # SonicStudio API domain and frontend
│   └── yt_crawler/         # YtCrawler YouTube ingestion
├── scripts/
│   ├── start_web.sh        # Launch the shared backend (port 8765)
│   ├── start_web.py
│   ├── start_pipeline.sh   # Compatibility alias for the shared backend
│   ├── start_pipeline.py
│   ├── start_studio.sh     # Compatibility alias for the shared backend
│   ├── start_studio.py
│   └── sync/               # Server synchronization scripts (code & data)
└── docs/
    ├── api_contract.md     # Public API documentation
    └── data_contract.md    # Return-object schemas and contracts
```
