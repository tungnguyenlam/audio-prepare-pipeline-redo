# Audio Prepare Pipeline Redo

High-throughput audio preparation, YouTube crawling, stem separation, speaker diarization, and benchmark evaluation suite.

## Overview

This repository provides modular, file-backed audio processing utilities, one
shared web backend, and two dedicated frontends:

1. **⚡ SonicPipeline (Large-Scale Processing Engine):** High-throughput batch audio ingestion (playlists, multi-URLs, folder scans), asynchronous worker task queue, dataset management, bulk stem separation (`BS-RoFormer`, `Mel-RoFormer`, `HTDemucs`, `MVSep-MDX23`), batch speaker diarization (`Sortformer`, `3D-Speaker`, `Pyannote`), separation benchmark matrix evaluation, hardware telemetry monitoring, and JSONL/CSV manifest exports.
2. **🎙️ SonicStudio (Interactive Exploration Studio):** Single-track audio workstation for manual audio cutting, waveform and spectrogram side-by-side visual comparison, model stem auditioning, and quick parameter exploration.

## Quickstart

### Prerequisites
- Python >= 3.13
- [`uv`](https://docs.astral.sh/uv/) package manager
- `ffmpeg` installed on the system

### Installation
```bash
uv sync
```

### Development and Model-Server Roles

`tungnl5@VF-TUNGNL5-L` is the development machine: it is used for writing and
reviewing code and does not run model inference. The web backend and model
inference run on `vsf@vsf-242`.

Synchronize source changes with the scripts in `scripts/sync/`. Install and
maintain the model-serving environments on the server rather than on the
development machine.

Sortformer uses a separate environment because NeMo pins shared packages that
conflict with the primary stack. On the model server, create it from the
repository's locked application environment and install the pinned NeMo
requirements:

```bash
uv venv --python .venv/bin/python .venv-sortformer
UV_PROJECT_ENVIRONMENT=.venv-sortformer uv sync --frozen --no-dev
uv pip install --python .venv-sortformer/bin/python -r requirements-sortformer.txt
```

3D-Speaker similarly needs an isolated environment
([modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker);
`speakerlab` is not published on PyPI). The toolkit sources are
shallow-cloned into `.data/3d-speaker` automatically on first use:

```bash
uv venv --python .venv/bin/python .venv-3dspeaker
UV_PROJECT_ENVIRONMENT=.venv-3dspeaker uv sync --frozen --no-dev
uv pip install --python .venv-3dspeaker/bin/python -r requirements-3dspeaker.txt
```

Always run the shared backend from the primary application environment:

```bash
./scripts/start_web.sh
```

When Sortformer is selected, the backend starts a persistent worker with
`.venv-sortformer/bin/python`, loads NeMo inside that process, and reuses the
model across batch items. When 3D-Speaker is selected, the backend starts
`.venv-3dspeaker/bin/python` the same way (and clones the toolkit into
`.data/3d-speaker` if needed). Other utilities remain in the primary `.venv`;
the server does not need to be restarted when switching models. Set
`SORTFORMER_PYTHON` or `THREEDSPEAKER_PYTHON` only when an isolated
interpreter is stored elsewhere. Override the 3D-Speaker checkout path with
`THREEDSPEAKER_ROOT` when it is not at `.data/3d-speaker`.

### Environment Configuration

The web server automatically loads the repository-root `.env` at startup. For
Pyannote diarization, define the Hugging Face token as:

```env
HF_TOKEN=hf_...
```

Hugging Face model files are cached under `.data/huggingface` by default. Set
`HF_HOME` in `.env` only when a different writable cache location is needed.

---

## Running the Web Applications

Start the single backend that serves both frontends on port `8765`:

```bash
./scripts/start_web.sh
# or
uv run python scripts/start_web.py --port 8765
```

Open SonicStudio at `http://127.0.0.1:8765/studio/` and SonicPipeline at
`http://127.0.0.1:8765/pipeline/`. Both frontends call the same `/api/*`
backend. The existing `start_studio.*` and `start_pipeline.*` commands remain
compatibility aliases and also start this unified service.

Long-running Studio work is serialized by default to avoid overlapping model
loads and running out of memory. Advanced users can raise the bounded worker
count with `STUDIO_QUEUE_CONCURRENCY=2` (supported range: 1–4).

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
