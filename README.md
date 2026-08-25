# Audio Prepare Pipeline Redo

High-throughput audio preparation, YouTube crawling, stem separation, speaker diarization, and benchmark evaluation suite.

## Overview

This repository provides modular, file-backed audio processing utilities, one
shared web backend, and two dedicated frontends:

1. **⚡ SonicPipeline (Large-Scale Processing Engine):** High-throughput batch audio ingestion (playlists, multi-URLs, folder scans), asynchronous worker task queue, dataset management, bulk stem separation (`BS-RoFormer`, `Mel-RoFormer`, `HTDemucs`, `MVSep-MDX23`), batch speaker diarization (`Sortformer`, `DiariZen`, `3D-Speaker`, `Pyannote`), separation benchmark matrix evaluation, hardware telemetry monitoring, and JSONL/CSV manifest exports.
2. **🎙️ SonicStudio (Interactive Exploration Studio):** Single-track audio workstation for manual audio cutting, waveform and spectrogram side-by-side visual comparison, model stem auditioning, and quick parameter exploration.

## Setup guide

Follow these steps on the machine that will run the web backend and model
inference (typically the model server). Development-only machines can stop after
installing the primary environment and editing code; see
[Development and model-server roles](#development-and-model-server-roles).

### Quick setup

For a fresh checkout, the minimum setup for the shared web application is:

```bash
cd /path/to/audio-prepare-pipeline-redo
uv sync
./scripts/start_web.sh
```

Add `HF_TOKEN=hf_...` to the repository-root `.env` before using gated Hugging
Face models. Diarization backends with incompatible dependencies also need the
isolated environments described in step 3 below.

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

# Optional: local Gemma 4 overlap verification
OVERLAP_VERIFIER=gemma4
UNSLOTH_ENDPOINT=http://127.0.0.1:8888/v1/chat/completions
UNSLOTH_MODEL=unsloth/gemma-4-12b-it-GGUF
UNSLOTH_API_KEY=sk-unsloth-...

# Or select Gemini overlap verification
# OVERLAP_VERIFIER=gemini
# GEMINI_API_KEY=...
```

`HF_TOKEN` is required for Pyannote and DiariZen (and for 3D-Speaker overlap
refinement). DiariZen's released weights are CC BY-NC 4.0 and therefore only
suitable for research and other non-commercial use.
`OVERLAP_VERIFIER` selects `gemma4` or `gemini`. The Gemma verifier sends WAV
or MP3 segments to the configured Unsloth Studio chat-completions endpoint;
`UNSLOTH_API_KEY` is optional only when that endpoint does not require
authentication. The Gemini verifier reads `GEMINI_API_KEY` and uses
`gemini-3.1-pro-preview` by default. Set `GEMINI_MODEL` to override that model.
Hugging Face caches default to `.data/huggingface`. Set `HF_HOME` in `.env`
only when a different writable cache location is needed.

Runtime artifacts (downloads, stems, plots, model checkouts) go under `.data/`
and are gitignored — do not commit media or caches.

### 3. Optional isolated diarizer environments

Sortformer/Clustering, DiariZen, and 3D-Speaker pin packages that conflict with the
primary stack. Create these only on the model server, and only if you need
those backends.

**Sortformer / Clustering (NeMo)** — from the repo root, after `uv sync`:

```bash
uv venv --python .venv/bin/python .venv-sortformer
UV_PROJECT_ENVIRONMENT=.venv-sortformer uv sync --frozen --no-dev
uv pip install --python .venv-sortformer/bin/python -r requirements-sortformer.txt
```

**DiariZen Large s80-v2** — uses the upstream Python 3.10 / Torch 2.1 stack
and DiariZen's Pyannote fork:

```bash
# Run these commands from the repository root on the model server.
uv python install 3.10
uv venv --python 3.10 .venv-diarizen
uv pip install --python .venv-diarizen/bin/python \
  torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-diarizen/bin/python -r requirements-diarizen.txt

# Confirm the exact pipeline import used by the worker succeeds.
.venv-diarizen/bin/python -c "import torch, psutil, accelerate; from diarizen.pipelines.inference import DiariZenPipeline; print(torch.__version__)"
```

The commands above are the direct fix for an error like:

```text
DiariZen worker Python does not exist: .../.venv-diarizen/bin/python
```

The default location is `<repo-root>/.venv-diarizen/bin/python`, so no
additional configuration is needed when the environment is created there. If
an existing DiariZen environment lives elsewhere, set its absolute interpreter
path in the repository-root `.env` instead:

```env
DIARIZEN_PYTHON=/absolute/path/to/diarizen-venv/bin/python
```

Restart the web backend after creating the environment or changing `.env`.
The source synchronization scripts deliberately exclude `.venv` and
`.venv-*`, so each isolated environment must be created separately on the
model server; it is not copied from the development machine.

**3D-Speaker** ([modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker);
`speakerlab` is not on PyPI). Toolkit sources are shallow-cloned into
`.data/3d-speaker` automatically on first use:

```bash
uv venv --python .venv/bin/python .venv-3dspeaker
UV_PROJECT_ENVIRONMENT=.venv-3dspeaker uv sync --frozen --no-dev
uv pip install --python .venv-3dspeaker/bin/python -r requirements-3dspeaker.txt
```

When an isolated backend is selected, the server starts a persistent worker in
its corresponding environment and reuses the loaded model. Other models stay
in the primary `.venv`; you do not need to restart the server when switching.

Optional overrides:

| Variable | Purpose |
|---|---|
| `SORTFORMER_PYTHON` | Path to the Sortformer/Clustering interpreter if not `.venv-sortformer` |
| `DIARIZEN_PYTHON` | Path to the DiariZen interpreter if not `.venv-diarizen` |
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

### 6. ViYT-Diar diarization benchmark (offline)

Offline evaluation of diarization systems on
[`tuanduy1612/ViYT-Diar`](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar)
(100 Vietnamese YouTube clips, `test` split). Scripts live under
`benchmark/diarization/`. Run this on the **model server**, not the development
machine.

**Baseline (v1):** Pyannote Community-1
(`pyannote/speaker-diarization-community-1`) in the primary `.venv`.

```bash
# Cache the dataset only (no model inference)
uv run python -m benchmark.diarization --prepare-only

# Smoke test (first N clips)
uv run python -m benchmark.diarization --systems pyannote_community --limit 5

# Full baseline (all 100 clips)
uv run python -m benchmark.diarization --systems pyannote_community

# Compare multiple systems (isolated worker venvs when needed)
uv run python -m benchmark.diarization \
  --systems pyannote_community,pyannote_31,sortformer,clustering,diarizen,3d_speaker
```

| System key | Backend | Environment |
|---|---|---|
| `pyannote_community` | Pyannote Community-1 (baseline) | primary `.venv` |
| `pyannote_31` | Pyannote 3.1 | primary `.venv` |
| `sortformer` | NeMo Sortformer | `.venv-sortformer` |
| `clustering` | NeMo Clustering | `.venv-sortformer` |
| `diarizen` | DiariZen Large s80-v2 | `.venv-diarizen` |
| `3d_speaker` | 3D-Speaker | `.venv-3dspeaker` |

Metrics: Diarization Error Rate (DER) with a 0.25 s collar (`pyannote.metrics`),
plus mean speaker-count absolute error. Outputs (gitignored):

| Path | Contents |
|---|---|
| `benchmark/cache/` | Cached ViYT-Diar WAVs + manifest |
| `benchmark/results/` | Per-run JSON metrics |
| `benchmark/figures/` | Comparison plots (mean DER, boxplot, speaker-count error) |

Requires `HF_TOKEN` in `.env` for Pyannote and DiariZen. Sortformer /
Clustering / DiariZen / 3D-Speaker need their optional environments from step
3. More detail:
[`benchmark/README.md`](benchmark/README.md).

---

## Development and model-server roles

`tungnl5@VF-TUNGNL5-L` is the development machine: write and review code there;
do not run model inference on it. The web backend and model inference run on
`vsf@vsf-242`.

Synchronize source changes with the scripts in `scripts/sync/` (credentials and
runtime `.data/` stay machine-local). Install and maintain the model-serving
environments (primary `.venv` plus optional `.venv-sortformer` /
`.venv-diarizen` / `.venv-3dspeaker`) on the server rather than on the
development machine.

---

## Repository Layout

```text
.
├── benchmark/              # Offline ViYT-Diar diarization benchmark (+ exported figures)
├── src/
│   ├── base/               # ManagedModel lifecycle (load/unload)
│   ├── benchmark/          # AudioMixer and separation benchmark schemas
│   ├── diarization/        # BaseDiarizer, Sortformer, Clustering, DiariZen, 3D-Speaker, Pyannote backends
│   ├── notebooks/          # Interactive Jupyter callers (pipeline1, benchmark, mixer)
│   ├── separation/         # BaseSeparator, BSRoFormer, MelRoFormer, HTDemucs, MVSepMDX23
│   ├── utils/              # File-backed Audio class, AudioCutter, Comparers
│   ├── web_backend/        # Shared API backend and frontend mounts
│   ├── web_pipeline/       # SonicPipeline API domain, queue, dataset manager, frontend
│   ├── web_studio/         # SonicStudio API domain, frontend
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
