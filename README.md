# Audio Prepare Pipeline Redo

High-throughput audio preparation, YouTube crawling, stem separation, speaker diarization, and benchmark evaluation suite.

## Overview

This repository provides modular, file-backed audio processing utilities and two dedicated web platforms:

1. **⚡ SonicPipeline (Large-Scale Processing Engine):** High-throughput batch audio ingestion (playlists, multi-URLs, folder scans), asynchronous worker task queue, dataset management, bulk stem separation (`BS-RoFormer`, `Mel-RoFormer`, `HTDemucs`, `MVSep-MDX23`), batch speaker diarization (`Sortformer`, `Pyannote`), separation benchmark matrix evaluation, hardware telemetry monitoring, and JSONL/CSV manifest exports.
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

---

## Running the Web Applications

### 1. SonicPipeline (Large-Scale Batch Engine)
Starts the batch processing and dataset engineering center on port `8081`:
```bash
./scripts/start_pipeline.sh
# or
uv run python scripts/start_pipeline.py --port 8081
```

### 2. SonicStudio (Interactive Exploration Studio)
Starts the interactive audio editor and spectrogram comparer on port `8080`:
```bash
./scripts/start_studio.sh
# or
uv run python scripts/start_studio.py --port 8080
```

### 3. Unified Web Starter
```bash
# Start large-scale pipeline (default)
./scripts/start_web.sh pipeline

# Start interactive studio
./scripts/start_web.sh studio
```

---

## Repository Layout

```text
.
├── src/
│   ├── base/               # ManagedModel lifecycle (load/unload)
│   ├── benchmark/          # AudioMixer and separation benchmark schemas
│   ├── diarization/        # BaseDiarizer, Sortformer, Pyannote backends
│   ├── notebooks/          # Interactive Jupyter callers (pipeline1, benchmark, mixer)
│   ├── separation/         # BaseSeparator, BSRoFormer, MelRoFormer, HTDemucs, MVSepMDX23
│   ├── utils/              # File-backed Audio class, AudioCutter, Comparers
│   ├── web_pipeline/       # SonicPipeline (Batch queue, dataset manager, server, UI)
│   ├── web_studio/         # SonicStudio (Interactive exploration server, UI)
│   └── yt_crawler/         # YtCrawler YouTube ingestion
├── scripts/
│   ├── start_pipeline.sh   # Launch SonicPipeline (port 8081)
│   ├── start_pipeline.py
│   ├── start_studio.sh     # Launch SonicStudio (port 8080)
│   ├── start_studio.py
│   ├── start_web.sh        # Unified runner
│   └── start_web.py
└── docs/
    ├── api_contract.md     # Public API documentation
    └── data_contract.md    # Return-object schemas and contracts
```
