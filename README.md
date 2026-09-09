# Audio Prepare Pipeline

Standalone, file-backed audio processing commands for YouTube ingestion, stem
separation, speaker diarization, speaker purity verification, and dataset curation.

Each command performs one operation, accepts paths and flags, and writes file
artifacts consumed by downstream commands. There is no implicit orchestration,
background worker, or hidden queue.

## Quick start

```bash
# Setup lightweight execution environment
./envs/setup_worker_envs.sh audio
# or manually:
# uv venv --python 3.13 .venvs/audio
# uv pip install --python .venvs/audio/bin/python -r envs/requirements-audio.txt

# Inspect an audio file
uv run python scripts/audio/info.py --input-file .data/source.wav

# Download a video as mono WAV (44.1 kHz default)
uv run python scripts/download/youtube.py --url 'https://www.youtube.com/watch?v=VIDEO' --output-dir .data/downloads
```

For detailed setup of model environments (Demucs, RoFormer, Pyannote, Sortformer,
NeMo, 3D-Speaker, DiariZen, verifiers), see [Standalone Commands Reference](scripts/COMMANDS.md).

## Command groups

| Group | Key commands | Launchers / Scripts |
|---|---|---|
| **Download** | Single video, playlist, channel | `scripts/download/{youtube,playlist,channel}.py` |
| **Separation** | HTDemucs, HTDemucs FT, BS-RoFormer, Mel-RoFormer, MVSEP-MDX23 | `scripts/separate/{htdemucs,htdemucs_ft,bs_roformer,mel_roformer,mvsep_mdx23}.sh` |
| **Diarization** | Pyannote 3.1 & Community-1, Sortformer, NeMo Clustering, 3D-Speaker, DiariZen | `scripts/diarize/{pyannote_31,pyannote_community1,sortformer,clustering,threed_speaker,diarizen}.sh` |
| **Audio tools** | Metadata info, format conversion, cutting, segment clip export, waveform & spectrogram comparer plots | `scripts/audio/{info,convert,cut,export_segments,compare_waveforms,compare_spectrograms}.py` |
| **Speaker ops** | Reference profile enrollment, turn scoring, threshold filtering, candidate sliding-window purity verification | `scripts/speaker/{enroll,score,filter,purity}.py` |
| **Purity stages** | Diarizer consensus, turn cleanup, collar adjustment, acoustic boundary snapping, word alignment, duration segmentation | `scripts/purity/{consensus,cleanup,collar,snap,align,segment}.py` |
| **Verification** | Gemma 4 direct-audio, Endpoint, Unsloth, vLLM, Gemini, MOSS, MiniCPM-o, Kimi, VibeVoice-ASR speaker count | `scripts/verify/{hf,endpoint,unsloth,vllm,gemini,moss,minicpm,kimi,vibevoice}.sh`, `evaluate_verifier.py` |
| **Mix & eval** | SMR-controlled speech+music mixing, SI-SDR separation metrics, DER diarization metrics, Gantt & metrics plots | `scripts/mix/mix.py`, `scripts/evaluate/{separation,diarization,plot_diarization,plot_metrics}.py` |
| **Dataset tools** | File-based directory indexing, duration/tag filtering, JSONL/CSV manifest export, ZIP bundling | `scripts/dataset/{index,filter,export,bundle}.py` |

## Typical operational workflow

Commands compose through standard filesystem paths and manifest JSON files:

```bash
# 1. Download YouTube source
uv run python scripts/download/youtube.py \
  --url "https://www.youtube.com/watch?v=EXAMPLE" \
  --output-dir .data/downloads

# 2. Separate vocal stem
bash scripts/separate/htdemucs_ft.sh \
  --input-file .data/downloads/example-48000.wav \
  --output-dir .data/separated

# 3. Diarize speaker turns (generates <stem>/segments.json + WAV clips)
bash scripts/diarize/sortformer.sh \
  --input-file .data/separated/example-48000_htdemucs_ft.wav \
  --output-dir .data/turns

# 4. Refine purity through independent stages
uv run python scripts/purity/cleanup.py \
  --input-manifest .data/turns/example-48000_htdemucs_ft/segments.json \
  --output-manifest .data/purity/cleaned.json

uv run python scripts/purity/collar.py \
  --input-manifest .data/purity/cleaned.json \
  --output-manifest .data/purity/collared.json

# 5. Render finalized clips
uv run python scripts/audio/export_segments.py \
  --input-manifest .data/purity/collared.json \
  --output-dir .data/clips/final
```

## CLI and file contracts

- `--input-file` takes precedence over `--input-dir`.
- For single-output commands, `--output-file` is the exact destination path.
- Diarizers output `<input-stem>/segments.json` plus turn clips.
- Audio-producing commands write sibling JSON metadata (`.wav` → `.json`).
- Manifest-editing purity stages output updated manifests; `export_segments.py` renders the resulting audio clips.
- See [`docs/api_contract.md`](docs/api_contract.md) for full CLI parameters and [`docs/data_contract.md`](docs/data_contract.md) for JSON schemas.
