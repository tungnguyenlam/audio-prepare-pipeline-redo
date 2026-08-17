# Diarization benchmarks

Offline evaluation scripts for speaker diarization systems. Artifacts:

| Path | Contents |
|---|---|
| `cache/` | Cached ViYT-Diar WAVs + reference RTTM (gitignored) |
| `results/` | Per-run JSON metrics (gitignored) |
| `figures/` | Exported comparison plots (gitignored) |

Runtime audio cache may also land under `.data/benchmark/` when configured.

## ViYT-Diar

Dataset: [`tuanduy1612/ViYT-Diar`](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar)
(100 Vietnamese YouTube clips, `test` split).

Systems run **sequentially**: one model evaluates all clips, unloads, then the
next model starts. After every system finishes, results are merged into one
JSON and comparison figures are written under `benchmark/figures/`.

| Key | System | Env |
|---|---|---|
| `pyannote_community` | Pyannote Community-1 (baseline) | primary `.venv` |
| `pyannote_31` | Pyannote 3.1 | primary `.venv` |
| `sortformer` | NeMo Sortformer | `.venv-sortformer` |
| `clustering` | NeMo Clustering | `.venv-sortformer` |
| `3d_speaker` | 3D-Speaker | `.venv-3dspeaker` |

Run on the **model server** (not the development laptop):

```bash
# Prepare cache only (no model inference)
uv run python -m benchmark.diarization.run_viyt_benchmark --prepare-only

# Smoke test (first N files, one system)
uv run python -m benchmark.diarization.run_viyt_benchmark \
  --systems pyannote_community --limit 5

# Full baseline only
uv run python -m benchmark.diarization.run_viyt_benchmark \
  --systems pyannote_community

# Full model comparison (all registered systems, sequential)
uv run python -m benchmark.diarization.run_viyt_benchmark --all

# Subset comparison
uv run python -m benchmark.diarization.run_viyt_benchmark \
  --systems pyannote_community,pyannote_31,sortformer,clustering,3d_speaker
```

Requires `HF_TOKEN` in the repo-root `.env` for Pyannote (and for 3D-Speaker
overlap refinement). Sortformer / Clustering need `.venv-sortformer`;
3D-Speaker needs `.venv-3dspeaker` (see root README).

### Outputs

For run id `YYYYMMDDTHHMMSSZ`:

- `benchmark/results/<run_id>_<system>.json` — per-model checkpoint (written
  as soon as that model finishes)
- `benchmark/results/<run_id>_viyt_diar.json` — combined multi-model result
- `benchmark/figures/<run_id>_mean_der.png` — mean DER bars
- `benchmark/figures/<run_id>_mean_jer.png` — mean JER bars
- `benchmark/figures/<run_id>_der_boxplot.png` — per-file DER distribution
- `benchmark/figures/<run_id>_speaker_count_error.png`
- `benchmark/figures/<run_id>_der_components.png` — FA / miss / confusion
- `benchmark/figures/<run_id>_per_file_der_compare.png` — per-file overlay
  (multi-model runs)

Metrics: Diarization Error Rate (DER) with 0.25 s collar via
`pyannote.metrics`, plus JER and speaker-count absolute error.
