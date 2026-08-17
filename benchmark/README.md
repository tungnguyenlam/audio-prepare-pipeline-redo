# Diarization benchmarks

Offline evaluation scripts for speaker diarization systems. Artifacts:

| Path | Contents |
|---|---|
| `cache/` | Cached ViYT-Diar WAVs + reference RTTM (gitignored) |
| `results/` | Per-run JSON metrics (gitignored) |
| `figures/` | Exported comparison plots (gitignored) |

Runtime audio cache may also land under `.data/benchmark/` when configured.

## ViYT-Diar baseline

Dataset: [`tuanduy1612/ViYT-Diar`](https://huggingface.co/datasets/tuanduy1612/ViYT-Diar)
(100 Vietnamese YouTube clips, `test` split).

**Baseline system (v1):** Pyannote Community-1
(`pyannote/speaker-diarization-community-1`), run from the primary `.venv`.

Run on the **model server** (not the development laptop):

```bash
# Prepare cache only (no model inference)
uv run python -m benchmark.diarization.run_viyt_benchmark --prepare-only

# Smoke test (first N files)
uv run python -m benchmark.diarization.run_viyt_benchmark \
  --systems pyannote_community --limit 5

# Full baseline
uv run python -m benchmark.diarization.run_viyt_benchmark \
  --systems pyannote_community

# Compare multiple systems (workers use isolated venvs when needed)
uv run python -m benchmark.diarization.run_viyt_benchmark \
  --systems pyannote_community,pyannote_31,sortformer,clustering,3d_speaker
```

Requires `HF_TOKEN` in the repo-root `.env` for Pyannote (and for 3D-Speaker
overlap refinement). Sortformer / Clustering need `.venv-sortformer`;
3D-Speaker needs `.venv-3dspeaker` (see root README).

Metrics: Diarization Error Rate (DER) with 0.25 s collar via
`pyannote.metrics`, plus speaker-count absolute error. Figures are written to
`benchmark/figures/`.
