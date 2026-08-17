"""Allow ``uv run python -m benchmark.diarization``."""

from __future__ import annotations

from benchmark.diarization.run_viyt_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
