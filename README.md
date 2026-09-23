# Audio Prepare Pipeline

Standalone, file-backed commands (Python 3.13) for building Vietnamese speech
datasets: download YouTube audio, separate vocal stems, diarize speakers, refine
turn boundaries, verify clip purity with audio LLMs, and curate manifests.

Each command does one thing, takes paths and flags, and writes files the next
command reads. There is no orchestrator, background worker, or shared in-memory
state — callers compose commands through `.data/` paths and JSON manifests.

## Quick start

```bash
./envs/setup_worker_envs.sh download                   # YouTube env (.venvs/download)
bash scripts/s1-download/youtube.sh --url 'https://www.youtube.com/watch?v=VIDEO'
./envs/setup_worker_envs.sh audio                       # audio env + cached Silero JIT in ~/.cache/silero-vad
bash scripts/audio/info.sh --input-file .data/s1-download/<family>/<file>.wav
```

The download target keeps the current `yt-dlp` (including its default EJS
solver) and project-local JavaScript runtimes separate from the downstream
audio tools. It installs Deno, Node/npm, Bun, and QuickJS under
`.venvs/download`; runtime caches stay under `.data/s1-download/`.

Model environments (Demucs/RoFormer, Pyannote, Sortformer, 3D-Speaker, DiariZen,
verifiers) are provisioned per target with the same script; see
[docs/commands.md](docs/commands.md).

## Command groups

| Group | Commands |
|---|---|
| `scripts/s1-download/` | `youtube`, `playlist`, `channel`, multi-source `crawl` |
| `scripts/s2-separate/` | `htdemucs`, `htdemucs_ft`, `bs_roformer`, `mel_roformer`, `mvsep_mdx23` |
| `scripts/s3-diarize/` | `sortformer`, `pyannote_community1`, `pyannote_31`, `clustering`, `threed_speaker`, `diarizen` |
| `scripts/audio/` | `info`, `convert`, `cut`, `segment_vad`, `segment_tts`, `export_segments`, `compare_waveforms`, `compare_spectrograms` |
| `scripts/speaker/` | `enroll`, `score`, `filter`, `purity` |
| `scripts/purity/` | `consensus`, `cleanup`, `merge`, `collar`, `snap`, `align`, `segment` |
| `scripts/s4-agent/` | raw audio-LLM generation: `gemini`, `endpoint`, `hf` |
| `scripts/s4-agent/verifier/` | pass/reject verifiers `gemini`, `hf`, `endpoint`, `unsloth`, `vllm`, `moss`, `minicpm`, `kimi`, `vibevoice`; offline `plot_verifier_analysis` (aliases `analysis`, `analyze`), `compare`, `evaluate_verifier`, `scaffold_experiment` |
| `scripts/mix/`, `scripts/evaluate/` | `mix`; `separation`, `diarization`, `plot_diarization`, `plot_metrics`, `prepare_viyt_diar`, `run_viyt_diar` |
| `scripts/s5-export/` | `index`, `filter`, `export`, `bundle`, `export_verifier_handoff` |
| `scripts/sync/` | rsync code/data to the model server and auxiliary hosts |

Every `.py` has a same-name `.sh` launcher that selects the right virtualenv.

## Typical flow

```bash
bash scripts/s1-download/youtube.sh     --url 'https://www.youtube.com/watch?v=VIDEO' --output-dir .data/dl
bash scripts/s2-separate/htdemucs_ft.sh --input-dir .data/dl --output-dir .data/sep
# Optional: precompute reusable .data/vad/<stem>.json reports with evaluate/silero_jit.sh.
bash scripts/s3-diarize/sortformer.sh   --input-dir .data/sep --output-dir .data/turns
bash scripts/purity/cleanup.sh       --input-manifest .data/turns/<stem>/segments.json --output-manifest .data/p/cleaned.json
bash scripts/purity/collar.sh        --input-manifest .data/p/cleaned.json --output-manifest .data/p/collared.json
bash scripts/audio/export_segments.sh --input-manifest .data/p/collared.json --output-dir .data/clips
bash scripts/s4-agent/verifier/gemini.sh --input-dir .data/clips                              # verdict JSON per clip
bash scripts/s4-agent/verifier/plot_verifier_analysis.sh --input-dir .data/s4-agent/verifier/gemini/gemini-3-8-flash/medium
```

## Conventions

- Diarization enables same-speaker merging across silence by default (fixed
  `--max-gap-s 1.0`). Dynamic merge (adjusting the merge gap in 0.1-second steps
  to hit a 7–10 second mean per video) is disabled by default; enable it with
  `--dynamic-merge` (or `--adjust-mean`). Merging can be disabled entirely with
  `--merge false`. The dynamic merge status is explicitly reported in video-level
  plot titles. The output directory contains the processed manifest, clips, plots,
  and the intermediate `segments.raw.json` plus `segments.merged.json`; see the
  [merge cookbook](docs/commands.md#merge-before-duration-filtering).
- Turns over 15 seconds are recursively cut at cached Silero VAD valleys by
  default, but only where the speech probability is strictly below
  `--vad-cut-threshold` (default `0.1`). If no report is supplied, the
  diarizer/exporter lazily creates or reuses a report under
  `.data/vad/auto/<source-sha256>.json`. `--vad-device auto` prefers `cuda:0`
  and falls back to CPU after an inference error; pass `--vad-report` (or
  `--vad-report-dir` for directory runs) to reuse precomputed reports. If no
  eligible valley exists, the oversized turn is left for the final duration
  filter; use `--long-segment-strategy drop` to skip VAD cuts explicitly.
- `--input-file` beats `--input-dir`; `--output-file` is an exact destination for single outputs.
- Audio outputs get a sibling `.json` sidecar; diarizers write `<stem>/segments.json` plus clips
  and `<stem>/plot/` (including before-merge, after-merge, and post-filter plots);
  directory diarization runs also write aggregate plots under each collection’s
  `_plot/`, including the parent output collection when inputs are nested.
  Purity stages write new manifests and `export_segments` renders them.
- Progress on stderr, output paths on stdout, runtime artifacts under `.data/` (gitignored).

## Documentation

- [Setup, CLI rules, and command cookbook](docs/commands.md)
- [Data and file contracts](docs/data_contract.md)
- [Agent generation and verifier behavior](docs/agent_verifier.md)
- [Hardware and provisioning notes](docs/hardware.md)
- [Experiment decisions](docs/experiments.md)

Coding-agent rules are in [AGENTS.md](AGENTS.md).
