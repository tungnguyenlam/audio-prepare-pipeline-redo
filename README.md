# Audio Prepare Pipeline

Standalone, file-backed commands (Python 3.13) for building Vietnamese speech
datasets: download YouTube audio, separate vocal stems, diarize speakers, refine
turn boundaries, verify clip purity with audio LLMs, and curate manifests.

Each command does one thing, takes paths and flags, and writes files the next
command reads. There is no orchestrator, background worker, or shared in-memory
state — callers compose commands through `.data/` paths and JSON manifests.

## Quick start

```bash
./envs/setup_worker_envs.sh audio                       # lightweight env (.venvs/audio)
bash scripts/download/youtube.sh --url 'https://www.youtube.com/watch?v=VIDEO'
bash scripts/audio/info.sh --input-file .data/download/<family>/<file>.wav
```

Model environments (Demucs/RoFormer, Pyannote, Sortformer, 3D-Speaker, DiariZen,
verifiers) are provisioned per target with the same script; see
[docs/commands.md](docs/commands.md).

## Command groups

| Group | Commands |
|---|---|
| `scripts/download/` | `youtube`, `playlist`, `channel` |
| `scripts/separate/` | `htdemucs`, `htdemucs_ft`, `bs_roformer`, `mel_roformer`, `mvsep_mdx23` |
| `scripts/diarize/` | `sortformer`, `pyannote_community1`, `pyannote_31`, `clustering`, `threed_speaker`, `diarizen` |
| `scripts/audio/` | `info`, `convert`, `cut`, `export_segments`, `compare_waveforms`, `compare_spectrograms` |
| `scripts/speaker/` | `enroll`, `score`, `filter`, `purity` |
| `scripts/purity/` | `consensus`, `cleanup`, `merge`, `collar`, `snap`, `align`, `segment` |
| `scripts/agent/` | raw audio-LLM generation: `gemini`, `endpoint`, `hf` |
| `scripts/agent/verifier/` | pass/reject verifiers `gemini`, `hf`, `endpoint`, `unsloth`, `vllm`, `moss`, `minicpm`, `kimi`, `vibevoice`; offline `analysis`, `compare`, `evaluate_verifier`, `scaffold_experiment` |
| `scripts/mix/`, `scripts/evaluate/` | `mix`; `separation`, `diarization`, `plot_diarization`, `plot_metrics` |
| `scripts/dataset/` | `index`, `filter`, `export`, `bundle` |
| `scripts/sync/` | rsync code/data to the model server and auxiliary hosts |

Every `.py` has a same-name `.sh` launcher that selects the right virtualenv.

## Typical flow

```bash
bash scripts/download/youtube.sh     --url 'https://www.youtube.com/watch?v=VIDEO' --output-dir .data/dl
bash scripts/separate/htdemucs_ft.sh --input-dir .data/dl --output-dir .data/sep
bash scripts/diarize/sortformer.sh   --input-dir .data/sep --output-dir .data/turns      # <stem>/segments.json + clips
bash scripts/purity/cleanup.sh       --input-manifest .data/turns/<stem>/segments.json --output-manifest .data/p/cleaned.json
bash scripts/purity/collar.sh        --input-manifest .data/p/cleaned.json --output-manifest .data/p/collared.json
bash scripts/audio/export_segments.sh --input-manifest .data/p/collared.json --output-dir .data/clips
bash scripts/agent/verifier/gemini.sh --input-dir .data/clips                              # verdict JSON per clip
bash scripts/agent/verifier/analysis.sh --input-dir .data/agent/verifier/gemini/gemini-3-8-flash/medium
```

## Conventions

- Add `--merge --max-gap-s 1 --silence-threshold-dbfs -40` to any diarization
  launcher to merge fragmented same-speaker turns across silence before duration
  filtering. Its normal output directory contains the processed manifest, clips,
  and plots, plus the original `segments.raw.json`; see the
  [merge cookbook](docs/commands.md#merge-before-duration-filtering).
- `--input-file` beats `--input-dir`; `--output-file` is an exact destination for single outputs.
- Audio outputs get a sibling `.json` sidecar; diarizers write `<stem>/segments.json` plus clips;
  purity stages write new manifests and `export_segments` renders them.
- Progress on stderr, output paths on stdout, runtime artifacts under `.data/` (gitignored).

## Documentation

[docs/README.md](docs/README.md) indexes the setup guide, CLI and data contracts,
agent/verifier guide, hardware notes, and experiment history. Agent rules for this
repo are in [AGENTS.md](AGENTS.md).
