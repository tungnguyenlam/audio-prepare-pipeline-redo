# Scripts Directory Index

This directory contains standalone commands, launchers, and utilities for the audio preparation pipeline.

For detailed command options and usage examples, see the [Command Cookbook and Contracts](COMMANDS.md).

Every public Python command has a matching Bash launcher. For example, run
`bash scripts/download/youtube.sh --url 'https://youtu.be/UuQgxxfU_Hc'`.
Use `.sh` with Bash; `.py` files require Python. Lightweight launchers select
`.venvs/audio`, `.venv-audio`, `.venvs/main`, then `.venv`, or the executable path
in `AUDIO_PYTHON`. Model launchers select their dedicated environments.

## Subdirectories

| Directory | Purpose | Key Commands |
| :--- | :--- | :--- |
| [`download/`](download/) | YouTube media ingestion | `youtube.py`, `playlist.py`, `channel.py` |
| [`separate/`](separate/) | Stem separation | `htdemucs.sh`, `htdemucs_ft.sh`, `bs_roformer.sh`, `mel_roformer.sh`, `mvsep_mdx23.sh` |
| [`diarize/`](diarize/) | Speaker diarization | `sortformer.sh`, `pyannote_community1.sh`, `pyannote_31.sh`, `clustering.sh`, `threed_speaker.sh`, `diarizen.sh` |
| [`audio/`](audio/) | Audio inspection & transformations | `info.py`, `convert.py`, `cut.py`, `export_segments.py`, `compare_waveforms.py`, `compare_spectrograms.py` |
| [`speaker/`](speaker/) | Target speaker operations | `enroll.py`, `score.sh`, `filter.py`, `purity.sh` |
| [`purity/`](purity/) | Purity refinement pipeline | `consensus.py`, `cleanup.py`, `collar.py`, `snap.py`, `align.sh`, `segment.py` |
| [`verify/`](verify/) | Audio candidate verifiers | `hf.sh`, `gemini.sh`, `endpoint.sh`, `unsloth.sh`, `moss.sh`, `minicpm.sh`, `kimi.sh`, `vibevoice.sh` |
| [`mix/`](mix/) | Speech + music mixing | `mix.py` |
| [`evaluate/`](evaluate/) | Metrics & plotting | `separation.py`, `diarization.py`, `plot_diarization.py`, `plot_metrics.py` |
| [`dataset/`](dataset/) | Dataset manifest management | `index.py`, `filter.py`, `export.py`, `bundle.py` |
| [`_common/`](_common/) | Private shared helpers | `files.py`, `segments.py` |
| [`sync/`](sync/) | Machine synchronization | `code_to_server.sh`, `data_to_server.sh`, etc. |

## Environment Setup Scripts

Grouped and organized under [`envs/`](../envs/) (with forwarding wrappers in `scripts/`):
- `envs/setup_worker_envs.sh`: Sets up isolated worker environments for Sortformer, 3D-Speaker, DiariZen, and VibeVoice.
- `envs/setup_kimi_env.sh`: Sets up isolated environment for Kimi-Audio.
- `envs/setup_minicpmo_env.sh`: Sets up isolated environment for MiniCPM-o.
