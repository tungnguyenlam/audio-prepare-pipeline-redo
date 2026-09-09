# Standalone audio commands

Each command in this repository is an independent standalone CLI that performs
one operation, accepts paths and flags, and writes file artifacts consumed by
downstream commands. There is no implicit orchestration or hidden queue.

## Environment setup

Use `bash scripts/<group>/<command>.sh` for Bash invocation. Every public Python
CLI has a same-name `.sh` launcher, including downloads and audio utilities.
Do not pass `.py` files to Bash: Bash parses them as shell code, causing
`from: command not found` and syntax errors. Python files remain callable with
an appropriate Python interpreter.

```bash
bash scripts/download/youtube.sh --url 'https://youtu.be/UuQgxxfU_Hc'
```

Launchers select an existing interpreter, forward arguments unchanged, and retain
the caller's working directory. They never install dependencies. System `ffmpeg`
and `ffprobe` must be available. Export authentication variables (`HF_TOKEN`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `UNSLOTH_API_KEY`) explicitly. `HF_HOME`
defaults to the repository's `.data/huggingface` when not already set.

Provision environments automatically with hardware auto-detection (AMD ROCm vs NVIDIA CUDA vs CPU):

```bash
# Provision all core environments (audio, separation, pyannote, verify, align)
./envs/setup_worker_envs.sh core

# Or provision a specific environment:
./envs/setup_worker_envs.sh separation
./envs/setup_worker_envs.sh pyannote
./envs/setup_worker_envs.sh diarizen
./envs/setup_worker_envs.sh sortformer
./envs/setup_worker_envs.sh 3dspeaker
./envs/setup_worker_envs.sh verify

# Check health and hardware acceleration across all environments:
./envs/setup_worker_envs.sh status
```

Or provision manually via `uv`:

```bash
uv venv --python 3.13 .venvs/audio
uv pip install --python .venvs/audio/bin/python -r envs/requirements-audio.txt

# On NVIDIA GPUs with driver < 580 (CUDA <= 12.8), specify the matching wheel index:
uv venv --python 3.13 .venvs/separation
uv pip install --python .venvs/separation/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
uv pip install --python .venvs/separation/bin/python -r envs/requirements-separation.txt

uv venv --python 3.13 .venvs/pyannote
uv pip install --python .venvs/pyannote/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
uv pip install --python .venvs/pyannote/bin/python -r envs/requirements-pyannote.txt

uv venv --python 3.13 .venvs/verify
uv pip install --python .venvs/verify/bin/python --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
uv pip install --python .venvs/verify/bin/python -r envs/requirements-verify.txt
```

| Launchers | Default environment | Interpreter override |
|---|---|---|
| `download/*.sh`, `audio/*.sh`, `dataset/*.sh`, `evaluate/*.sh`, `mix/mix.sh`, `speaker/{enroll,filter}.sh`, `purity/{consensus,cleanup,collar,snap,segment}.sh`, `verify/evaluate_verifier.sh`, `scaffold_verifier_experiment.sh` | `.venvs/audio` (fallbacks: `.venv-audio`, `.venvs/main`, `.venv`) | `AUDIO_PYTHON` |
| `separate/{htdemucs,htdemucs_ft,bs_roformer,mel_roformer,mvsep_mdx23}.sh` | `.venvs/separation` (fallback: `.venvs/main`) | `SEPARATION_PYTHON` |
| `diarize/{pyannote,pyannote_31,pyannote_community1}.sh` | `.venvs/pyannote` (fallback: `.venvs/main`) | `DIARIZATION_PYTHON` |
| `diarize/{sortformer,clustering}.sh` | `.venvs/sortformer` | `DIARIZATION_PYTHON` |
| `diarize/threed_speaker.sh` | `.venvs/3dspeaker` | `DIARIZATION_PYTHON` |
| `diarize/diarizen.sh` | `.venvs/diarizen` | `DIARIZATION_PYTHON` |
| `speaker/{score,purity}.sh` | `.venvs/pyannote` (fallback: `.venvs/main`) | `DIARIZATION_PYTHON` |
| `purity/align.sh` | `.venvs/align` (fallback: `.venvs/main`) | `ALIGN_PYTHON` |
| `verify/{hf,endpoint,unsloth,gemini}.sh` | `.venvs/verify` (fallback: `.venvs/main`) | `VERIFIER_PYTHON` |
| `verify/vllm.sh` | `.venvs/vllm` | `VLLM_PYTHON` |
| `verify/moss.sh` | `.venvs/moss` | `VERIFIER_PYTHON` |
| `verify/minicpm.sh` | `.venvs/minicpmo` | `VERIFIER_PYTHON` |
| `verify/kimi.sh` | `.venvs/kimi` | `VERIFIER_PYTHON` |
| `verify/vibevoice.sh` | `.venvs/vibevoice` | `VERIFIER_PYTHON` |

## Command cookbook

Run Python commands with `.venvs/main/bin/python` (or `uv run python`), or use
`bash` with the matching `.sh` launcher. `AUDIO_PYTHON` takes an executable path;
it overrides interpreter selection for lightweight commands.
Relative paths below are relative to your current working directory.

### Download

```bash
# Single video download (mono WAV, 16 kHz default)
bash scripts/download/youtube.sh --url 'https://www.youtube.com/watch?v=VIDEO' --output-dir .data/downloads

# Playlist download
bash scripts/download/playlist.sh --url 'https://www.youtube.com/playlist?list=PLAYLIST' --output-dir .data/downloads

# Channel download
bash scripts/download/channel.sh --url 'https://www.youtube.com/@CHANNEL/videos' --output-dir .data/downloads
```

### Audio utilities

```bash
# Inspect audio properties (emits JSON to stdout)
uv run python scripts/audio/info.py --input-file .data/source.wav

# Convert sample-rate, channels, format
uv run python scripts/audio/convert.py --input-dir .data/input --output-dir .data/converted --sample-rate 48000 --channels 1

# Sample-accurate cutting
uv run python scripts/audio/cut.py --input-file .data/source.wav --start 12.34 --end 18.92 --output-file .data/cut.wav

# Render clips from a segment manifest
uv run python scripts/audio/export_segments.py --input-manifest .data/turns/segments.json --output-dir .data/clips

# Compare audio waveforms visually
uv run python scripts/audio/compare_waveforms.py --audio1 .data/a.wav --audio2 .data/b.wav --output-file .data/waveform_comparison.png

# Compare spectrograms visually
uv run python scripts/audio/compare_spectrograms.py --audio1 .data/a.wav --audio2 .data/b.wav --output-file .data/spectrogram_comparison.png
```

### Stem separation

```bash
# HTDemucs / HTDemucs FT
bash scripts/separate/htdemucs_ft.sh --input-dir .data/downloads --output-dir .data/separated --stem vocals
bash scripts/separate/htdemucs.sh --input-file .data/source.wav --output-file .data/vocals.wav

# BS-RoFormer / Mel-Band RoFormer
bash scripts/separate/bs_roformer.sh --input-file .data/source.wav --stem vocals
bash scripts/separate/mel_roformer.sh --input-file .data/source.wav --stem vocals

# MVSEP-MDX23
bash scripts/separate/mvsep_mdx23.sh --input-file .data/source.wav --stem vocals
```

### Speaker diarization

```bash
# Sortformer diarization (defaults to exporting clips between 2s and 15s)
bash scripts/diarize/sortformer.sh --input-dir .data/separated --output-dir .data/turns

# Override clip duration bounds when needed
bash scripts/diarize/sortformer.sh --input-file .data/source.wav --output-dir .data/turns --min-duration-s 1.0 --max-duration-s 30.0

# Pyannote Community-1 / 3.1 (default 2s to 15s)
bash scripts/diarize/pyannote_community1.sh --input-file .data/source.wav --output-dir .data/turns
bash scripts/diarize/pyannote_31.sh --input-file .data/source.wav --output-dir .data/turns

# NeMo Clustering / 3D-Speaker / DiariZen (all default to 2s to 15s)
bash scripts/diarize/clustering.sh --input-file .data/source.wav --output-dir .data/turns
bash scripts/diarize/threed_speaker.sh --input-file .data/source.wav --output-dir .data/turns
bash scripts/diarize/diarizen.sh --input-file .data/source.wav --output-dir .data/turns

# DiariZen with custom segmentation step and binarize thresholds
bash scripts/diarize/diarizen.sh --input-file .data/source.wav --output-dir .data/turns --segmentation-step 0.05 --binarize-onset 0.5 --binarize-offset 0.6
```

### Target speaker operations

```bash
# Enroll reference speaker clips
uv run python scripts/speaker/enroll.py --name khanh_vy --clip .data/ref1.wav --clip .data/ref2.wav

# Score diarization turns against enrolled profile
bash scripts/speaker/score.sh --input-manifest .data/turns/segments.json --profile khanh_vy --output-manifest .data/scored.json

# Filter scored turns by similarity threshold and duration (pure post-processing)
uv run python scripts/speaker/filter.py --input-manifest .data/scored.json --threshold 0.6 --min-duration-s 1.5 --exclude-overlap --output-manifest .data/filtered.json

# Candidate-level purity check with sliding identity windows and overlap veto
bash scripts/speaker/purity.sh --input-manifest .data/turns/segments.json --profile khanh_vy --similarity-threshold 0.6 --output-manifest .data/pure.json
```

### Purity processing stages

```bash
# Diarizer consensus
uv run python scripts/purity/consensus.py --input-manifest .data/primary/segments.json --secondary-manifest .data/secondary/segments.json --output-manifest .data/consensus.json

# Turn cleanup
uv run python scripts/purity/cleanup.py --input-manifest .data/consensus.json --output-manifest .data/cleaned.json

# Collar adjustment
uv run python scripts/purity/collar.py --input-manifest .data/cleaned.json --output-manifest .data/collared.json

# Acoustic boundary snapping
uv run python scripts/purity/snap.py --input-manifest .data/collared.json --output-manifest .data/snapped.json

# ASR word alignment
bash scripts/purity/align.sh --input-manifest .data/snapped.json --output-manifest .data/aligned.json

# Duration segmentation
uv run python scripts/purity/segment.py --input-manifest .data/aligned.json --words-file .data/words.json --output-manifest .data/short.json
```

### Audio verification

Every verifier accepts `--input-file` or `--input-dir`. Pair a single input
with either an exact `--output-file` or a derived path under `--output-dir`;
directory input requires `--output-dir`. Model prompts are placed before audio
in multimodal messages.

```bash
# Gemini direct-audio verifier (3.8 Flash, 3.5 Flash-Lite) with custom prompt and concurrency
bash scripts/verify/gemini.sh --input-dir .data/clips --output-dir .data/verdicts/gemini \
  --model gemini-3.8-flash --reasoning-effort medium --prompt-file prompts/acoustic_defect.txt \
  --concurrency 4 --max-tokens 2048 --temperature 0.0

# Freeform Gemini audio experiment: preserve arbitrary text plus the full API response
bash scripts/verify/freeform/gemini.sh --input-dir .data/clips \
  --output-dir .data/verify/freeform-gemini/baseline \
  --prompt-file .data/prompts/describe_audio.txt --temperature 0.2

# The same freeform contract for an OpenAI-compatible endpoint or local HF model
bash scripts/verify/freeform/endpoint.sh --input-dir .data/clips \
  --output-dir .data/verify/freeform-endpoint/baseline \
  --endpoint http://localhost:8000/v1/chat/completions --model google/gemma-4-E2B-it \
  --prompt-file .data/prompts/describe_audio.txt
bash scripts/verify/freeform/hf.sh --input-dir .data/clips \
  --output-dir .data/verify/freeform-hf/baseline --model-id google/gemma-4-E2B-it \
  --prompt-file .data/prompts/describe_audio.txt

# Gemma 4 / HF direct-audio verifier (E2B, E4B, 12B) with custom prompt
bash scripts/verify/hf.sh --input-dir .data/clips --output-dir .data/verdicts/hf \
  --model-id google/gemma-4-E2B-it --prompt-file prompts/acoustic_defect.txt

# Gemma 4 via vLLM (Offline batch or Server mode)
bash scripts/verify/vllm.sh --input-dir .data/clips --output-dir .data/verdicts/vllm \
  --model google/gemma-4-E2B-it --prompt-file prompts/acoustic_defect.txt

# Unsloth / GGUF direct-audio verifier
bash scripts/verify/unsloth.sh --input-dir .data/clips --output-dir .data/verdicts/unsloth \
  --model unsloth/gemma-4-12b-it-GGUF --gguf-variant UD-Q6_K_XL --prompt-file prompts/acoustic_defect.txt

# Generic OpenAI-compatible endpoint
bash scripts/verify/endpoint.sh --input-dir .data/clips --output-dir .data/verdicts/endpoint \
  --endpoint http://localhost:8000/v1/chat/completions --prompt-file prompts/acoustic_defect.txt

# Multimodal audio verifiers (MOSS, MiniCPM-o, Kimi)
bash scripts/verify/moss.sh --input-dir .data/clips --output-dir .data/verdicts
bash scripts/verify/minicpm.sh --input-dir .data/clips --output-dir .data/verdicts
bash scripts/verify/kimi.sh --input-dir .data/clips --output-dir .data/verdicts

# VibeVoice-ASR speaker count purity verifier
bash scripts/verify/vibevoice.sh --input-dir .data/clips --output-dir .data/verdicts

# Benchmark and evaluate verifier predictions against Gemini teacher reference
uv run python scripts/verify/evaluate_verifier.py \
  --predictions-dir .data/verdicts/vllm \
  --reference-dir .data/verdicts/gemini \
  --output-file .data/evaluate/vllm_vs_gemini.json
```

### Mixing and evaluation

```bash
# Speech + Music mixing with controlled SMR
uv run python scripts/mix/mix.py --speech .data/speech.wav --music .data/music.wav --smr-db 6 --seed 42 --output-dir .data/mix/example

# Separation evaluation (SI-SDR, SDR)
uv run python scripts/evaluate/separation.py --input-file .data/prediction.wav --reference-file .data/mix/example/speech_reference.wav --mixture-file .data/mix/example/mixture.wav --output-file .data/metrics.json

# Diarization evaluation (DER, JER, Confusion)
uv run python scripts/evaluate/diarization.py --input-manifest .data/predicted/segments.json --reference-manifest .data/reference/segments.json --duration 120 --output-file .data/der.json

# Diarization timeline Gantt plot
uv run python scripts/evaluate/plot_diarization.py --input-manifest .data/predicted/segments.json --output-file .data/diarization_plot.png

# Metrics comparison plot
uv run python scripts/evaluate/plot_metrics.py --metrics-file .data/model_a_der.json --metrics-file .data/model_b_der.json --output-file .data/metrics_comparison.png
```

### Dataset utilities

```bash
# Directory indexing
uv run python scripts/dataset/index.py --input-dir .data/audio --output-manifest .data/manifest.json --tag raw

# Manifest filtering
uv run python scripts/dataset/filter.py --input-manifest .data/manifest.json --output-manifest .data/filtered.json --min-duration 1.0 --max-duration 15.0

# Export manifest to JSONL / CSV
uv run python scripts/dataset/export.py --input-manifest .data/filtered.json --output-file .data/dataset.jsonl --format jsonl

# Create dataset bundle ZIP
uv run python scripts/dataset/bundle.py --input-manifest .data/filtered.json --output-file .data/bundle.zip
```

## File and path contract

- `--input-file` wins over `--input-dir`. For single-output operations,
  `--output-file` wins over `--output-dir` and is the exact destination.
  Directory-only input with `--output-file` is rejected.
- Diarizers accept `--output-dir`, not `--output-file`. Read-only `info.py`
  emits JSON Lines to stdout instead of writing an artifact file.
- Defaults are repository-relative `.data/<operation>/<model>/out` and `work`;
  non-model operations omit the model directory. User paths resolve against the
  caller's working directory.
- Directory input is snapshotted recursively in sorted order. The output
  subtree is excluded; identical input/output roots and colliding derived names
  are rejected before processing. Subdirectories are mirrored.
- Audio names and directories (`safe_name`) strictly allow `[a-zA-Z0-9_-]`. Spaces
  and dots inside stems/directories are converted to `-`, Unicode diacritics are
  transliterated to ASCII (e.g. Vietnamese `đ/Đ` -> `d/D`), and special characters
  are stripped to prevent shell argument-splitting and path issues.
- Files are processed sequentially. A model is loaded once per invocation.
  Individual failures do not stop later files. Batch summaries go to stderr;
  successful output paths go to stdout; any file failure yields nonzero status.
- Outputs cannot overwrite audio inputs in place. Matching source/settings and intact
  output hashes allow skips. Missing or damaged output with matching metadata
  is retried. Unrecognized/conflicting destinations require `--overwrite`.
- Completion metadata is published after audio. An incomplete matching record
  permits retry following an interrupted write. Old unused clips can remain in
  an overwritten segment directory; the manifest is authoritative.

Audio-producing commands write a sibling JSON file (`recording.wav` →
`recording.json`) with `schema_version`, `source`, `operation`, `model`,
`parameters`, and `output`. Local sources include an absolute path and SHA-256;
download sources retain full video title, ID, and URL. Output properties include
sample rate, channels, frames, duration, format, and SHA-256. A verified prior
sidecar's source identity is carried as `source.origin`.

Each diarized source gets `<input-stem>/segments.json`. Its `turns` array contains
`speaker_id`, `start_s`, `end_s`, `start_sample`, `end_sample`, `overlap`,
`overlap_with` (zero-based turn indices), `clip`, `clip_sha256`, and `clip_frames`.
Sample intervals are half-open and refer to `source_sample_rate` in the diarized
input. Millisecond filename labels are rounded display values. Clip paths are
relative to the manifest. Empty turn arrays represent no detected speech.
`complete` is published last. Export resampling does not change source boundaries.

Purity outputs retain the source and explicitly invalidate clip references;
`export_segments.py` renders their revised sample boundaries. Verification JSON
contains provenance plus `verdict`; evaluation JSON contains provenance plus
`metrics`. Neither operation implicitly executes an upstream stage.
