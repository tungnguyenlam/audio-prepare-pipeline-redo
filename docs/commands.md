# Environment setup and command cookbook

[← Overview](../README.md) · [Data contract](data_contract.md)

## Invocation

Every public Python command has a same-name Bash launcher. Use
`bash scripts/<group>/<command>.sh ...` (never `bash file.py`). Launchers pick an
existing interpreter, forward arguments unchanged, keep the caller's working
directory, and never install dependencies. `ffmpeg`/`ffprobe` must be on `PATH`.
`HF_HOME` defaults to `.data/huggingface`. Export secrets (`HF_TOKEN`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `UNSLOTH_API_KEY`, `VLLM_API_KEY`) as
environment variables; the shared parser logs parsed options.

## CLI and file rules

- `--input-file` takes precedence over `--input-dir`; `--output-file` is an exact
  destination and requires a single input. Relative paths use the caller's working
  directory.
- Downloads are named `<video_id>_<title10>-<sample_rate>.wav`. Other names are
  sanitized to `[a-zA-Z0-9_-]`; default outputs live under
  `.data/<operation>[/<model>]/<audio-family>/`.
- Audio outputs have an atomic sibling `.json` sidecar. Matching outputs with a
  valid hash are cached; conflicting outputs require `--overwrite`. Audio is never
  modified in place.
- Commands consuming a segment manifest verify `source.sha256`; an explicit
  `--input-file` override must preserve the original timeline.
- `--concurrency` and `--batch-size` default to 1. Progress/configuration goes to
  stderr, successful artifact paths go to stdout, and any item failure makes the
  command exit nonzero after remaining items finish.
- Run any launcher with `-h` for its authoritative flags and defaults. JSON schemas
  and artifact layouts are defined in [data_contract.md](data_contract.md).

| Launchers | Default venv (fallbacks) | Override variable |
|---|---|---|
| `download/*.sh`, `audio/*.sh`, `dataset/*.sh`, `evaluate/*.sh`, `mix/mix.sh`, `speaker/{enroll,filter}.sh`, `purity/{consensus,cleanup,merge,collar,snap,segment}.sh`, `agent/verifier/{analysis,analyze,compare,evaluate_verifier,scaffold_experiment}.sh` | `.venvs/audio` (`.venv-audio`, `.venvs/main`, `.venv`) | `AUDIO_PYTHON` |
| `separate/*.sh` | `.venvs/separation` (`.venv-separation`, `.venvs/main`, `.venv`) | `SEPARATION_PYTHON` |
| `diarize/{pyannote,pyannote_31,pyannote_community1}.sh`, `speaker/{score,purity}.sh` | `.venvs/pyannote` (`.venv-pyannote`, `.venvs/main`, `.venv`) | `DIARIZATION_PYTHON` |
| `diarize/{sortformer,clustering}.sh` | `.venvs/sortformer` (`.venv-sortformer`) | `DIARIZATION_PYTHON` |
| `diarize/threed_speaker.sh` | `.venvs/3dspeaker` (`.venv-3dspeaker`) | `DIARIZATION_PYTHON` |
| `diarize/diarizen.sh` | `.venvs/diarizen` (`.venv-diarizen`) | `DIARIZATION_PYTHON` |
| `purity/align.sh` | `.venvs/align` (`.venv-align`, `.venvs/main`, `.venv`) | `ALIGNMENT_PYTHON` |
| `agent/{gemini,endpoint,hf}.sh`, `agent/verifier/{gemini,endpoint,hf,unsloth}.sh` | `.venvs/verify` (`.venv-verify`, `.venvs/main`, `.venv`) | `VERIFIER_PYTHON` |
| `agent/verifier/vllm.sh` | `.venvs/vllm` (`.venv-vllm`, `.venvs/verify`, `.venvs/main`) | `VLLM_PYTHON`, then `VERIFIER_PYTHON` |
| `agent/verifier/moss.sh` | `.venvs/moss` (`.venv-moss`, `.venvs/verify`, `.venvs/main`) | `VERIFIER_PYTHON` |
| `agent/verifier/minicpm.sh` | `.venvs/minicpmo` (`.venv-minicpmo`) | `VERIFIER_PYTHON` |
| `agent/verifier/kimi.sh` | `.venvs/kimi` (`.venv-kimi`) | `VERIFIER_PYTHON` |
| `agent/verifier/vibevoice.sh` | `.venvs/vibevoice` (`.venv-vibevoice`) | `VERIFIER_PYTHON` |

## Provisioning

`envs/setup_worker_envs.sh` detects AMD ROCm / NVIDIA CUDA / CPU and provisions
one venv per requirements file under `envs/`. `scripts/setup_*.sh` are thin
wrappers to the same scripts.

```bash
./envs/setup_worker_envs.sh all        # core + workers
./envs/setup_worker_envs.sh core       # audio, separation, pyannote, verify, align
./envs/setup_worker_envs.sh workers    # sortformer, 3dspeaker, vibevoice, diarizen, minicpmo, kimi
./envs/setup_worker_envs.sh <target>   # one env; add --force to recreate
./envs/setup_worker_envs.sh status     # health + accelerator report
```

| Target | venv | Python | Contents |
|---|---|---|---|
| `audio` | `.venvs/audio` | 3.13 | download, audio tools, dataset, evaluate, mix, purity (non-ASR), analysis |
| `separation` | `.venvs/separation` | 3.13 | Demucs, BS-RoFormer, Mel-RoFormer, MVSEP-MDX23 |
| `pyannote` | `.venvs/pyannote` | 3.13 | Pyannote 3.1 / Community-1, speaker scoring and purity |
| `verify` | `.venvs/verify` | 3.13 | Gemini, OpenAI-compatible endpoints, HF Gemma, Unsloth |
| `align` | `.venvs/align` | 3.13 | whisper-timestamped word alignment |
| `sortformer` | `.venvs/sortformer` | 3.13 | NeMo Sortformer and clustering diarizers |
| `3dspeaker` | `.venvs/3dspeaker` | 3.13 | ModelScope 3D-Speaker |
| `vibevoice` | `.venvs/vibevoice` | 3.13 | VibeVoice-ASR speaker-count verifier |
| `diarizen` | `.venvs/diarizen` | 3.10 | DiariZen WavLM |
| `minicpmo` | `.venvs/minicpmo` | 3.11 | MiniCPM-o (also `envs/setup_minicpmo_env.sh [--clean]`) |
| `kimi` | `.venvs/kimi` | 3.11 | Kimi-Audio (also `envs/setup_kimi_env.sh [--clean]`, submodule + FlashAttention) |

Manual equivalent (repeat per environment; pick the torch index for your driver,
e.g. `--index-url https://download.pytorch.org/whl/cu128`):

```bash
uv venv --python 3.13 .venvs/audio
uv pip install --python .venvs/audio/bin/python -r envs/requirements-audio.txt
```

## Cookbook

Paths are relative to the current working directory. Omit `--output-dir` to use
the per-family defaults described above.

### Download

```bash
bash scripts/download/youtube.sh  --url 'https://www.youtube.com/watch?v=VIDEO' --output-dir .data/downloads
bash scripts/download/youtube.sh  --url-file urls.txt                                # one URL/ID per line
bash scripts/download/playlist.sh --url 'https://www.youtube.com/playlist?list=PL' --limit 5
bash scripts/download/channel.sh  --url 'https://www.youtube.com/@CHANNEL/videos' --limit 10
bash scripts/download/crawl.sh --source-file scripts/download/sources/vi_en_codeswitch.json --metadata-only
bash scripts/download/crawl.sh --source-file scripts/download/sources/vi_en_codeswitch.json --concurrency 2
```

`crawl` accepts a JSON collection of named playlist URLs, channel tabs, and yt-dlp
search targets. It evaluates global and source-specific title regexes, duration
bounds when the listing exposes duration, and live status; it then deduplicates
accepted entries by YouTube video ID. The included `vi_en_codeswitch.json` source
collection covers the Hana's Lexis, IELTS Thùy Anh, Đặng Trần Tùng, Nguyễn Huyền,
YouPass, and IELTS cùng Daniel sources. Use repeatable `--source '<exact name>'`
to restrict a run and `--max-items` to cap the accepted collection.

Use `--metadata-only` first to inspect the default
`.data/download/<collection>/crawl.json`; omit it to download 48 kHz mono WAVs
through the same cache-aware path as `youtube`, `playlist`, and `channel`.
Title metadata cannot establish natural code-switch density, single-speaker purity,
or the absence of short music/effect inserts. Those properties require reviewing
the crawl manifest and applying the standalone separation, diarization, and audio
verification commands after download.

### Audio utilities

```bash
bash scripts/audio/info.sh    --input-file .data/source.wav                       # JSON Lines to stdout
bash scripts/audio/convert.sh --input-dir .data/input --output-dir .data/converted --sample-rate 48000 --channels 1
bash scripts/audio/cut.sh     --input-file .data/source.wav --start 12.34 --end 18.92 --output-file .data/cut.wav
bash scripts/audio/export_segments.sh --input-manifest .data/purity/collar/<family>/segments.json --output-dir .data/clips
# default output (when --output-dir is omitted): .data/audio/clips/<family>/
# export also writes plot/timeline.png, plot/timeline_duration.png, plot/timeline_cutoff.png
bash scripts/audio/compare_waveforms.sh    --input-file .data/a.wav --reference-file .data/b.wav --output-file .data/waveforms.png
bash scripts/audio/compare_spectrograms.sh --input-file .data/a.wav --reference-file .data/b.wav --output-file .data/spectrograms.png
```

### Stem separation

```bash
bash scripts/separate/htdemucs_ft.sh --input-dir .data/downloads --output-dir .data/separated --stem vocals
bash scripts/separate/htdemucs.sh    --input-file .data/source.wav --output-file .data/vocals.wav
bash scripts/separate/bs_roformer.sh --input-file .data/source.wav --stem vocals
bash scripts/separate/mel_roformer.sh --input-file .data/source.wav --stem vocals
bash scripts/separate/mvsep_mdx23.sh --input-file .data/source.wav --stem vocals
```

Outputs are named `<stem>_<model>.wav` (`_htdemucs`, `_htdemucs_ft`, `_bs_roformer`,
`_mel_roformer`, `_mvsep_mdx23`) with a sibling `.json`.

### Diarization (all default to 1–15 s clips)

```bash
bash scripts/diarize/sortformer.sh          --input-dir .data/separated --output-dir .data/turns
bash scripts/diarize/sortformer.sh          --input-file x.wav --min-duration-s 1.0 --max-duration-s 30.0
bash scripts/diarize/pyannote_community1.sh --input-file x.wav --num-speakers 2
bash scripts/diarize/pyannote_31.sh         --input-file x.wav
bash scripts/diarize/clustering.sh          --input-file x.wav
bash scripts/diarize/threed_speaker.sh      --input-file x.wav --include-overlap
bash scripts/diarize/diarizen.sh            --input-file x.wav --segmentation-step 0.05 --binarize-onset 0.5 --binarize-offset 0.6
# each run preserves pre-filter turns in segments.raw.json
# each run also writes plot/timeline.png, plot/timeline_duration.png, plot/timeline_cutoff.png
# under the output folder next to segments.json
```

### Merge before duration filtering

All diarization launchers merge fragmented same-speaker turns and adjust the
mean clip duration by default, then export the result in one invocation:

```bash
bash scripts/diarize/sortformer.sh \
  --input-file .data/recording.wav --output-dir .data/turns \
  --merge --max-gap-s 1.0 --silence-threshold-dbfs -40 \
  --min-duration-s 1 --max-duration-s 15
```

Use `--merge false` to disable merging or `--adjust-mean false` to keep the
configured `--max-gap-s` unchanged. Boolean flags accept `true` or `false`; a
bare `--merge` or `--adjust-mean` means `true`.

The same flags work with `pyannote.sh`, `pyannote_31.sh`,
`pyannote_community1.sh`, `clustering.sh`, `threed_speaker.sh`, and `diarizen.sh`,
for both `--input-file` and `--input-dir`. Merge is on by default, with
`--max-gap-s 1`, `--silence-threshold-dbfs -40`, and `--frame-ms 20`.
Mean adjustment targets a 7–10 second mean per input video and changes
`max_gap_s` by 0.1 seconds per retry. Because a larger maximum gap permits more
merges, it increases the gap when the mean is below 7 seconds and decreases it
when the mean is above 10 seconds. The adjustment is bounded by 100 retries and
the available same-speaker gaps.

The example writes processed `segments.json`, merged clips that pass the duration
filter, and three plots under `.data/turns/recording/plot/`. Omit `--output-dir` to use
the normal `.data/diarize/<model>/<family>/` default. `segments.raw.json` retains
the original backend turns before merging and duration filtering. The processed
manifest includes the merge audit, statistics, and mean-adjustment attempts.
Merge settings are part of the cached request: changing merge or mean-adjustment
settings in an existing destination requires `--overwrite`, which reruns
inference and rebuilds clips and plots.

To process an existing raw manifest without rerunning the model, the standalone
merge and export commands remain available. All diarizers preserve
`segments.raw.json` before the clip duration filter; feed that file to merge so
short turns are available. Rerun diarization for older outputs missing this file;
filtered manifests cannot recover discarded turns.

```bash
bash scripts/purity/merge.sh \
  --input-manifest .data/turns/recording/segments.raw.json \
  --output-manifest .data/purity/merge/recording/segments.json \
  --max-gap-s 1.0 --silence-threshold-dbfs -40
bash scripts/audio/export_segments.sh \
  --input-manifest .data/purity/merge/recording/segments.json \
  --output-dir .data/clips/recording \
  --min-duration-s 1 --max-duration-s 15
```

Merge preserves speaker labels and requires silence in every channel across the
entire gap, with no other speaker intersecting the combined span. `-40` dBFS is
a starting threshold to calibrate, not a universal silence level. Use
`--input-file` to override the source waveform, keeping its original timeline.
Standalone merge does not apply a duration filter; exported length includes the
preserved pauses. With `--merge`, diarization applies its duration limits after
merging, using the same merge and clip export helpers as the standalone workflow.
The integrated merge also stops before adding a same-speaker turn if the
resulting span would exceed `--max-duration-s`; the rejected turn starts a new
merge chain. Any individual turn that is already over the limit is still
discarded by the subsequent duration filter. Standalone `purity/merge` retains
its unfiltered behavior, so its output can still contain overlong merged
chains until `audio/export_segments` applies its duration filter.
See [the file contract](data_contract.md#silence-aware-merge) for audit fields.

Merge and export verify the source audio against the input manifest's recorded
SHA-256. If the source was replaced, restore it or explicitly select a waveform
on the same timeline with `--input-file`.

When changing merge settings, use `--overwrite` on merge and export to reuse their
destinations. Export rebuilds even a matching cached result and removes obsolete
clips listed in its previous manifest. For output folders that already contain
untracked WAVs from older exports, choose a fresh `--output-dir`; unrelated files
are preserved. Diarization `--overwrite` also forces inference to rerun.

### Target speaker

```bash
bash scripts/speaker/enroll.sh --name khanh_vy --clip ref1.wav --clip ref2.wav
bash scripts/speaker/score.sh  --input-manifest .data/turns/<stem>/segments.json --profile khanh_vy
bash scripts/speaker/filter.sh --input-manifest .data/speaker/score/<family>/segments.json --threshold 0.6 --min-duration-s 1.5 --exclude-overlap
bash scripts/speaker/purity.sh --input-manifest .data/turns/<stem>/segments.json --profile khanh_vy --similarity-threshold 0.6
```

### Purity stages (each reads one manifest, writes a new one)

```bash
bash scripts/purity/consensus.sh --input-manifest primary/segments.json --secondary-manifest secondary/segments.json
bash scripts/purity/cleanup.sh   --input-manifest consensus.json
bash scripts/purity/collar.sh    --input-manifest cleaned.json --collar-s 0.05
bash scripts/purity/snap.sh      --input-manifest collared.json
bash scripts/purity/align.sh     --input-manifest snapped.json --words-file words.json
bash scripts/purity/segment.sh   --input-manifest aligned.json --words-file words.json
bash scripts/audio/export_segments.sh --input-manifest short.json --output-dir .data/clips/final
```

### Agent generation and verification

See [agent_verifier.md](agent_verifier.md) for defaults and artifacts.

```bash
# Raw generation (unparsed text + metadata sidecar)
bash scripts/agent/gemini.sh   --input-dir .data/clips --prompt-file prompts/vi-prompt-alam.txt --temperature 0.2
bash scripts/agent/endpoint.sh --input-dir .data/clips --endpoint http://localhost:8000/v1/chat/completions --model google/gemma-4-E2B-it --prompt-file p.txt
bash scripts/agent/hf.sh       --input-dir .data/clips --model-id google/gemma-4-E2B-it --prompt-file p.txt

# Hardened verifiers (default prompt: prompts/acoustic_defect-3.txt)
bash scripts/agent/verifier/gemini.sh   --input-dir .data/clips --model gemini-3.8-flash --reasoning-effort medium
bash scripts/agent/verifier/gemini.sh   --input-file clip.wav --inference-mode standard      # skip Batch API
bash scripts/agent/verifier/hf.sh       --input-dir .data/clips --model-id google/gemma-4-E2B-it --max-new-tokens 1024
bash scripts/agent/verifier/vllm.sh     --input-dir .data/clips --model google/gemma-4-E2B-it
bash scripts/agent/verifier/unsloth.sh  --input-dir .data/clips --model unsloth/gemma-4-12b-it-GGUF --gguf-variant UD-Q6_K_XL
bash scripts/agent/verifier/endpoint.sh --input-dir .data/clips --endpoint http://localhost:8000/v1/chat/completions
bash scripts/agent/verifier/moss.sh     --input-dir .data/clips
bash scripts/agent/verifier/minicpm.sh  --input-dir .data/clips
bash scripts/agent/verifier/kimi.sh     --input-dir .data/clips
bash scripts/agent/verifier/vibevoice.sh --input-dir .data/clips

# Offline analysis and comparison (no model calls)
bash scripts/agent/verifier/analysis.sh --input-dir .data/agent/verifier/gemini/gemini-3-8-flash/low
bash scripts/agent/verifier/compare.sh  --reference-dir .data/agent/verifier/gemini/gemini-3-8-flash/medium \
                                        --candidates-dir .data/agent/verifier/gemini/gemini-3-8-flash/low
bash scripts/agent/verifier/evaluate_verifier.sh --predictions-dir .data/verdicts/vllm --reference-dir .data/verdicts/gemini --output-file .data/eval.json
```

### Mix and evaluate

```bash
bash scripts/mix/mix.sh --speech speech.wav --music music.wav --smr-db 6 --seed 42 --output-dir .data/mix/example
bash scripts/evaluate/separation.sh  --input-file pred.wav --reference-file .data/mix/example/speech_reference.wav --mixture-file .data/mix/example/mixture.wav --output-file .data/metrics.json
bash scripts/evaluate/diarization.sh --input-manifest pred/segments.json --reference-manifest ref/segments.json --duration 120 --collar 0.25 --output-file .data/der.json
bash scripts/evaluate/plot_diarization.sh --input-manifest pred/segments.json --reference-manifest ref/segments.json --output-file .data/gantt.png
# writes .data/gantt.png, .data/gantt_duration.png, .data/gantt_cutoff.png
# aggregate every segments.json below a family (or model root):
bash scripts/evaluate/plot_diarization.sh --input-dir .data/diarize/sortformer/<family> --overwrite
# defaults to <input-dir>/plot/; use --output-dir to choose another folder
bash scripts/evaluate/plot_metrics.sh --metrics-file a.json --metrics-file b.json --output-file .data/metrics.png
```

Folder mode recursively reads every `segments.json` below `--input-dir` and
combines their final exported turns. `timeline_duration.png` is the pooled
segment-duration histogram (with count, mean, and median); `timeline_cutoff.png`
shows pooled remaining segment count and audio seconds for each minimum-duration
cutoff. Point `--input-dir` at one family for that family, or at a model root to
combine all its families.

### Dataset

```bash
bash scripts/dataset/index.sh  --input-dir .data/audio --output-manifest .data/manifest.json --tag raw
bash scripts/dataset/filter.sh --input-manifest .data/manifest.json --output-manifest .data/filtered.json --min-duration 1.0 --max-duration 15.0
bash scripts/dataset/export.sh --input-manifest .data/filtered.json --output-file .data/dataset.jsonl --format jsonl
bash scripts/dataset/bundle.sh --input-manifest .data/filtered.json --output-file .data/bundle.zip
```
