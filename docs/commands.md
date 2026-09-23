# Environment setup and command cookbook

[← Overview](../README.md) · [Data contract](data_contract.md)

## Invocation

Every public Python command has a same-name Bash launcher. Use
`bash scripts/<group>/<command>.sh ...` (never `bash file.py`). Launchers pick an
existing interpreter, forward arguments unchanged, keep the caller's working
directory, and never install dependencies. `ffmpeg`/`ffprobe` must be on `PATH`.
`HF_HOME` defaults to `.data/huggingface`. Export secrets (`HF_TOKEN`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`, `UNSLOTH_API_KEY`, `VLLM_API_KEY`) as
environment variables; the shared parser logs parsed options without printing
secret-valued options or long prompt contents.

## CLI and file rules

- **Flag shorthands**: All pipeline commands support concise standard shorthands alongside their canonical long names:
  - Input/Output: `-i` or `-if` (`--input-file`), `-id` (`--input-dir`), `-o` or `-of` (`--output-file`), `-od` (`--output-dir`), `-wd` (`--work-dir`)
  - Execution: `-w` or `-ow` (`--overwrite`), `--continue` (Gemini Standard/Flex/Batch), `-c` (`--concurrency`), `-b` or `-bs` (`--batch-size`)
  - Audio/DSP: `-sr` (`--sample-rate`), `-ch` (`--channels`), `-s` (`--start`), `-e` (`--end`), `-min` (`--min-duration-s`), `-max` (`--max-duration-s`)
  - Manifests: `-im` (`--input-manifest`), `-om` (`--output-manifest`), `-sm` (`--secondary-manifest`), `-rm` (`--reference-manifest`)
  - Models/Agents: `-m` (`--model` / `--model-id`), `-d` (`--device`), `-p` or `-pf` (`--prompt-file`), `-spf` (`--system-prompt-file`), `-mt` (`--max-tokens` / `--max-new-tokens`), `-t` (`--temperature`) and `-tp` (`--top-p`) on local/endpoint backends only (Gemini 3 ignores sampling parameters, so its commands do not accept them), `-ep` (`--endpoint`), `-ap` (`--adapter-path`), `-v` (`--verbose`)
  - Ingest/Export: `-u` (`--url`), `-uf` (`--url-file`), `-sf` (`--source-file`), `-n` or `-l` (`--limit` / `--max-items`), `-f` (`--format`), `-t` (`--tag`)
- `--input-file` takes precedence over `--input-dir`; `--output-file` is an exact
  destination and requires a single input. Relative paths use the caller's working
  directory.
- Downloads are named `<video_id>_<title>-<sample_rate>.wav` (title sanitized and cut at a word boundary to 40 characters). Other names are
  sanitized to `[a-zA-Z0-9_-]`; single-video outputs live under
  `.data/s1-download/<audio-family>/`. Playlist and channel downloads resolve the
  remote collection name and group files under
  `.data/s1-download/<playlist-or-channel>/` (or under
  `<output-dir>/<playlist-or-channel>/` when `--output-dir` is supplied).
  Every download directory also gets an `index.jsonl` (one line per completed
  download: id, title, channel, upload date, duration, path, hash), rebuilt after
  each download; `s1-download/index.sh` rebuilds them for an existing tree.
- Audio outputs have an atomic sibling `.json` sidecar. Matching outputs with a
  valid hash are cached; conflicting outputs require `--overwrite`. Audio is never
  modified in place.
- Commands consuming a segment manifest verify `source.sha256`; an explicit
  `--input-file` override must preserve the original timeline.
- `--concurrency` and `--batch-size` default to 1. Download commands serialize
  YouTube HTTP regardless of `--concurrency` and abort a run that stays
  rate-limited. Progress/configuration goes to stderr, successful artifact paths
  go to stdout, and any item failure makes the command exit nonzero after
  remaining items finish.
- Every batch reports start, per-item start/completion, and final counts on
  stderr. Agent commands add response length, latency, token usage, and item
  cost when the provider reports it. Verifiers add the parsed decision or stable
  failure stage/code. The final `TOTAL_COST` line reports cumulative estimated
  USD cost for priced providers such as Gemini; local models and endpoints that
  do not expose a rate card report pricing as unavailable. These logs never
  replace the artifact paths written to stdout.
- Run any launcher with `-h` for its authoritative flags and defaults. JSON schemas
  and artifact layouts are defined in [data_contract.md](data_contract.md).

| Launchers | Default venv (fallbacks) | Override variable |
|---|---|---|
| `s1-download/*.sh` | `.venvs/download` (`.venv-download`) | `DOWNLOAD_PYTHON` |
| `audio/*.sh`, `s5-export/*.sh` (handoff exception below), `evaluate/*.sh`, `mix/mix.sh`, `speaker/{enroll,filter}.sh`, `purity/{consensus,cleanup,merge,collar,snap,segment}.sh`, `s4-agent/verifier/{plot_verifier_analysis,analysis,analyze,compare,evaluate_verifier,scaffold_experiment}.sh` | `.venvs/audio` (`.venv-audio`, `.venvs/main`, `.venv`) | `AUDIO_PYTHON` |
| `s2-separate/*.sh` | `.venvs/separation` (`.venv-separation`, `.venvs/main`, `.venv`) | `SEPARATION_PYTHON` |
| `s3-diarize/{pyannote,pyannote_31,pyannote_community1}.sh`, `speaker/{score,purity}.sh` | `.venvs/pyannote` (`.venv-pyannote`, `.venvs/main`, `.venv`) | `DIARIZATION_PYTHON` |
| `s3-diarize/{sortformer,clustering}.sh` | `.venvs/sortformer` (`.venv-sortformer`) | `DIARIZATION_PYTHON` |
| `s3-diarize/threed_speaker.sh` | `.venvs/3dspeaker` (`.venv-3dspeaker`) | `DIARIZATION_PYTHON` |
| `s3-diarize/diarizen.sh` | `.venvs/diarizen` (`.venv-diarizen`) | `DIARIZATION_PYTHON` |
| `purity/align.sh` | `.venvs/align` (`.venv-align`, `.venvs/main`, `.venv`) | `ALIGNMENT_PYTHON` |
| `asr/*.sh` | `.venvs/vibevoice` (`.venv-vibevoice`) | `ASR_PYTHON`, `VIBEVOICE_PYTHON` |
| `s4-agent/{gemini,endpoint,hf}.sh`, `s4-agent/verifier/{gemini,endpoint,hf,unsloth}.sh` | `.venvs/verify` (`.venv-verify`, `.venvs/main`, `.venv`) | `VERIFIER_PYTHON` |
| `s4-agent/verifier/vllm.sh` | `.venvs/vllm` (`.venv-vllm`, `.venvs/verify`, `.venvs/main`) | `VLLM_PYTHON`, then `VERIFIER_PYTHON` |
| `s4-agent/verifier/moss.sh` | `.venvs/moss` (`.venv-moss`, `.venvs/verify`, `.venvs/main`) | `VERIFIER_PYTHON` |
| `s4-agent/verifier/minicpm.sh` | `.venvs/minicpmo` (`.venv-minicpmo`) | `VERIFIER_PYTHON` |
| `s4-agent/verifier/kimi.sh` | `.venvs/kimi` (`.venv-kimi`) | `VERIFIER_PYTHON` |
| `s4-agent/verifier/vibevoice.sh` | `.venvs/vibevoice` (`.venv-vibevoice`) | `VERIFIER_PYTHON` |

## Provisioning

`envs/setup_worker_envs.sh` detects AMD ROCm / NVIDIA CUDA / CPU and provisions
one venv per requirements file under `envs/`. `scripts/setup_*.sh` are thin
wrappers to the same scripts. The `download` target is CPU-only and also
provisions project-local JavaScript runtimes for yt-dlp.

```bash
./envs/setup_worker_envs.sh all        # core + workers
./envs/setup_worker_envs.sh core       # download, audio, separation, pyannote, verify, align
./envs/setup_worker_envs.sh workers    # sortformer, 3dspeaker, vibevoice, diarizen, minicpmo, kimi
./envs/setup_worker_envs.sh download   # YouTube + JavaScript runtime environment
./envs/setup_worker_envs.sh <target>   # one env; add --force to recreate
./envs/setup_worker_envs.sh status     # health + accelerator report
```

| Target | venv | Python | Contents |
|---|---|---|---|
| `download` | `.venvs/download` | 3.13 | current yt-dlp[default], Deno, Node/npm, Bun, QuickJS |
| `audio` | `.venvs/audio` | 3.13 | audio tools, s5-export, evaluate, mix, purity (non-ASR), analysis |
| `separation` | `.venvs/separation` | 3.13 | Demucs, BS-RoFormer, Mel-RoFormer, MVSEP-MDX23 |
| `pyannote` | `.venvs/pyannote` | 3.13 | Pyannote 3.1 / Community-1, speaker scoring and purity |
| `verify` | `.venvs/verify` | 3.13 | Gemini, OpenAI-compatible endpoints, HF Gemma, Unsloth |
| `align` | `.venvs/align` | 3.13 | whisper-timestamped word alignment |
| `sortformer` | `.venvs/sortformer` | 3.13 | NeMo Sortformer and clustering diarizers |
| `3dspeaker` | `.venvs/3dspeaker` | 3.13 | ModelScope 3D-Speaker |
| `vibevoice` | `.venvs/vibevoice` | 3.13 | VibeVoice-ASR transcription, PhoWhisper/whisper-timestamped alignment, speaker-count verifier |
| `diarizen` | `.venvs/diarizen` | 3.10 | DiariZen WavLM |
| `minicpmo` | `.venvs/minicpmo` | 3.11 | MiniCPM-o (also `envs/setup_minicpmo_env.sh [--clean]`) |
| `kimi` | `.venvs/kimi` | 3.11 | Kimi-Audio (also `envs/setup_kimi_env.sh [--clean]`, submodule + FlashAttention) |

Manual equivalent (repeat per environment; pick the torch index for your driver,
e.g. `--index-url https://download.pytorch.org/whl/cu128`):

```bash
./scripts/setup_download_env.sh       # Python, current yt-dlp, and local JavaScript runtimes

uv venv --python 3.13 .venvs/audio
uv pip install --python .venvs/audio/bin/python -r envs/requirements-audio.txt
```

## Cookbook

Paths are relative to the current working directory. Omit `--output-dir` to use
the defaults described above. Bulk playlist/channel downloads always add their
resolved, sanitized collection name below the output root.

### Download

```bash
bash scripts/s1-download/youtube.sh  --url 'https://www.youtube.com/watch?v=VIDEO' --output-dir .data/downloads
bash scripts/s1-download/youtube.sh  --url-file urls.txt                                # one URL/ID per line
bash scripts/s1-download/playlist.sh --url 'https://www.youtube.com/playlist?list=PL' --limit 5
bash scripts/s1-download/playlist.sh --url-file playlists.txt --limit 5                  # one playlist URL per line
bash scripts/s1-download/channel.sh  --url 'https://www.youtube.com/@CHANNEL/videos' --limit 10
bash scripts/s1-download/channel.sh  --url-file channels.txt --limit 10                  # one /videos URL per line
bash scripts/s1-download/crawl.sh --source-file scripts/s1-download/sources/vi_en_codeswitch.json --metadata-only
bash scripts/s1-download/crawl.sh --source-file scripts/s1-download/sources/vi_en_codeswitch.json
bash scripts/s1-download/index.sh                                                       # rebuild .data/s1-download/**/index.jsonl
```

Download commands use `.venvs/download` by default. Provisioning updates
`yt-dlp[default]` and installs Deno, Node/npm, Bun, and QuickJS into that
environment; Deno is available to yt-dlp automatically, while the other
runtimes remain available for explicit yt-dlp runtime selection. Bun is pinned
to the current yt-dlp-compatible 1.3.14 release. Download
commands pace YouTube by default: 1.5s between metadata requests,
5–15s of jitter before each media download, a 2M download cap, and serialized
yt-dlp access even when `--concurrency` is raised (concurrency only parallelizes
convert/publish). 429 and bot-check errors cool down for 5–40 minutes and retry
the item; three consecutive items that stay throttled abort the run so a block
is not extended. Cached WAVs are skipped, so rerunning the same command resumes.
Keep `--concurrency 1` for a large crawl. Prefer `--metadata-only` on `crawl`
first, then download in a later session, and use `--max-items` / source
`max_items` to spread work. `--cookie-file` is optional; a personal account can
be banned under bulk crawling, so use a throwaway login only if a bot-check
forces it. `playlist` and `channel` also accept `--url-file` (one URL per line;
blank lines and `#` comments are skipped). `--limit` caps videos from each URL,
not across the file. Override pacing with `--sleep-requests`, `--sleep-interval`,
`--max-sleep-interval`, `--rate-limit 0`, `--throttle-retries`, and
`--throttle-abort-after`.

`crawl` accepts a JSON collection of named playlist URLs, channel tabs, and yt-dlp
search targets. It evaluates global and source-specific title regexes, duration
bounds when the listing exposes duration, and live status; it then deduplicates
accepted entries by YouTube video ID. The included `vi_en_codeswitch.json` source
collection covers the Hana's Lexis, IELTS Thùy Anh, Đặng Trần Tùng, Nguyễn Huyền,
YouPass, and IELTS cùng Daniel sources. Use repeatable `--source '<exact name>'`
to restrict a run and `--max-items` to cap the accepted collection.
Downloaded crawl items are grouped under each source's resolved playlist or
channel name; the resolved name is also recorded in each completed source entry
in the crawl manifest.

Use `--metadata-only` first to inspect the default
`.data/s1-download/<collection>/crawl.json`; omit it to download 48 kHz mono WAVs
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
bash scripts/audio/segment_vad.sh --input-file .data/source.wav --vad-report .data/vad.json --output-file .data/vad-plan/segments.json
bash scripts/audio/export_segments.sh --input-manifest .data/purity/collar/<family>/segments.json --output-dir .data/clips
# default output (when --output-dir is omitted): .data/audio/clips/<family>/
# export also writes plot/timeline.png, plot/timeline_duration.png, plot/timeline_cutoff.png
bash scripts/audio/compare_waveforms.sh    --input-file .data/a.wav --reference-file .data/b.wav --output-file .data/waveforms.png
bash scripts/audio/compare_spectrograms.sh --input-file .data/a.wav --reference-file .data/b.wav --output-file .data/spectrograms.png
```

### Stem separation

```bash
bash scripts/s2-separate/htdemucs_ft.sh --input-dir .data/downloads --output-dir .data/separated --stem vocals
bash scripts/s2-separate/htdemucs.sh    --input-file .data/source.wav --output-file .data/vocals.wav
bash scripts/s2-separate/bs_roformer.sh --input-file .data/source.wav --stem vocals
bash scripts/s2-separate/mel_roformer.sh --input-file .data/source.wav --stem vocals
bash scripts/s2-separate/mvsep_mdx23.sh --input-file .data/source.wav --stem vocals
```

Outputs are named `<stem>_<model>.wav` (`_htdemucs`, `_htdemucs_ft`, `_bs_roformer`,
`_mel_roformer`, `_mvsep_mdx23`) with a sibling `.json`. Mel-RoFormer and
BS-RoFormer take `--device` only; current `melband-roformer-infer` /
`bs-roformer-infer` git builds are Torch (CPU/CUDA) and no longer accept a
`--backend` selector.

### Diarization (all default to 1.5–15 s clips)

The default `--long-segment-strategy vad` recursively splits any turn longer
than `--max-duration-s` at the lowest cached Silero speech-probability frames,
but only when that probability is strictly below `--vad-cut-threshold` (default
`0.1`). If no report is supplied, the command lazily creates or reuses a
content-addressed report under `.data/vad/auto/<source-sha256>.json`. The
default `--vad-device auto` tries `cuda:0` first and retries on CPU if GPU
loading or inference fails. For one input, pass a completed report with
`--vad-report`; for a directory, pass reports named `<audio-stem>.json` through
`--vad-report-dir` to reuse them. If no eligible frame exists, the oversized
interval remains overlong and the final duration filter removes it; the
rejection is recorded in `long_segment_audit`. Use
`--long-segment-strategy drop` to explicitly retain the old behavior of
discarding oversized turns. `--vad-threshold` is an alias for
`--vad-cut-threshold`.

```bash
bash scripts/s3-diarize/sortformer.sh          --input-dir .data/separated --output-dir .data/turns
bash scripts/s3-diarize/sortformer.sh          --input-file x.wav --min-duration-s 1.0 --max-duration-s 30.0
bash scripts/s3-diarize/pyannote_community1.sh --input-file x.wav --num-speakers 2
bash scripts/s3-diarize/pyannote_31.sh         --input-file x.wav
bash scripts/s3-diarize/clustering.sh          --input-file x.wav
bash scripts/s3-diarize/threed_speaker.sh      --input-file x.wav --include-overlap
bash scripts/s3-diarize/diarizen.sh            --input-file x.wav --segmentation-step 0.05 --binarize-onset 0.5 --binarize-offset 0.6
# each run preserves pre-filter turns in segments.raw.json
# each run also writes plot/before_merge/, plot/after_merge/, and plot/ with
# timeline.png, timeline_duration.png, and timeline_cutoff.png in each folder
# (before merge, after merge/before filtering, and after filtering respectively)
# directory runs additionally write collection-level aggregate duration and cutoff plots under _plot/
```

You can still prepare a report before a run that may emit overlong turns. This
remains an independent stage and can be reused by diarization and later
`export_segments`; omitting it uses the automatic cache above:

```bash
bash scripts/evaluate/silero_jit.sh --input-file .data/recording.wav \
  --devices auto --output-file .data/vad/recording.json
bash scripts/s3-diarize/sortformer.sh --input-file .data/recording.wav \
  --output-dir .data/turns --vad-report .data/vad/recording.json
```

For directory input, put one report per source under `.data/vad/` using the
source stem as the filename and pass `--vad-report-dir .data/vad`.

For a directory run, aggregation follows the output layout rather than inferred
video IDs. A flat `--input-dir` writes one aggregate under
`.data/s3-diarize/<model>/<input-dir-name>/_plot/` (or `<output-dir>/_plot/`
when `--output-dir` is set). Nested input subdirectories each get their own
`_plot/`, and the parent output root also gets an aggregate `_plot/` pooling
all nested manifests. `plot/` is a symlink to `_plot/` when unused.

### Merge before duration filtering

All diarization launchers merge fragmented same-speaker turns across silence by
default (fixed `--max-gap-s 1.0`), then export the result in one invocation:

```bash
bash scripts/s3-diarize/sortformer.sh \
  --input-file .data/recording.wav --output-dir .data/turns \
  --merge --max-gap-s 1.0 --silence-threshold-dbfs -40 \
  --min-duration-s 1 --max-duration-s 15
```

Merging is enabled by default (`--merge true`). Dynamic merge adjustment (iteratively
adjusting `max_gap_s` by 0.1 seconds around 1.0s to achieve a 7–10 second mean) is
**disabled by default** (`default: false`). Enable it with `--dynamic-merge` (or `--adjust-mean`).
Use `--merge false` to disable turn merging entirely. Boolean flags accept `true` or `false`;
a bare `--dynamic-merge` or `--adjust-mean` means `true`.

The same flags work with `pyannote.sh`, `pyannote_31.sh`,
`pyannote_community1.sh`, `clustering.sh`, `threed_speaker.sh`, and `diarizen.sh`,
for both `--input-file` and `--input-dir`. Merge is on by default, with
`--max-gap-s 1`, `--silence-threshold-dbfs -40`, and `--frame-ms 20`.
When enabled with `--dynamic-merge` or `--adjust-mean`, mean adjustment targets a
7–10 second mean per input video and changes `max_gap_s` by 0.1 seconds per retry.
Because a larger maximum gap permits more merges, it increases the gap when the mean is
below 7 seconds and decreases it when the mean is above 10 seconds. The adjustment is
bounded by 100 retries and the available same-speaker gaps. The initial merge is retry 0, so the 100-retry
limit permits at most 101 total merge evaluations; most files stop earlier when
the target is reached or the usable gap range is exhausted. Progress output
reports merge turns before the long-segment strategy, post-strategy turns,
VAD cuts/rejections, and duration-filter removals separately.

The example writes processed `segments.json`, `segments.merged.json`, merged clips
that pass the duration filter, and three stage plot sets under
`.data/turns/recording/plot/`. Omit `--output-dir` to use
the normal `.data/s3-diarize/<model>/<family>/` default. `segments.raw.json` retains
the original backend turns before merging and duration filtering. The processed
manifest includes the merge audit, statistics, and mean-adjustment attempts.
Merge settings are part of the cached request: changing merge or mean-adjustment
settings in an existing destination requires `--overwrite`, which reruns
inference and rebuilds clips and plots.

To process an existing raw manifest without rerunning the model, the standalone
merge and export commands remain available. All diarizers preserve
`segments.raw.json` before the clip duration filter; feed that file to merge so
short turns are available. Rerun diarization for older outputs missing either
stage manifest; filtered manifests cannot recover discarded turns.

```bash
bash scripts/purity/merge.sh \
  --input-manifest .data/turns/recording/segments.raw.json \
  --output-manifest .data/purity/merge/recording/segments.json \
  --max-gap-s 1.0 --silence-threshold-dbfs -40
bash scripts/audio/export_segments.sh \
  --input-manifest .data/purity/merge/recording/segments.json \
  --output-dir .data/clips/recording \
  --min-duration-s 1 --max-duration-s 15 \
  --vad-report .data/vad/recording.json
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
merge chain. After merging, the default VAD strategy recursively splits any
individual turn still over the limit before the duration filter, using the
strict `--vad-cut-threshold` gate. An interval with no lower-probability cut is
kept in the merged stage and then removed by the final duration filter. The
split/rejection audit is stored in `long_segment_audit`;
`--long-segment-strategy drop` instead discards those turns. Standalone `purity/merge` retains its unfiltered behavior,
so its output can still contain overlong merged chains until
`audio/export_segments` applies the selected duration strategy.
See [the file contract](data_contract.md#silence-aware-merge) for audit fields.

Merge and export verify the source audio against the input manifest's recorded
SHA-256. If the source was replaced, restore it or explicitly select a waveform
on the same timeline with `--input-file`.

When changing merge settings, use `--overwrite` on merge and export to reuse their
destinations. Export rebuilds even a matching cached result and removes obsolete
clips listed in its previous manifest. For output folders that already contain
untracked WAVs from older exports, choose a fresh `--output-dir`; unrelated files
are preserved. Diarization `--overwrite` also forces inference to rerun.

### ASR and speech transcription (`asr`)

`vibevoice.sh` provides VibeVoice-ASR transcription combined with Whisper cross-attention forced alignment to produce speaker-attributed turns with word-level timestamps. `--align-model` selects the PhoWhisper size (`tiny`, `base`, `small`, `medium`, `large`; default `medium` → `vinai/PhoWhisper-medium`) or a full Hugging Face / OpenAI Whisper identifier. Alignment needs `whisper-timestamped` in `.venvs/vibevoice`. `--quantization` selects the Transformers checkpoint: `none` (default, `microsoft/VibeVoice-ASR-HF` BF16), `int8` (`Dubedo/VibeVoice-ASR-HF-INT8`, ~10–11 GB), or `nf4`/`int4` (`Dubedo/VibeVoice-ASR-HF-NF4`, ~7–8 GB). INT8/NF4 need NVIDIA CUDA and `bitsandbytes>=0.48.1` in `.venvs/vibevoice`; they are not supported on ROCm/HIP. GGUF, AWQ, and BitNet checkpoints are unsupported. `--model-id` still overrides the VibeVoice Hugging Face ID or local directory.

`phowhisper.sh` transcribes with the same PhoWhisper size flag on `--model-id` (default `large`).

Experimental, independent cut planning is available as `audio/segment_tts.sh`.
It consumes nested ASR words plus a completed `evaluate/silero_jit.sh` probability
report. It writes candidate boundaries and rejection audits, not approved TTS
data. See the [testing guide](tts_segmentation_testing.md) for the end-to-end
recipe and [verification and integration](tts_segmentation_verification.md) for
the earlier negative result. A speaker override should use **unfiltered**
`segments.raw.json`; already-exported manifests may have discarded all long turns.

```bash
# Native JIT only; setup_worker_envs downloads the pinned model into ~/.cache/silero-vad.
# cuda:0 selects NVIDIA CUDA or AMD ROCm for the installed torch build.
# SILERO_PYTHON (then ASR_PYTHON) can select an existing torch/torchaudio environment.
bash scripts/evaluate/silero_jit.sh --input-file .data/recording.wav \
  --devices auto --output-file .data/tts/vad.json

# No model inference here: inspect this plan before separately rendering it.
bash scripts/audio/segment_tts.sh --input-manifest .data/asr/recording_vibevoice.json \
  --vad-report .data/tts/vad.json --output-file .data/tts/plan/segments.json

bash scripts/audio/export_segments.sh --input-manifest .data/tts/plan/segments.json \
  --output-dir .data/tts/clips --min-duration-s 1.5 --max-duration-s 15
```

`silero_jit.sh` searches `.venvs/vibevoice`, `.venv-vibevoice`, `.venvs/align`, and
`.venv-align`; worker setup targets also cache the pinned native JIT model at
`~/.cache/silero-vad/silero_vad.jit`. Its JSON retains every frame probability,
source/model hashes, device versions, synchronized timing, repeat differences,
and CPU/GPU threshold disagreements. `--devices auto` (the default) records a
GPU error when necessary and returns success after a CPU retry; explicit device
lists retain strict failure behavior. Use `--devices cpu` on a CPU-only host.
This is a single-recording benchmark, not a GPU multi-stream throughput benchmark.

The planner uses the normal audio launcher (`AUDIO_PYTHON` override). It accepts
one ASR file and optional `--speaker-manifest`; source hashes must match the VAD
report and audio. `--output-file` is exact. Strategy (`experimental-v3`):

1. Every word ending in `.`, `!`, `?`, or `…` is a cut candidate. It is accepted
   only when a Silero pause exists nearby: every frame below `--silence-threshold`
   (default 0.1) for at least `--min-silence-ms` (64). The search window reaches
   `--cut-search-ms` (400) into either neighbouring word, because forced alignment
   stretches word timestamps over pauses; the cut lands at the pause centre, keeping
   between `--collar-ms` and `--max-edge-silence-ms` of pause on each side.
2. Any piece still longer than `--hard-max` (15 s) is split at the accepted
   word-gap pause nearest its middle, repeatedly. Pieces with no such pause are
   rejected as `no_pause_within_hard_max`.
3. Contiguous fragments are merged greedily toward `--target-min`/`--target-max`
   (7–10 s), never past `--hard-max`. `--merge false` keeps the raw fragments.
   Results under `--hard-min` (1.5 s) are rejected as `too_short`.

Runs never join across a speaker change, an aligned gap over `--max-join-gap`,
or an unassigned word. A bounded `--edge-search-ms` VAD search protects outer run
edges, with explicit `*_collar_truncated` flags if a collar cannot fit. The
stderr summary reports how many sentence ends passed the VAD gate; PhoWhisper
punctuates sparsely, so VibeVoice transcripts are expected to raise that number.
Cached probabilities can be reused with different cut settings without rerunning
the VAD benchmark.

For a deliberately simple VAD-only baseline, `audio/segment_vad.sh` requires no
ASR or diarization manifest. Every interval longer than `--max-duration-s` (15 by
default) is split at the lowest Silero speech-probability frame within that whole
interval, provided it is strictly below `--vad-cut-threshold` (default `0.1`).
The rule is then applied independently to each oversized child until all output
intervals satisfy the limit. Equal minima prefer the point nearest the interval
midpoint, then the earlier point. This is not fixed-duration slicing: VAD chooses
boundaries, while the output remains a gapless partition of the full source
timeline.

```bash
bash scripts/audio/segment_vad.sh --input-file .data/recording.wav \
  --vad-report .data/tts/vad.json --vad-device auto \
  --vad-cut-threshold 0.1 \
  --output-file .data/vad-plan/segments.json

bash scripts/audio/export_segments.sh --input-manifest .data/vad-plan/segments.json \
  --output-dir .data/vad-clips --min-duration-s 0 --max-duration-s 15
```

The planner reuses the cached native-JIT probability track and records every cut,
parent interval, recursion depth, VAD frame, and probability. It never invokes a
model or renders clips. An oversized interval without a frame below the threshold
is retained as an explicit rejection in the plan; integrated diarization and
`audio/export_segments` carry it to their final duration filter, which removes
the overlong interval. The final partial VAD frame is ineligible because Silero
inference zero-pads it beyond the real source duration.
See the [recorded baseline run](vad_segmentation_baseline.md) for the observed
tiny-fragment behavior and Gemini 3.8 Flash word-completeness review.

```bash
# Transcribe single file with Whisper word-level alignment (writes <stem>_vibevoice.json and .txt)
bash scripts/asr/vibevoice.sh --input-file .data/recording.wav

# Transcribe directory of audio files with live turn & word output on stderr
bash scripts/asr/vibevoice.sh --input-dir .data/separated --verbose

# Run turn-level transcription without Whisper word-level alignment
bash scripts/asr/vibevoice.sh --input-file .data/recording.wav --no-align-words

# Choose compute device and a smaller PhoWhisper alignment size
bash scripts/asr/vibevoice.sh --input-file .data/recording.wav --device cuda:0 --align-model small --verbose

# Selective INT8 or NF4 4-bit checkpoints (NVIDIA CUDA)
bash scripts/asr/vibevoice.sh --input-file .data/recording.wav --quantization int8
bash scripts/asr/vibevoice.sh --input-file .data/recording.wav --quantization nf4

# PhoWhisper transcription (default vinai/PhoWhisper-large)
bash scripts/asr/phowhisper.sh --input-dir .data/separated --model-id medium --verbose
```

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
bash scripts/s4-agent/gemini.sh   --input-dir .data/clips --prompt-file prompts/vi-prompt-alam.txt
bash scripts/s4-agent/endpoint.sh --input-dir .data/clips --endpoint http://localhost:8000/v1/chat/completions --model google/gemma-4-E2B-it --prompt-file p.txt
bash scripts/s4-agent/hf.sh       --input-dir .data/clips --model-id google/gemma-4-E2B-it --prompt-file p.txt

# Hardened verifiers (default prompt: prompts/full-tags-prompt.md)
bash scripts/s4-agent/verifier/gemini.sh   --input-dir .data/clips --model gemini-3.8-flash --reasoning-effort medium
bash scripts/s4-agent/verifier/gemini.sh   --input-dir .data/clips --inference-mode flex          # synchronous, Batch-priced
bash scripts/s4-agent/verifier/gemini.sh   --input-file clip.wav --inference-mode standard      # skip Batch API, full price
bash scripts/s4-agent/verifier/gemini.sh   --input-dir .data/clips --continue                  # resume unfinished Standard work
bash scripts/s4-agent/verifier/gemini.sh   --input-dir .data/clips --inference-mode batch --continue # reconnect to matching saved Batch jobs
bash scripts/s4-agent/verifier/gemini.sh   --input-dir .data/clips --inference-mode flex --cache-prompt --cache-ttl-s 3600
bash scripts/s4-agent/gemini.sh           --input-dir .data/clips --prompt-file prompts/full-tags-prompt.md --inference-mode batch --cache-prompt  # Batch cache, 25h TTL
bash scripts/s4-agent/verifier/hf.sh       --input-dir .data/clips --model-id google/gemma-4-E2B-it --max-new-tokens 1024
bash scripts/s4-agent/verifier/vllm.sh     --input-dir .data/clips --model google/gemma-4-E2B-it
bash scripts/s4-agent/verifier/unsloth.sh  --input-dir .data/clips --model unsloth/gemma-4-12b-it-GGUF --gguf-variant UD-Q6_K_XL
bash scripts/s4-agent/verifier/endpoint.sh --input-dir .data/clips --endpoint http://localhost:8000/v1/chat/completions
bash scripts/s4-agent/verifier/moss.sh     --input-dir .data/clips
bash scripts/s4-agent/verifier/minicpm.sh  --input-dir .data/clips
bash scripts/s4-agent/verifier/kimi.sh     --input-dir .data/clips
bash scripts/s4-agent/verifier/vibevoice.sh --input-dir .data/clips
bash scripts/s4-agent/verifier/vibevoice.sh --input-dir .data/clips --quantization int8
bash scripts/s4-agent/verifier/vibevoice.sh --input-dir .data/clips --quantization nf4

# Offline analysis and comparison (no model calls)
bash scripts/s4-agent/verifier/plot_verifier_analysis.sh --input-dir .data/s4-agent/verifier/gemini/gemini-3-8-flash/low
bash scripts/s4-agent/verifier/compare.sh  --reference-dir .data/s4-agent/verifier/gemini/gemini-3-8-flash/medium \
                                        --candidates-dir .data/s4-agent/verifier/gemini/gemini-3-8-flash/low
bash scripts/s4-agent/verifier/evaluate_verifier.sh --predictions-dir .data/verdicts/vllm --reference-dir .data/verdicts/gemini --output-file .data/eval.json
```

### Mix and evaluate

```bash
bash scripts/mix/mix.sh --speech speech.wav --music music.wav --smr-db 6 --seed 42 --output-dir .data/mix/example
bash scripts/evaluate/separation.sh  --input-file pred.wav --reference-file .data/mix/example/speech_reference.wav --mixture-file .data/mix/example/mixture.wav --output-file .data/metrics.json
bash scripts/evaluate/diarization.sh --input-manifest pred/segments.json --reference-manifest ref/segments.json --duration 120 --collar 0.25 --output-file .data/der.json
# Audit audio duration loss across diarization, silence, and post-merge filtering:
bash scripts/evaluate/duration_loss.sh --input-manifest .data/s3-diarize/pyannote_community1/<stem>/segments.json
# Aggregate duration loss and yield across an entire diarization folder or collection:
bash scripts/evaluate/duration_loss.sh --input-dir .data/s3-diarize/pyannote_community1 --format table
# Save audit report to JSON or CSV:
bash scripts/evaluate/duration_loss.sh --input-dir .data/s3-diarize/pyannote_community1 --output-file .data/loss_report.json

bash scripts/evaluate/plot_diarization.sh --input-manifest pred/segments.json --reference-manifest ref/segments.json --output-file .data/gantt.png
# writes .data/gantt.png, .data/gantt_duration.png, .data/gantt_cutoff.png
# aggregate every segments.json below a collection (or model root):
bash scripts/evaluate/plot_diarization.sh --input-dir .data/s3-diarize/sortformer/<family> --overwrite
# one collection defaults to <input-dir>/_plot/; an explicit output root uses
# <output-dir>/_plot/ or <output-dir>/<subdir>/_plot/ for nested collections
bash scripts/evaluate/plot_metrics.sh --metrics-file a.json --metrics-file b.json --output-file .data/metrics.png

# ViYT-Diar (isolated evaluate commands; do not import s3-diarize Python)
bash scripts/evaluate/prepare_viyt_diar.sh
bash scripts/evaluate/run_viyt_diar.sh --systems pyannote_community1 --limit 2
bash scripts/evaluate/run_viyt_diar.sh --all
```

`prepare_viyt_diar` downloads the public 100-file `tuanduy1612/ViYT-Diar` `test`
split into `.data/evaluate/viyt-diar/audio/` and writes matching reference
`segments.json` files. `run_viyt_diar` then calls the existing diarizer Bash
launchers as subprocesses (`--merge false`) and scores each clip's
`segments.raw.json` with `evaluate_diarization` at a 0.25 s collar. Systems run
one at a time. Results and figures land under `.data/evaluate/viyt-diar/results/`
and `figures/`. The runner raises the clip-duration window so WAV clip export is
skipped; DER still uses unfiltered raw turns. `--oracle-speakers` passes
`--num-speakers` per clip when the launcher has that flag (Sortformer does not).

Folder mode recursively reads every `segments.json` below `--input-dir` and
groups them by enclosing collection folder (`<collection>/<stem>/segments.json`).
For one collection without `--output-dir`, the final aggregate plots are under
`<input-dir>/_plot/`. Nested collections, or an explicit output root, keep each
collection aggregate inside that tree as `_plot/`, and also write a parent-root
`_plot/` pooling all nested manifests (with a `plot/` symlink when unused).
The final plots are directly in `_plot/`; pre-merge and after-merge aggregates
are under `_plot/before_merge/` and `_plot/after_merge/`.
`timeline_duration.png` is the pooled segment-duration histogram (with count,
mean, and median), and `timeline_cutoff.png` shows pooled remaining segment
count and audio seconds for each minimum-duration cutoff.

### Dataset export (`s5-export`)

```bash
bash scripts/s5-export/index_audio_manifest.sh  --input-dir .data/audio --output-manifest .data/manifest.json --tag raw
bash scripts/s5-export/filter_audio_manifest.sh --input-manifest .data/manifest.json --output-manifest .data/filtered.json --min-duration 1.0 --max-duration 15.0
bash scripts/s5-export/export_manifest_table.sh --input-manifest .data/filtered.json --output-file .data/dataset.jsonl --format jsonl
bash scripts/s5-export/bundle_manifest_audio.sh --input-manifest .data/filtered.json --output-file .data/bundle.zip
```

The four manifest commands were renamed from `index`, `filter`, `export`, and
`bundle`, respectively. Update local invocations to the names above; old launchers
are removed. `export_manifest_table` writes metadata only; `bundle_manifest_audio`
packs the audio named by a manifest. Neither selects verifier outcomes.

For a nontechnical audio/transcript review handoff (only these two paths are needed):

```bash
bash scripts/s5-export/export_verifier_handoff.sh \
  --input-dir .data/s4-agent/verifier/<backend>/<model> \
  --input-manifest .data/clips/<family>/segments.json
```

`--input-manifest` is repeatable and accepts complete indexed audio manifests or
exported segment manifests with clip paths and SHA-256 values. Select the inventory
actually submitted to the verifier. A complete run requires valid pass/reject
results for every expected clip; rejected clips do not make it incomplete.
Missing/failed/uncertain/invalid results or unknown coverage block export unless
`--allow-partial` is explicitly used. Changed/missing passed audio always blocks it.
The manifest selects its clips from the supplied directory before checking settings;
unrelated verdicts are ignored. Different settings are allowed when each selected
clip has one result and are recorded per clip and in the summary. Competing results
for a clip produce a list of folders/settings hashes: narrow `--input-dir` or pass
`--configuration HASH` (unique prefix accepted). The exporter never guesses the
latest/best verdict. Configuration selection may leave missing inputs, which still
require resolution or explicit partial export.

Without `--output-file`, output is `.data/s5-export/<family>_v1.zip`; the name comes
from the first segments manifest's parent (or an indexed manifest's stem, or the
input folder if no manifest is given). Override with `--dataset-name`, `--version`,
or an exact `--output-file .data/s5-export/custom.zip`. Output remains under `.data/`,
outside the run. The resolved path is printed before scanning.

This command needs only an existing Python interpreter, preferring `AUDIO_PYTHON`
or an existing local environment and falling back to `python3`. It does **not**
provision the audio environment or install packages. Duration comes from the
inventory metadata; missing durations are blank and the summary reports known
hours plus the number of clips without duration metadata. Use a new version or explicit
`--overwrite` for an existing destination. No model is called. The ZIP contains
HTML, CSV **and XLSX**, original audio in `passed/` and `needs_attention/`, instructions,
summary, provenance and checksums. Human review starts pending. Missing transcripts
are flagged for manual transcription, not synthesized. See the
[handoff contract](data_contract.md#verifier-review-handoff) for the exact fields
and [review workflow](agent_verifier.md#handing-a-run-to-a-transcript-reviewer).
