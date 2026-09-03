# API contract

```mermaid
flowchart TD
    CALLER[Pipeline caller or notebook] --> CRAWLER[YtCrawler]
    CRAWLER --> AUDIO[Audio]
    AUDIO --> SEPARATOR[BaseSeparator implementation]
    SEPARATOR --> CLEAN_AUDIO[Audio]
    AUDIO --> DIARIZER[BaseDiarizer implementation]
    DIARIZER --> DIARIZATION[DiarizationResult]
    AUDIO --> OVERLAP_VERIFIER[BaseOverlapVerifier implementation]
    OVERLAP_VERIFIER --> OVERLAP_RESULT[OverlapVerificationResult]
    AUDIO --> VIBEVOICE[VibeVoicePurityVerifier]
    VIBEVOICE --> VIBEVOICE_RESULT[VibeVoicePurityResult]
    AUDIO --> MIXER[AudioMixer]
    CLEAN_AUDIO --> MIXER
    MIXER --> MIX_RESULT[AudioMixResult]

    MANAGED[ManagedModel lifecycle] -. load / unload .-> MANAGED_BACKENDS[BSRoFormer, MelRoFormer]
    MANAGED -. load / unload .-> DIARIZER
    CLEAN_AUDIO --> SAVE[Audio.save_to]
    SAVE --> FILE[Audio file on disk]
```

This document describes what the public methods do, what they accept and
return, their side effects, and the conditions required to call them. The
field-level shape of returned objects is documented in
[`data_contract.md`](data_contract.md).

## 1. Audio ingestion API

### `YtCrawler.ingest(...) -> Audio`

**Defined in:** `src/yt_crawler/YtCrawlerClass.py`

A convenience class method that creates a crawler and delegates to
`download()`.

**Inputs:**

- `link: str` — YouTube URL.
- Optional output/work directories and normalization settings.
  Default channel count is 1 (mono). Default sample rate is 44,100 Hz.
  Pass `sample_rate=None` to keep the source/native rate (format + mono
  conversion still apply).
- Additional crawler options through `**kwargs`, such as cookies or proxy.

**Returns:** One `Audio` instance representing the normalized downloaded file.

**Side effects:** Creates output/work directories, invokes `yt-dlp` and possibly
`ffmpeg`, writes the final audio artifact, and removes the temporary download
session directory.

### `YtCrawler.download(url: str) -> Audio`

Downloads one URL and returns an `Audio` object.

**Behavior contract:**

1. Creates an isolated temporary session directory.
2. Runs `yt-dlp` without downloading a playlist.
3. Reads the generated `.info.json` metadata.
4. Selects an audio file rather than a video artifact.
5. Records `native_sample_rate` from the pre-normalize source (or yt-dlp
   `asr`), then normalizes to the crawler's configured format and channel
   count (default 1 / mono). When `sample_rate` is an int, ffmpeg resamples
   to that rate; when `sample_rate` is `None`, the source rate is kept.
   Saved as `<channel_id-or-name>/<sanitized_title>__<source_id>.<format>`.
6. Records `source_url`, `channel_id`, `channel_name`, and `channel_url` from
   yt-dlp metadata, then generates a companion identity sidecar JSON
   (`<stem>.json`) alongside the audio file.
7. Returns an `Audio` object whose `path` points to the final output.
8. Removes the temporary session directory even if processing fails.

**Raises:** `DownloadError` when `yt-dlp`, metadata parsing, audio discovery,
or `ffmpeg` processing fails.

### `parse_crawl_sample_rate(value, *, default=44100) -> int | None`

Parses a UI/API sample-rate choice. Accepts `"native"` / `0` / missing empty
values (falls back to `default`), or a positive Hz int/string. Returns
`None` for native-rate crawls.

### `YtCrawler.build_command(url: str, target_work_dir: Path) -> list[str]`

Builds, but does not execute, the `yt-dlp` command. The returned value is a
list of command-line tokens, not a data-contract class. The command includes
`--newline` so yt-dlp emits one progress line at a time, uses yt-dlp's
`no-certifi` compatibility option with the system CA bundle when available, and
passes `--postprocessor-args ExtractAudio:-ac N` so extraction matches the
crawler's channel count (default 1). `YtCrawler.cancel()` terminates an active
yt-dlp process group.

## 2. `Audio` API

**Defined in:** `src/utils/AudioClass.py`

`Audio` is a file-backed dataclass. Its fields are documented in
[`data_contract.md`](data_contract.md).

### `Audio.from_file(path, *, source_id=None, title=None, source_url=None, channel_id=None, channel_name=None, channel_url=None, native_sample_rate=None, history=None) -> Audio`

Creates an `Audio` object for an existing file.

**Behavior contract:**

- Resolves `path` to an absolute path.
- If `{stem}.json` exists next to the audio (written by `save_to` /
  `quick_save`), restores video identity, source/channel URL and identity,
  `native_sample_rate`, and `history` from it. Explicit keyword arguments
  override the sidecar.
- Uses the file stem as the default `source_id` and `title` when no sidecar
  is present.
- Detects the extension as `format`.
- Probes WAV files for sample rate, duration, and channel count (these
  probed values win over sidecar snapshots).
- Sets `native_sample_rate` to the sidecar value, else the probed file rate,
  unless the caller passes the original source rate.
- Uses class defaults for metadata that cannot be probed.
- Initializes `history` to the provided sequence, else sidecar history, else
  an empty tuple.

**Raises:** `FileNotFoundError` if the path does not identify a file.

### `Audio.metadata(*, target_sample_rate=None) -> dict`

Returns a serializable snapshot of the `Audio` fields (including `history` as a list). `path` is a string.
When `target_sample_rate` is given, the dict also includes that rate and
`resample_action` (`upscale`, `downscale`, or `keep`).

### `Audio.resample_action(target_sample_rate) -> "upscale" | "downscale" | "keep"`

Compares the current file `sample_rate` with a model's expected rate so the
caller can decide whether to resample. This uses the current file rate, not
`native_sample_rate`. `native_sample_rate` is still useful to detect that the
file was already resampled from the original source.

### `Audio.add_step(step_tag) -> Audio`

Returns a new `Audio` object with the sanitized `step_tag` appended to `history`.

### `Audio.fingerprint -> str`

Returns a formatted string representing the audio's processing fingerprint, combining
the source identifier, step history, and audio specs (`duration_s`, `sample_rate`, `channels`)
using `__` as segment separators.

### `Audio.save_to(dest) -> Audio`

Copies the represented audio file to `dest` and updates the same object's
`path` to the destination. It returns the same `Audio` instance, not a new
independent object. Destinations without a suffix default to `.wav`.

**Side effects:** Creates destination parent directories, copies the file
when the source and destination differ, and writes `{stem}.json` next to
the audio with identity metadata (`source_id`, `title`, `native_sample_rate`,
`history`, and last-known audio specs).

**Raises:** `FileNotFoundError` if the current source file does not exist.

### `Audio.quick_save(output_dir=None, *, name=None, prefix=None, suffix=None, tag=None) -> Audio`

Copies the represented audio file to a quick-save temporary directory (defaulting to
`<project_root>/.data/quick_save/`) as WAV unless `name` includes another extension, with an
informative filename generated from the `Audio` object's `fingerprint` (or explicit
`name`), prints `Quick saved to: <destination>` to standard output, and updates the
same object's `path` to the destination. It returns the same `Audio` instance.

**Side effects:** Creates the destination directory, copies the file when source
and destination differ, writes `{stem}.json` next to the audio, and prints the
destination path to stdout.

**Raises:** `FileNotFoundError` if the current source file does not exist.

### `Audio.write_sidecar() -> Audio`

Writes identity metadata next to `self.path` as `{stem}.json` without modifying or moving the underlying audio file. It returns the same `Audio` instance.

### `Audio.notebook_display(dest=None) -> None`

Optionally copies the audio file to `dest` via `save_to` and displays it using
an interactive IPython audio player. It returns `None` and is intended for notebooks.
`Audio.display` is provided as an alias.

**Raises:** `FileNotFoundError` if the represented audio path does not exist.

### `Audio.show_mel_spectrogram(...) -> None`

Loads the audio, computes a mel spectrogram, and displays it with Matplotlib.
It does not modify the `Audio` object and returns `None`.

**Raises:** `FileNotFoundError` if the represented path no longer exists.

### `repr(audio) -> str`

Returns a human-readable representation containing the source ID, title, path,
sample rate, native sample rate, duration, channel count, format, and history (if non-empty).

## 3. Source-separation API

### Common interface: `BaseSeparator.separate(audio: Audio) -> Audio`

**Defined in:** `src/separation/BaseSeparator.py`

Every separator consumes one `Audio` object and returns one separated `Audio`
object. The returned object has the same nine-field class contract, while its
`path` points to the normalized separated output and its `history` records the separation step.

### Concrete implementations

| Class | Method | Model lifecycle requirement |
|---|---|---|
| `HTDemucs` | `separate(audio) -> Audio` | No explicit `load()` required. |
| `BSRoFormer` | `separate(audio) -> Audio` | Call `load()` first, or use a context manager. |
| `MelRoFormer` | `separate(audio) -> Audio` | Call `load()` first, or use a context manager. |
| `MVSepMDX23` | `separate(audio) -> Audio` | No `ManagedModel` load step; dependencies/checkpoints may be fetched on first use. Requires `onnxruntime-gpu` for CUDA execution; explicit `device="cuda"` raises `MVSepMDX23Error` if CUDA or ONNX CUDA execution provider is unavailable. Indexed devices such as `cuda:1` isolate the upstream subprocess to that physical GPU. |

**Common behavior:**

- Verifies that the input path exists.
- Runs the backend-specific separation process.
- Normalizes the selected stem to the configured sample rate and channel count.
- Probes the output WAV.
- Preserves the input `source_id`, `title`, and `native_sample_rate`.
- Returns a new `Audio` instance with `format="wav"`.

**Common failure behavior:** Each backend raises its own runtime error type
when the input, external command, model, or expected output is unavailable.

`MVSepMDX23` uses a resource-conscious vocal-separation default of one Kim
ONNX model (`single_onnx=True`) and `0.25` overlap. Callers that need the
upstream maximum-quality ensemble can pass `single_onnx=False`,
`overlap_large=0.6`, and `overlap_small=0.5`. Its optional
`progress_callback(message)` receives unbuffered upstream status lines.
Long inputs are processed as bounded 10-minute WAV segments by default and the
selected output stem is concatenated afterward; `max_segment_seconds=None`
disables this behavior.
`cancel()` requests non-blocking cancellation, while `close()` terminates and
reaps any active upstream process group. The cloned CLI is patched to print
`PROGRESS: N%` lines from the upstream percent callback so the web queue can
show a real progress bar when those lines are present.

`HTDemucs` streams Demucs CLI output, exposes `progress_callback(message)` and
`cancel()`, and terminates its process group on cancel. Tqdm percent lines are
forwarded when Demucs prints them.

### `BaseSeparator.close() -> None`

Default no-op resource cleanup method. `BSRoFormer` and `MelRoFormer` provide a
compatibility implementation that delegates to `unload()`. `MVSepMDX23.close()`
cancels and reaps an active CLI subprocess. `HTDemucs.close()` cancels an
active Demucs process group.

## 4. Managed model lifecycle API

**Defined in:** `src/base/model.py`

`ManagedModel` is used by `PyannoteDiarizer`, `SortformerDiarizer`,
`ClusteringDiarizer`, `DiariZenDiarizer`, `ThreeDSpeakerDiarizer`, `BSRoFormer`,
and `MelRoFormer`.

### `is_loaded -> bool`

Read-only property indicating whether the underlying resource is currently
loaded.

### `load() -> None`

Loads the resource once by calling the subclass's `_load()` implementation.
Repeated calls while loaded are no-ops.

### `unload() -> None`

Releases the resource once by calling `_unload()`. Repeated calls while already
unloaded are no-ops.

### Context-manager behavior

```python
with model:
    result = model.some_operation()
```

- `__enter__() -> ManagedModel` calls `load()` and returns the model.
- `__exit__(...) -> None` calls `unload()`, including when leaving because of
  an exception.

Subclasses must implement `_load() -> None` and `_unload() -> None`.

## 5. Speaker-diarization API

### `BaseDiarizer.diarize(audio: Audio) -> DiarizationResult`

**Defined in:** `src/diarization/BaseDiarizer.py`

Abstract interface for converting audio into backend-independent speaker
information. Every backend returns schema 2.0, retains the input file-backed
`Audio` as `source_audio`, and copies optional channel identity into the
result. Results validate source identity, declared speakers, turn bounds, and
overlap state before downstream use.

### `evaluate_diarization(reference_turns, hypothesis_turns, *, duration_s, collar_s=0.0, skip_overlap=False) -> dict`

**Defined in:** `src/diarization/evaluation.py`

Computes diarization error from exact interval boundaries without sampled time
bins. A maximum-weight one-to-one assignment maps hypothesis speakers to
reference speakers before DER, JER, missed speech, false alarm, speaker
confusion, and per-speaker coverage are calculated. `collar_s` excludes that
duration on each side of each reference boundary; `skip_overlap=True` excludes
regions containing multiple active reference speakers. Invalid timestamps or
turns outside the shared positive `duration_s` raise `TypeError` / `ValueError`.

### `clean_speaker_turns(turns, *, min_turn_duration_s=0.5, merge_same_speaker_gap_s=1.0, boundary_collar_s=0.04, jitter_max_duration_s=3.0) -> list[SpeakerTurn]`

**Defined in:** `src/diarization/turn_cleanup.py`

Creates a non-mutating, high-precision output view from canonical raw turns.
It corrects bounded, non-overlapping `A-B-A` label jitter; trims 40 ms from
each side of close different-speaker boundaries; merges consecutive
same-speaker turns across gaps up to 1 second; and removes residual turns
shorter than 0.5 seconds. The thresholds are configurable and must be finite,
non-negative numbers. Existing `overlaps_other_speaker` evidence is retained
through relabeling and merging; cleanup does not claim to remove overlapping
speech.

### `pad_and_merge_intervals(intervals, *, pre_roll_s=0.0, post_roll_s=0.0, start_bound_s=0.0, end_bound_s=None, blocker_intervals=None) -> list[tuple[float, float]]`

**Defined in:** `src/diarization/turn_cleanup.py`

Expands extraction windows by `pre_roll_s` / `post_roll_s`, clamps them to
`start_bound_s` and optional `end_bound_s` (typically source duration), and
merges windows that overlap or touch. Optional `blocker_intervals` are
other-speaker windows; extra before/after stops at those bounds so foreign
speech is not mixed into the cut. Canonical `start_s` / `end_s` values are
not rewritten. Empty or inverted inputs are dropped. Bounds must be finite
numbers; roll values must be non-negative.

### `PyannoteDiarizer(model_id=..., device=..., token=..., num_speakers=..., min_speakers=..., max_speakers=..., batch_size=1)`

**Defined in:** `src/diarization/PyannoteDiarizer.py`

- `model_id`: Hugging Face repository ID. Defaults to `DEFAULT_PYANNOTE_MODEL_ID` (`"pyannote/speaker-diarization-community-1"`).
- `device`: Compute target (`"auto"`, `"cuda"`, `"cpu"`, etc.).
- `token`: Optional Hugging Face authentication token (or reads `HF_TOKEN` from environment).
- `num_speakers`, `min_speakers`, `max_speakers`: Optional speaker-count constraints.
- `batch_size`: Segmentation and embedding inference batch size.

### `PyannoteDiarizer.diarize(audio: Audio, *, num_speakers=None, min_speakers=None, max_speakers=None, hook=None) -> DiarizationResult`

Requires the Pyannote pipeline to be loaded first.

**Behavior contract:**

- Decodes the file with `soundfile` and invokes Pyannote with a float32
  `(channel, time)` waveform tensor plus its sample rate. Downmixes stereo/multi-channel to mono when applicable. This keeps file
  decoding independent of Pyannote's optional `torchcodec` integration.
- Passes `num_speakers`, `min_speakers`, `max_speakers`, and `hook` if specified.
- Extracts speaker turns from both `output.speaker_diarization` (pyannote 3.3/4.0+ and community models) or direct Annotation outputs (pyannote 3.1).
- Skips zero- or negative-duration segments so they cannot fail schema validation.
- Converts backend speaker labels into result-local IDs such as `spk_00`.
- Creates one `Speaker` record for each distinct label.
- Creates one `SpeakerTurn` record for each annotated segment.
- Sets `audio_id` to `audio.source_id`.
- Includes `DiarizationModelInfo` with backend `"pyannote"` and the configured
  model ID.

**Raises:** `RuntimeError` if `load()` has not completed or the underlying
pipeline is unavailable; `ValueError` if the decoded audio is empty. Audio
decoder errors are propagated.

### `PyannoteDiarizer._load() -> None` and `_unload() -> None`

These are lifecycle implementations rather than caller-facing processing
methods:

- `_load()` loads the configured Hugging Face/Pyannote pipeline and places it
  on the requested device.
- `_unload()` releases the pipeline and clears CUDA cache when available.

### `SortformerDiarizer.diarize(audio: Audio, *, enrollment_name=None, enrollment_clips=None) -> DiarizationResult`

Requires the dependencies pinned in `requirements-sortformer.txt` and a
completed `load()`. They are intentionally kept out of the primary `uv.lock`
because NeMo constrains shared packages such as Lightning and Hugging Face Hub;
install them in an isolated worker/environment. The default model artifact is
downloaded from
`nvidia/diar_sortformer_4spk-v1` at revision
`f059506485424eb68a90a7af84c8e63e67f381fd`.

Deployment note: model inference is run on the dedicated model server
(`vsf@vsf-242`), not on the development machine (`tungnl5@VF-TUNGNL5-L`). The
server loads `HF_TOKEN` from the repository-root `.env` when the web process
starts. Hugging Face downloads use `.data/huggingface` as the default cache
(`HF_HOME` can override this location).

The constructor's `batch_size` controls NeMo diarization inference and defaults
to `1`.

**Behavior contract:**

- Inspects and normalizes the actual input file to mono, 16 kHz PCM WAV rather
  than trusting `Audio` metadata.
- Processes at most six minutes per inference call by default, with a one-minute
  overlap between adjacent windows.
- Converts the model's 80 ms, four-channel activity probabilities into speaker
  turns with configurable hysteresis and duration post-processing. Boundary
  defaults are `onset=0.74`, `offset=0.64`, `pad_onset_s=0.12`, and
  `pad_offset_s=0.20`; hysteresis requires `onset >= offset`.
- Aligns speaker slots between windows using activity in the shared region.
- Uses TitaNet embeddings as a fallback when a speaker is absent from the
  overlap, unless speaker similarity is disabled.
- When one enrollment identity is supplied, embeds its clean reference clips
  with the pipeline's own TitaNet encoder before target-audio inference. The
  normalized enrollment centroid seeds global speaker zero during window
  stitching. A matching result speaker carries the profile name in
  `global_speaker_id`; the enrollment anchor is not updated from target-audio
  observations. The enrolled speaker record remains present with zero turns
  when no window clears the identity threshold, making non-detection explicit.
- Resolves duplicate predictions in shared regions at the overlap midpoint,
  preserves simultaneous speakers, sorts turns chronologically, and emits
  result-local IDs such as `spk_00`. Anonymous speaker records are built only
  from speakers that retain at least one turn after overlap ownership clipping.
- May produce more than four speakers in the complete result. The hard limit of
  four distinct speakers applies independently to each inference window.
- Retries with three-minute windows by default when a six-minute inference
  raises an accelerator out-of-memory error.
- Includes `DiarizationModelInfo` with backend `"nemo-sortformer"`, the model
  repository ID, and the pinned revision.

**Device behavior:** `device="auto"` selects CUDA, then MPS, then CPU. MPS is
experimental in NeMo and may use CPU fallbacks when
`PYTORCH_ENABLE_MPS_FALLBACK=1`. An explicitly requested unavailable device
raises `RuntimeError`.

**Raises:** `RuntimeError` when the model is not loaded, optional NeMo support
is unavailable, output is in an unknown format, or the selected device cannot
be initialized. Audio conversion and file errors are propagated.

### `SortformerWorkerDiarizer`

The web applications use `SortformerWorkerDiarizer` from the primary `.venv`.
Its public lifecycle and enrollment-aware `diarize(...) -> DiarizationResult`
contract match `SortformerDiarizer`, but `load()` starts
`.venv-sortformer/bin/python -m src.diarization.sortformer_worker`. The worker
loads NeMo once and is reused across all `diarize()` calls until `unload()` or
`close()`. This keeps NeMo's dependency pins out of the web server process and
allows the unified UI to switch between Sortformer and other utilities without
a server restart. `SORTFORMER_PYTHON` or the `worker_python` constructor option
can point to a non-default isolated interpreter. `cancel()` terminates active
worker inference.

### `SortformerDiarizer._load() -> None` and `_unload() -> None`

- `_load()` downloads the exact `.nemo` artifact (unless `checkpoint_path` is
  supplied), restores it with `strict=False`, and moves it to the selected
  device.
- TitaNet is loaded lazily only when a recording needs multiple windows.
- `_unload()` releases both models and clears available accelerator caches.

### `ClusteringDiarizer.diarize(audio: Audio, *, num_speakers=None) -> DiarizationResult`

**Defined in:** `src/diarization/ClusteringDiarizer.py`

NeMo's cascaded clustering pipeline: MarbleNet voice-activity detection,
multi-scale TitaNet speaker embeddings, then spectral clustering. Requires the
same isolated NeMo environment as Sortformer (`requirements-sortformer.txt`).
Default models are `vad_multilingual_marblenet` and `titanet_large`.
The constructor's `batch_size` controls VAD and embedding extraction and
defaults to `64`.

**Behavior contract:**

- Inspects and normalizes the actual input file to mono, 16 kHz PCM WAV rather
  than trusting `Audio` metadata.
- Writes a one-file NeMo manifest and runs `ClusteringDiarizer.diarize()`.
- Parses predicted RTTM into result-local IDs such as `spk_00`, preserving
  first-seen speaker order.
- Per-call `num_speakers` overrides the constructor value for that inference.
  When an oracle count is set (constructor or call), or when min and max
  speaker bounds are equal via `resolve_speaker_settings`, clustering uses
  that exact count. Otherwise it estimates the count up to
  `max_num_speakers` (default 8).
- Includes `DiarizationModelInfo` with backend `"nemo-clustering"` and a
  `model_id` of `{vad_model}+{speaker_model}`.

**Device behavior:** `device="auto"` selects CUDA, then MPS, then CPU. An
explicitly requested unavailable device raises `RuntimeError`.

**Raises:** `RuntimeError` when the model is not loaded, optional NeMo support
is unavailable, clustering fails without writing RTTM, diarization finishes
without an RTTM under `pred_rttms/`, or the selected device cannot be
initialized. Audio conversion and file errors are propagated.

### `ClusteringWorkerDiarizer`

The web applications use `ClusteringWorkerDiarizer` from the primary `.venv`.
Its public lifecycle and `diarize(audio, *, num_speakers=None) -> DiarizationResult`
contract match `ClusteringDiarizer`, but `load()` starts
`.venv-sortformer/bin/python -m src.diarization.clustering_worker`. The worker
loads MarbleNet and TitaNet once and reuses them until `unload()` or
`close()`. `CLUSTERING_PYTHON`, `SORTFORMER_PYTHON`, or the `worker_python`
constructor option can point to a non-default isolated interpreter.
`cancel()` terminates active worker inference.

### `ClusteringDiarizer._load() -> None` and `_unload() -> None`

- `_load()` constructs NeMo `ClusteringDiarizer` with the configured VAD and
  speaker-embedding models and places them on the selected device.
- `_unload()` releases both models and clears available accelerator caches.

### `DiariZenDiarizer(model_id=..., device="auto", token=None, num_speakers=None, min_speakers=None, max_speakers=None, batch_size=1, ffmpeg_bin="ffmpeg")`

**Defined in:** `src/diarization/DiariZenDiarizer.py`

DiariZen's overlap-aware pipeline: WavLM Large neural segmentation, WeSpeaker
speaker embeddings, and VBx clustering. The default checkpoint is
`BUT-FIT/diarizen-wavlm-large-s80-md-v2`. Its weights are CC BY-NC 4.0 and are
limited to research and other non-commercial use. Dependencies are isolated in
`.venv-diarizen` using `requirements-diarizen.txt`.

`batch_size` controls both segmentation and embedding inference. It defaults to
`1` to keep peak VRAM bounded on 10 GiB GPUs; callers may raise it when the
selected device has sufficient free memory.

**Behavior contract:**

- Normalizes input to mono, 16 kHz PCM WAV before inference.
- Converts the returned Pyannote Annotation into result-local `spk_NN` labels,
  preserving simultaneous turns for overlapping speakers. Clamps turn
  timestamps to the source `Audio.duration_s` so last-frame overshoot cannot
  fail schema 2.0 duration validation.
- Constructor speaker bounds are used unless a corresponding per-call override
  is supplied. An exact speaker count sets both clustering bounds for that call.
- Includes `DiarizationModelInfo` with backend `"diarizen"` and the configured
  Hugging Face model ID.

**Device behavior:** `device="auto"` selects CUDA and then CPU. CUDA and CPU
are supported; other device types raise `RuntimeError`.

**Raises:** `RuntimeError` when the model is not loaded, dependencies are
unavailable, model loading fails, or inference fails; `FileNotFoundError` for a
missing source; `ValueError` for invalid speaker bounds or empty audio.

### `DiariZenWorkerDiarizer`

The web applications and benchmark use `DiariZenWorkerDiarizer` from the
primary `.venv`. It starts
`.venv-diarizen/bin/python -m src.diarization.diarizen_worker`, loads the model
once, and reuses it until `unload()` or `close()`. `DIARIZEN_PYTHON` or the
`worker_python` constructor option can select another interpreter. A requested
`cuda:N` is isolated with `CUDA_VISIBLE_DEVICES` so DiariZen's upstream
`cuda:0` initialization runs on the selected physical GPU. `cancel()`
terminates active worker inference.

### `DiariZenDiarizer._load() -> None` and `_unload() -> None`

- `_load()` authenticates from `token` or `HF_TOKEN`, downloads the configured
  checkpoint through the upstream pipeline, and moves it to the selected
  device.
- `_unload()` releases the pipeline and clears available CUDA allocations.

### `ThreeDSpeakerDiarizer.diarize(audio: Audio, *, num_speakers=None) -> DiarizationResult`

**Defined in:** `src/diarization/ThreeDSpeakerDiarizer.py`

ModelScope [3D-Speaker](https://github.com/modelscope/3D-Speaker) audio-only
pipeline: FSMN voice-activity detection, CAM++ speaker embeddings, then
spectral clustering, with optional pyannote overlap refinement. Requires the
isolated environment pinned in `requirements-3dspeaker.txt`. The 3D-Speaker
repository is shallow-cloned into `.data/3d-speaker` on first `load()` when
missing (override with `THREEDSPEAKER_ROOT`). Model downloads default to
`.data/modelscope`.

The constructor's `batch_size` controls CAM++ embedding batches and, when
overlap refinement is enabled, Pyannote segmentation batches. It defaults to
the upstream value `64`.

**Behavior contract:**

- Normalizes the input file to mono, 16 kHz PCM WAV rather than trusting
  `Audio` metadata.
- Runs `speakerlab.bin.infer_diarization.Diarization3Dspeaker` and converts
  `[[start, end, speaker_id], ...]` segments into result-local IDs such as
  `spk_00`, preserving first-seen speaker order. Frame-quantized segment
  bounds are clamped to the source audio duration.
- Per-call `num_speakers` overrides the constructor value for that inference.
  When an oracle count is set (constructor or call), or when min and max
  speaker bounds are equal via `resolve_speaker_settings`, clustering uses
  that exact count. Otherwise it estimates the count.
- When `include_overlap=True`, enables pyannote `segmentation-3.0` overlap
  refinement and requires `token` or `HF_TOKEN`.
- `chunk_duration_s` / `chunk_step_s` control embedding subsegment window and
  hop (upstream defaults `1.5` / `0.75`). Values must satisfy
  `0 < chunk_step_s <= chunk_duration_s`.
- Includes `DiarizationModelInfo` with backend `"3d-speaker"` and a `model_id`
  of `{vad_model}+{embedding_model}`.

**Device behavior:** `device="auto"` selects CUDA, then MPS, then CPU. An
explicitly requested unavailable device raises `RuntimeError`.

**Raises:** `RuntimeError` when the model is not loaded, speakerlab/ModelScope
support is unavailable, the 3D-Speaker checkout cannot be cloned or is
incomplete, inference fails, or the selected device cannot be initialized.
Audio conversion and file errors are propagated.

### `ThreeDSpeakerWorkerDiarizer`

The web applications use `ThreeDSpeakerWorkerDiarizer` from the primary
`.venv`. Its public lifecycle and
`diarize(audio, *, num_speakers=None) -> DiarizationResult` contract match
`ThreeDSpeakerDiarizer`, but `load()` starts
`.venv-3dspeaker/bin/python -m src.diarization.threed_speaker_worker`. The
worker loads ModelScope/speakerlab models once and reuses them until
`unload()` or `close()`. `THREEDSPEAKER_PYTHON` or the `worker_python`
constructor option can point to a non-default isolated interpreter.
`cancel()` terminates active worker inference.

### `ThreeDSpeakerDiarizer._load() -> None` and `_unload() -> None`

- `_load()` adds the 3D-Speaker checkout to `sys.path` (shallow-cloning
  https://github.com/modelscope/3D-Speaker into `.data/3d-speaker` when
  missing), constructs `Diarization3Dspeaker` with the configured device /
  overlap settings, overrides embedding `chunk(dur, step)` with
  `chunk_duration_s` / `chunk_step_s`, and caches ModelScope weights under
  `.data/modelscope`.
- `_unload()` releases the pipeline and clears available accelerator caches.

### `SpeakerVerifier(*, model_id=..., device=..., token=..., profiles_dir=...)`

**Defined in:** `src/diarization/SpeakerVerifier.py`

Global speaker-profile storage plus compatibility verification-based segment
filtering. Wraps a
pyannote speaker-embedding model (default `DEFAULT_EMBEDDING_MODEL_ID`,
`"pyannote/wespeaker-voxceleb-resnet34-LM"`) behind the `ManagedModel`
lifecycle. Profiles are stored under `profiles_dir` (default
`.data/speaker_profiles/<name>/`) as copied reference clips plus a
`profile.json` manifest; clips are the source of truth and embeddings are
recomputed at scoring time.

### `SpeakerVerifier.enroll(name, clips, *, overwrite=False, channel_id=None, channel_name=None, channel_url=None) -> SpeakerProfile`

Does **not** require the model to be loaded.

**Behavior contract:**

- Sanitizes `name` to a filesystem-safe identifier.
- Copies each clip file into `<profiles_dir>/<name>/clips/clip_NN.<ext>` and
  writes `profile.json` (schema version, name, timestamps, clip names, and
  optional source-channel provenance). Profiles are globally reusable and are
  not restricted to one channel.
- Replaces an existing profile only when `overwrite=True`.

**Raises:** `SpeakerVerifierError` for empty clip lists, invalid names, or an
existing profile without `overwrite`; `FileNotFoundError` for missing clip
files.

### `SpeakerVerifier.load_profile(name) -> SpeakerProfile`, `list_profiles() -> list[str]`, `delete_profile(name) -> None`

Profile management; none of these require the model to be loaded.
`load_profile` and `delete_profile` raise `SpeakerVerifierError` when the
profile or any of its clips is missing.

### `SpeakerVerifier.add_clips(name, clips) -> SpeakerProfile`, `remove_clip(name, clip_name) -> SpeakerProfile`

Append or remove clean source clips without rebuilding the identity. Clips
remain the source of truth; adding or removing one updates the profile
timestamp. Removing the final clip is rejected—delete the speaker profile
instead.

### `SpeakerVerifier.extract_embedding(audio, start_s=None, end_s=None) -> np.ndarray`

Requires `load()` (or a `with` block) first. `audio` is `Audio | Path | str`.

**Behavior contract:**

- Extracts a single L2-normalized 1D embedding vector representing the whole
  audio file, or a specific `[start_s, end_s]` interval.
- Reads only the requested audio frames via soundfile seeking without loading
  the entire audio track into memory.
- Returns a 1D float64 numpy vector normalized to unit length.

**Raises:** `RuntimeError` if not loaded; `FileNotFoundError` for missing audio
files; `SpeakerVerifierError` if interval bounds are empty/invalid or embedding
fails.

### `SpeakerVerifier.score(audio, result: DiarizationResult, profile) -> TargetSpeakerResult`

Requires `load()` (or a `with` block) first.

**Behavior contract:**

- Computes the profile centroid as the L2-normalized mean of the clip
  embeddings.
- Embeds every diarization turn independently (`Inference.crop` on the file
  slice) and stores the cosine similarity to the centroid.
- Turns shorter than `MIN_EMBEDDING_DURATION_S` (0.15 s) or failing embedding
  get similarity `-1.0` so they can never pass a threshold.
- Sets `overlaps_other_speaker` on turns intersecting a turn of a different
  speaker.
- Returns **all** turns scored; no filtering is applied.
- Includes `DiarizationModelInfo` with backend `"pyannote-embedding"`.
- Copies channel identity from the input `Audio` into the result.

**Raises:** `RuntimeError` if not loaded; `FileNotFoundError` for a missing
audio file; `SpeakerVerifierError` if no profile clip can be embedded.

### `SpeakerVerifier.filter(scored, *, threshold, min_duration_s=1.5, exclude_overlap=True) -> TargetSpeakerResult` (static)

Pure post-processing; no model needed. Returns a new result keeping only
segments with `similarity >= threshold`, duration `>= min_duration_s`, and
(when `exclude_overlap`) no overlap with other speakers' turns. Precision-first:
different thresholds can be tried cheaply on one `score` result. Channel
identity is preserved from the scored result.

### `SpeakerVerifier.verify_purity(source, profile, *, candidates=None, similarity_threshold, min_candidate_duration_s=1.5, max_overlap_duration_s=0.05, window_duration_s=2.0, window_hop_s=0.75) -> list[SpeakerPurityResult]`

Requires `load()` (or a `with` block) first. `source` is
`Audio | DiarizationResult`:

- A schema-2.0 `DiarizationResult` is the canonical input: its embedded
  `source_audio` is the verified file and every turn is one candidate by
  default. `candidates` may select a subset of the result's turns; the
  complete `DiarizationResult` remains the overlap authority. Passing a turn
  that does not belong to the result is rejected.
- A plain `Audio` is verified as a single whole-file candidate attributed to
  the profile's speaker. This path has no overlap authority, so the overlap
  veto cannot fire; the sliding identity windows still apply.

**Behavior contract:**

- Audio/turn identity is structural, never re-checked by string comparison:
  `DiarizationResult` construction already enforces
  `source_audio.source_id == audio_id` and turn bounds against the source
  duration. A result without `source_audio` (legacy schema-1.0 payload) is
  rejected with `ValueError`.
- Rejects candidates shorter than `min_candidate_duration_s` with reason
  `candidate_too_short`.
- Computes the union duration of other-speaker turns intersecting each
  candidate. It rejects when that duration exceeds
  `max_overlap_duration_s`, with reason `overlap_detected`. Callers requiring
  simultaneous-speaker protection must supply results from an overlap-aware
  diarizer.
- Scores the remaining candidate in sliding `window_duration_s` windows at
  `window_hop_s`; the final window is end-anchored so the candidate tail is
  always covered. Candidates shorter than the configured window use one
  whole-candidate window.
- Rejects when any window has cosine similarity below
  `similarity_threshold`, with reason
  `target_similarity_below_threshold`.
- Returns `decision="error"` and reason `embedding_failed` when any required
  identity window cannot be embedded. Error results never pass but remain
  distinguishable from semantic contamination for retry/reporting.
- Short-circuits identity inference after a duration or overlap veto. This
  follows the precision-first rule that one calibrated contamination signal
  is sufficient to reject.
- Stores the decision reason, overlap measurements, every successful window
  score, and embedding model metadata in each `SpeakerPurityResult`. Its
  `passed` property is true only for `decision="pass"`;
  `min_target_similarity` is derived from its window evidence. Callers should
  persist the call thresholds alongside batch results.

The current embedding backend remains
`pyannote/wespeaker-voxceleb-resnet34-LM`. Enrollment clips are still the
model-independent source of truth, so another verified embedding backend can
consume the same profiles without changing their on-disk schema.

### `SpeakerVerifier._load() -> None` and `_unload() -> None`

- `_load()` loads the pyannote embedding model (`Model.from_pretrained` with
  `token` or `HF_TOKEN`) wrapped in `Inference(window="whole")` and moves it to
  the resolved device.
- `_unload()` releases the inference wrapper and clears CUDA cache when
  available.

### Direct-audio overlap verifiers

**Defined in:** `src/diarization/OverlapVerifier.py`

`BaseOverlapVerifier.verify(audio: Audio) -> OverlapVerificationResult` is the
shared interface. Both implementations read the file at `audio.path`, send the
audio bytes directly to the selected multimodal model, and normalize the
structured answer to:

```python
{"overlap": bool, "reason": str}
```

- `Gemma4OverlapVerifier` calls an OpenAI-compatible Unsloth Studio
  `/v1/chat/completions` endpoint with an `input_audio` content block. WAV and
  MP3 are supported. Constructor values fall back to `UNSLOTH_ENDPOINT`,
  `UNSLOTH_MODEL`, and optional `UNSLOTH_API_KEY` environment variables. When
  `UNSLOTH_ENDPOINT` is unset, the URL uses `UNSLOTH_HOST` and `UNSLOTH_PORT`,
  which default to `localhost` and `8888`.
- `GeminiOverlapVerifier` calls Gemini `generateContent` with inline audio and
  a JSON response schema. It requires `GEMINI_API_KEY`; the model defaults to
  `gemini-3.1-pro-preview` and can be changed with `GEMINI_MODEL` or the
  constructor. The `gemini-flash-lite` factory backend selects the
  audio-capable `gemini-3.1-flash-lite` model using the same verifier and API
  key. Web entrypoints load these values from the repository-root `.env`
  before constructing pipeline components.
- `Gemma4OverlapVerifier.check_ready()` probes Unsloth at `/v1/models` before
  any candidate audio is sent. Unreachable hosts, empty model lists, HTTP
  5xx, and OpenAI-style `error` objects fail as readiness errors instead of
  a guessed overlap answer. Empty assistant content is also an explicit
  error (the model may still be loading).
- `GeminiOverlapVerifier.check_ready()` confirms `GEMINI_API_KEY` is set.
- Both constructors accept `prompt` (defaulting to `OVERLAP_PROMPT`) and
  `max_output_tokens` (default `128`) in addition to `model`, `api_key`, and
  `timeout_s`. The prompt is required to be non-empty and the token limit must
  be a positive integer. The JSON response schema is fixed even when callers
  customize the instruction.

`create_overlap_verifier(config)` selects the backend from a flat mapping:

```python
from src.diarization import create_overlap_verifier

verifier = create_overlap_verifier(
    {
        "backend": "gemma4",  # or "gemini" / "gemini-flash-lite"
        "endpoint": "http://localhost:8888/v1/chat/completions",
        "model": "unsloth/gemma-4-12b-it-GGUF",
        "prompt": "Reject if simultaneous speakers are audible.",
        "max_output_tokens": 128,
    }
)
result = verifier.verify(audio_segment)
```

SonicStudio Speaker Purity is LLM-only. Speaker embeddings are not used on
that tab:

- `GET /api/purity/verifier-status` probes Gemma 4 / Unsloth (or the Gemini
  API-key configuration for Pro or Flash-Lite) and returns
  `{ready, message, models}`. A not-ready Unsloth server is reported instead
  of starting a candidate batch.
- `POST /api/diarization/results/verify` first applies speaker / duration /
  overlap / prior-state filters, then every remaining candidate is cut and
  judged by Gemma 4, Gemini 3.1 Pro, Gemini 3.1 Flash-Lite, or VibeVoice-ASR.
  Embeddings do not run.
- `POST /api/purity/verify` verifies a chosen session/library track with the
  same LLM path. Omitted or empty `turns` means the whole file is one
  candidate. Disabling the verifier is rejected.

If Unsloth is down or still loading, the job fails immediately with that
message. Per-candidate request failures stay visible (`error` plus the
Errors tab). Fail-open still records the error text; it does not hide it.

Duration and diarization-overlap measurements are still recorded on
direct-audio rows but do not veto. A positive overlap result rejects the
candidate with reason `direct_overlap_detected`. Request failures either
become `error` with reason `direct_overlap_verification_failed` (the default
fail-closed policy) or keep the candidate as `pass` (the fail-open policy)
while still storing the error.

The web report records backend, model, endpoint, timeout, token budget,
prompt, and failure policy but never returns the API key.

When the mapping omits `backend`, selection falls back to
`OVERLAP_VERIFIER`. Callers compose this verification step where needed; it is
not automatically chained to crawling, separation, or diarization.

### `VibeVoicePurityVerifier`

**Defined in:** `src/diarization/VibeVoicePurityVerifier.py`

VibeVoice-ASR itself is the speaker-purity verifier. It runs the **whole**
candidate file through Transformers-native `microsoft/VibeVoice-ASR-HF`
(`transformers>=5.3.0`), or a selective bitsandbytes checkpoint from
`VIBEVOICE_MODEL_CHOICES`, ignores the transcript, and classifies from
speaker-count plus per-speaker duration:

- exactly one speaker → `pass` (`single_speaker`)
- a second speaker whose total duration is `>= min_secondary_speech_s`
  (default `0.25`) → `reject` (`multiple_speakers`)
- empty output, missing speaker labels, a tiny secondary turn, or an
  inference exception → `uncertain`

This answers “does this clip contain more than one speaker?”, not “is this
the enrolled identity?”. Do not slice the candidate into sub-second windows
first; the model uses the full clip as context.

```python
from src.diarization import VibeVoicePurityVerifier

with VibeVoicePurityVerifier(device="cuda") as verifier:
    result = verifier.verify(candidate_audio)
# result.decision in {"pass", "reject", "uncertain"}
```

- `verify(audio: Audio) -> VibeVoicePurityResult` requires `load()` (or a
  `with` block). Missing files raise `FileNotFoundError`. Model/generation
  failures return `uncertain` / `inference_error` so a batch can continue.
- `verify_batch(audios)` groups files into model forward passes according to
  constructor `batch_size` (default `1`) and preserves input order.
- `classify_vibevoice_segments(segments, *, audio_id, ...)` is the pure
  decision function over already-parsed `{start_time, end_time, speaker_id}`
  dicts.
- `_load()` / `_unload()` follow `ManagedModel`. Full-precision CUDA loads use
  `bfloat16` and `.to(device)`. Bitsandbytes INT8 / NF4 checkpoints (catalog:
  `Dubedo/VibeVoice-ASR-HF-INT8`, `Dubedo/VibeVoice-ASR-HF-NF4`) use
  `device_map` on CUDA and are not moved with `.to()`. They require
  `bitsandbytes>=0.48.1` in `.venv-vibevoice`. GGUF, AWQ, BitNet, and standalone
  `microsoft/VibeVoice-ASR` quants are unsupported.
- The model defaults to Transformers `attn_implementation="eager"`, which is
  required because the current VibeVoice-ASR architecture does not implement
  scaled-dot-product attention. If Transformers nevertheless rejects the
  configured attention backend for that reason, loading clears partial state
  and retries once with eager attention forced across every nested backbone.
  Authentication, CUDA, out-of-memory, and other errors are not retried.

The web applications use `VibeVoicePurityWorkerVerifier`, which starts
`.venv-vibevoice/bin/python -m src.diarization.vibevoice_purity_worker`.
Override the interpreter with `VIBEVOICE_PYTHON`. `cuda:N` is isolated with
`CUDA_VISIBLE_DEVICES`. Create that environment only on the model server.
`VibeVoicePurityWorkerVerifier.check_ready()` probes that local interpreter,
the required Transformers class, requested device, and (for catalog quantized
checkpoints) bitsandbytes plus CUDA, without loading weights.
The web readiness endpoint uses this local probe; VibeVoice is not an HTTP
service, and model weights are still loaded only when verification starts.

SonicStudio `overlap_verifier.backend: "vibevoice"` on
`POST /api/diarization/results/verify` and `POST /api/purity/verify` cuts
each candidate and runs this verifier. Embeddings do not run. Serialized
rows attach a `vibevoice` evidence object (speaker count, dominant speaker,
secondary duration, speaker turns) and may use `decision: "uncertain"`.
Worker-process failures follow `failure_policy` (`fail_closed` → `error` /
`vibevoice_verification_failed`; `fail_open` → keep `pass`).
The request's `overlap_verifier.batch_size` selects the VibeVoice model batch
size from `1` to `256`; lower values use less inference VRAM.
`GET /api/purity/config` returns LLM-verifier defaults only (backend, prompt,
timeout, Unsloth/Gemini/VibeVoice settings). It does not return an embedding
model or cosine threshold. VibeVoice defaults include `models` (`id` / `label`
catalog for the Studio checkpoint select). `VIBEVOICE_MODEL` is an optional
default only; the Speaker Purity UI selects full BF16, INT8, or NF4 per run.

### `run_zero_contamination_pipeline(audio: Audio, config: ZeroContaminationConfig, progress_callback=None) -> ZeroContaminationResult`

**Defined in:** `src/diarization/zero_contamination.py`

High-precision speaker diarization pipeline engineered specifically for TTS training data harvesting, where missed speech (false negatives) carries zero penalty and multi-speaker contamination / boundary bleed is strictly unacceptable.

#### Pipeline stages:
1. **Asymmetric Detection & Competitor Tripwires:** Runs the primary model (`Sortformer`, `DiariZen`, or `Pyannote 3.1`) with a strict target onset threshold (`target_onset`, default 0.80) and a paranoid competitor tripwire threshold (`competitor_onset`, default 0.20) that vetoes frames if any secondary speaker is even faintly detected.
2. **Dual-Engine Mutual Consensus:** When `enable_consensus=True`, runs an orthogonal secondary engine (e.g. DiariZen or Pyannote 3.1) and maps speakers using the Hungarian bipartite matching algorithm (`_maximum_weight_assignment`). An interval is kept **if and only if both engines unanimously agree** on the speaker identity and neither model detects overlapping speech.
3. **Boundary & Syllable Integrity Gate (Premature Truncation Prevention):**
   - **Option A: Context-Aware Handoff Guard (`enable_context_collar`):** Only shaves inward margins if another speaker speaks within `handoff_risk_distance_s` (default 0.80s). If the utterance transitions into natural silence, the tail is preserved and gently extended by `silence_tail_buffer_s` (default +0.15s), preventing truncation of Vietnamese syllable codas ($-p, -t, -k, -m, -n, -ng$) and tonal contours.
   - **Option B: Micro-Acoustic Energy & RMS Valley Snapping (`enable_energy_snapping`):** Scans the audio micro-waveform in a `±energy_search_window_s` (default ±150ms) window with 2ms hop. Snaps the boundary timestamp directly to the nearest vocal cord closure valley / zero-crossing, preventing slicing through voiced phonemes.
   - **Option C: Syllable & Word Forced Alignment Lock (`enable_syllable_alignment`):** Employs PyTorch's `torchaudio.pipelines.MMS_FA` (or a remote Whisper ASR endpoint) to lock boundaries to complete word/syllable tokens, strictly forbidding mid-word slicing.
4. **Dense Sliding WeSpeaker Homogeneity Filter:** When `enable_homogeneity=True`, slides short `homogeneity_window_s` (1.0s, hop 0.25s) across candidate turns using `pyannote/wespeaker-voxceleb-resnet34-LM`. Drops any turn where the cosine similarity between sub-windows and the turn centroid dips below `min_homogeneity_similarity` (default 0.75).
5. **In-Loop Foundation Model Verification (Remote Host or Dedicated Secondary GPU):**
   - **Microsoft VibeVoice-ASR:** When `enable_vibevoice=True`, analyzes candidate audio with autoregressive speaker tokens. Drops turn immediately if secondary speech duration exceeds `max_secondary_speech_s` (default 0.0s). Can run on a dedicated device (`vibevoice_device`, e.g. `cuda:1` to prevent OOM with primary diarizer on `cuda:0`) or call a remote HTTP server via `vibevoice_endpoint`.
   - **Gemma 4 Direct Audio:** When `enable_gemma=True`, sends direct audio to Gemma 4 via Unsloth, vLLM, or OpenAI-compatible multimodal endpoint (`gemma_endpoint`, `gemma_api_key`, `gemma_timeout_s`). Drops turn if `overlap == True`.

#### Reusable modular functions:
- `compute_consensus_turns(primary_turns, secondary_turns, audio_duration_s) -> tuple[list[SpeakerTurn], dict[str, str]]`
- `erode_turn_boundaries(turns, collar_s=0.35, min_duration_s=0.80, transition_exclusion_s=0.50) -> list[SpeakerTurn]`
- `apply_context_aware_collar(turns, collar_s=0.35, handoff_risk_s=0.80, silence_tail_s=0.15, min_duration_s=0.80, transition_exclusion_s=0.50, audio_duration_s=None) -> tuple[list[SpeakerTurn], list[dict]]`
- `snap_boundaries_to_acoustic_valleys(audio, turns, search_window_s=0.15, energy_floor_db=-30.0, frame_len_ms=10.0, hop_len_ms=2.0) -> tuple[list[SpeakerTurn], list[dict]]`
- `align_and_lock_syllable_boundaries(audio, turns, aligner_engine="mms_fa", aligner_endpoint=None, aligner_device="auto", token=None) -> tuple[list[SpeakerTurn], list[dict]]`
- `filter_by_embedding_homogeneity(audio, turns, window_s=1.0, hop_s=0.25, min_similarity=0.75, device="auto", token=None) -> tuple[list[SpeakerTurn], list[tuple]]`
- `filter_by_foundation_models(audio, turns, config, progress_callback=None) -> tuple[list[SpeakerTurn], list[tuple]]`

## 6. Benchmark mixing API

### `AudioMixer.mix(...) -> AudioMixResult`

**Defined in:** `src/benchmark/separation/mixer.py`

**Inputs:**

- `speech: Audio`
- `music: Audio`
- `target_smr_db: float`
- `seed: int`
- `output_dir: str | Path`

**Behavior contract:**

1. Loads both input files as floating-point waveforms.
2. Converts both inputs to the configured channel layout.
3. Crops or loops music to the speech duration using the supplied seed.
4. Applies music gain to reach the requested speech-to-music ratio.
5. Applies a common output gain when the peak exceeds the configured ceiling.
6. Writes `speech_reference.wav`, `music_reference.wav`, and `mixture.wav`.
7. Returns an `AudioMixResult` containing three `Audio` objects and one
   `MixingParameters` object.

**Raises:** `ValueError` for invalid mixer settings, effectively silent input,
or an output directory that would overwrite an input audio file.

## 7. Shared audio utility API

These functions support the public pipeline methods but return no pipeline
classes.

| Function | Return | Contract |
|---|---|---|
| `probe_wav(path)` | `tuple[int, float, int]` | `(sample_rate, duration_s, channels)`. |
| `normalize_wav(src, dest, ...)` | `None` | Copies or converts audio using `ffmpeg`. |

`normalize_wav` raises `AudioConvertError` when `ffmpeg` conversion fails.

## 8. Dataclass method note

`Speaker`, `SpeakerTurn`, `DiarizationModelInfo`, `DiarizationResult`,
`ScoredSegment`, `TargetSpeakerResult`, `SpeakerProfile`,
`SpeakerPurityResult`, `VibeVoiceSpeakerTurn`, `VibeVoicePurityResult`,
`BenchmarkDefinition`, `MixingParameters`, `AudioMixResult`, and
`SeparationBenchmarkSample` are dataclasses. `DiarizationResult` additionally
defines canonical `to_dict()` / `from_dict()` round-tripping, atomic
`save()` / `load()` persistence, source/turn validation, and derived summary
properties. `VibeVoicePurityResult` also defines `to_dict()` / `from_dict()`.
Python supplies the remaining standard dataclass methods.

## 9. Web application platforms

The repository provides two specialized web platforms:

### Shared web backend

- **Entrypoint:** `scripts/start_web.py` / `scripts/start_web.sh` (default port
  `8765`).
- **Frontend mounts:** SonicStudio is served at `/studio/`; SonicPipeline is
  served at `/pipeline/`.
- **Health endpoint:** `GET /api/health` reports backend status and both
  frontend mount points.
- **Shutdown:** Stopping the backend cancels every pending and running
  SonicStudio task and SonicPipeline job, kills job child processes
  (`yt-dlp`, `ffmpeg`, Demucs, MVSEP), and does not resume leftover jobs
  after restart.
- **Compatibility:** The former `start_studio.*` and `start_pipeline.*`
  launchers start the same unified backend.

### `src/web_studio/` (SonicStudio API domain and frontend)
- **Role:** Interactive audio workbench for ingest, vocal separation, global
  known-speaker management, enrollment-aware diarization, A/B audition, and
  library browsing.
- **Frontend layout:** Flat `static/index.html` + `app.js` + `style.css` (no
  HTML partial composer). Tabs: Workspace, Separation, Diarization, Annotate &
  Evaluate, Speaker Purity, Audition, and Library.
- **Audio selection:** Separation and diarization selectors group active
  in-memory audio objects with persistent project-library files. Selecting a
  library file loads it into the active registry before processing. Repeated
  loads of the same resolved file path reuse its existing audio ID.
- **Library browsing:** `GET /api/library` scans `benchmarks/`, `data/`,
  `temp/`, and `.data/` (skipping caches such as `huggingface/`, `.cache/`,
  and `work/`) and returns `{files, total, category_counts}`. Each file
  includes `category`, a stable `category_id` (`speech`, `music`, `cuts`,
  `stems`, `verified`, `diarized`, `ingest`, `pipeline`, `uploads`, `temp`,
  `data`, `other`), sidecar metadata, and Pipeline registry tags when
  present. `GET /api/library/stream` and `GET /api/library/download` serve
  one permitted file. `POST /api/library/delete` and
  `POST /api/library/bulk-delete` remove audio plus sidecar JSON and drop
  matching Studio session / Pipeline registry entries.
- **Library loading:** `POST /api/library/load` accepts `{"path": "..."}` and
  returns `audio_id`, `metadata`, and `reused`. `reused` is `true` when that
  resolved path was already present in the active registry. The Sample
  Library modal and Library tab share this catalog; processing selectors
  group library files by `category_id` and load a `lib:` path into the
  session before running.
- **Background task queue:** YouTube ingestion, single-model separation,
  diarization, and multi-model comparison use **per-device FIFO queues**.
  Each GPU (`cuda:0`, `cuda:1`, …) and the `cpu`/`mps` lane have independent
  workers so work for one accelerator never blocks another. Default is
  `1` worker per device; `STUDIO_QUEUE_CONCURRENCY` sets workers-per-device
  (1–4). CPU-only jobs (YouTube crawl) always use the `cpu` lane.
  Long-running studio jobs use **polling** via `GET /api/tasks/{id}` (no
  live-reload SSE).
- **Task endpoints:** `GET /api/tasks` lists tasks and queue counts,
  `GET /api/tasks/{id}` returns one task, and `DELETE /api/tasks/{id}` cancels
  a queued or running task. Running CLI backends (yt-dlp, Demucs, MVSEP,
  Sortformer) are force-stopped. In-process models (BS-RoFormer, Mel-RoFormer,
  Pyannote) are marked cancelled; the current forward pass may finish.
- **Queue progress:** Shared queue items expose `progress` as 0–100 and
  `progress_known`. When a backend prints a detectable percent or fraction, or
  a batch job reports item counts, the bar shows that value. Otherwise the UI
  reports that numeric progress is unavailable instead of faking a stub
  percentage.
- **Shared Queue endpoints (registered once by Studio):** `GET /api/queue/shared`
  aggregates hardware telemetry and active/queued workloads across Studio and
  Pipeline, including a `device_queues` map (`running` / `queued` / `workers`
  per lane). `DELETE /api/queue/shared/{id}` and
  `POST /api/queue/shared/{id}/cancel` cancel a workload in either domain.
- **Telemetry:** `GET /api/telemetry` is owned by the Studio route table and
  returns host/GPU metrics from `hardware_monitor`.
- **Waveform windows:** `GET /api/audio/{id}/waveform` accepts `start_s`,
  `end_s`, and `bins` query parameters. Defaults are the full track and 1200
  bins. Bounds must satisfy `0 <= start_s < end_s <= duration`; `bins` must be
  an integer from 1 through 8192. The JSON response contains `sample_rate`,
  `duration_s`, `total_frames`, `start_frame`, `end_frame`, `start_s`, `end_s`,
  `frame_count`, `channel_count`, `requested_bins`, `bins`, and `channels`.
  Every item in `channels` has equal-length signed `min` and `max` arrays;
  channels are never averaged together and values retain their original
  linear full-scale amplitude and polarity. The server reads bounded chunks
  with `soundfile`, caches only a reusable full-track overview, and computes
  zoomed windows on demand.
- **Spectrogram windows:** `GET /api/audio/{id}/spectrogram` accepts `start_s`,
  `end_s`, `width`, and `height`. Time validation matches the waveform endpoint;
  image dimensions must be 32–4096 pixels wide and 32–2048 pixels high. It
  returns a marginless PNG of the requested window using native sample rate,
  a linear-Hz 0-to-Nyquist scale, and mean STFT power across channels.
- **Known-speaker endpoints (registered once by Studio, shared by both
  frontends):** `GET/POST /api/speaker-profiles` list/create global profiles;
  `GET/DELETE /api/speaker-profiles/{name}` inspect/delete one;
  `POST /api/speaker-profiles/{name}/clips` appends session cuts; and
  `GET/DELETE /api/speaker-profiles/{name}/clips/{clip_name}` auditions/removes
  a stored reference. `POST /api/diarization/run` accepts one optional
  `enrollment_profile`; currently NeMo Sortformer supports this genuine
  pre-inference anchor. Other backends reject enrollment instead of silently
  substituting post-diarization similarity scoring. The legacy
  `POST /api/diarization/target-speaker-score` endpoint remains for Pipeline's
  explicit target-filter jobs, but is not part of Studio's interactive flow.
- **Canonical diarization-result endpoints:**
  `GET /api/diarization/results` lists durable results with source/model
  summaries and verification state;
  `GET /api/diarization/results/{result_id}` returns one complete result and
  re-registers its source audio into the Studio session when the file still
  exists;
  `DELETE /api/diarization/results/{result_id}` removes one persisted result;
  `POST /api/diarization/results/clear` deletes the catalog;
  `GET /api/diarization/results/{result_id}/turns/{turn_index}/audio` lazily
  cuts and streams a turn without registering it;
  `GET /api/audio/{id}/segment?start=&end=` lazily cuts `[start, end)`
  (seconds) of a session audio item without registering a cut and returns the
  WAV as an attachment (optional `filename`), or as an inline audio stream
  when `inline=1` (with `Accept-Ranges` and `Content-Type: audio/wav` for in-browser playback);
  `POST /api/audio/{id}/segments.zip` accepts
  `{segments: [{start, end, filename}], filename}` and returns a zip of those
  cuts after reading the source file once (Studio's Turns Inspector download
  path; at most 2000 segments). These download endpoints are not
  `POST /api/audio/{id}/cut`, which still registers a session cut; and
  `POST /api/diarization/results/verify` queues filtered turns from one or more
  result IDs as one batch. This is the workbench's **Verify All Eligible
  Turns** path. Dropdowns filter the candidate set first. Gemma 4, Gemini, or
  VibeVoice-ASR then decides every remaining candidate; embeddings do not run.
  `GET /api/purity/verifier-status` reports whether Unsloth (or Gemini) is
  ready before a run. If Unsloth is down, the job fails instead of marking
  candidates silently. Reports persist under `.data/diarization/verifications/`.
  `POST /api/purity/verify` verifies a chosen session/library track
  (**Verify chosen audio**) with the same LLM verifier. Omitted or empty
  `turns` means the whole file is one candidate; in-memory diarization
  turns are used when that track is the active timeline.
- **Manual annotation and evaluation endpoints:**
  `GET /api/diarization/annotations` lists durable ground-truth references;
  `GET /api/diarization/annotations/{annotation_id}` returns a full reference
  and re-registers its source audio when the file remains available;
  `POST /api/diarization/annotations` creates or revision-saves a validated
  reference with atomic file replacement. For a new reference it optionally
  accepts `seed_result_id`; the server requires that durable result to match the
  selected source timeline, copies its speakers and raw turns into an editable
  annotation, normalizes same-speaker overlap, and records immutable
  `seed.result_id`, `seed.model`, and `seed.created_at` provenance;
  `DELETE /api/diarization/annotations/{annotation_id}` deletes only the
  reference JSON; and `POST /api/diarization/evaluate` compares selected
  compatible durable model results with the reference. A result is compatible
  when it shares an exact audio fingerprint, an exact resolved path, or the
  same `audio_id` and duration (within 50 ms) without an AudioCutter `cut_*`
  history step. Full-length stems from the annotated YouTube source therefore
  score against the same reference. Evaluation accepts
  `annotation_id`, `result_ids`, `collar_s` (seconds excluded on each side of
  reference boundaries), and `skip_overlap`. It returns ranked DER/JER reports,
  error components, optimal speaker mappings, and per-speaker coverage. Manual
  references live under `.data/diarization/annotations/`; source audio and model
  results are never rewritten by annotation or evaluation.
- **Clean-turn output policy:** `POST /api/diarization/clean-turns` accepts a
  durable `result_id` or an explicit raw `turns` list plus optional cleanup
  `settings`, and returns derived turns and raw/clean count-duration summaries
  without saving over the result. Studio's **Clean Turns** toggle uses this
  view for timeline preview, active-timeline target/manual-purity review, and
  speaker-stem extraction.
  `POST /api/diarization/extract-speaker` and
  `POST /api/diarization/extract-all-speakers` accept `clean_turns` plus the
  same `settings`. They and `POST /api/purity/export-audio` also accept
  `extraction_settings` with two opt-in post-processing flags. Default is
  raw labeled windows (`add_extra` and `stop_at_other_speakers` both false).
  `add_extra` applies `pre_roll_s` / `post_roll_s` (field presets 0.12 and
  0.20 seconds). `stop_at_other_speakers` then clamps that extra to
  neighboring other-speaker turns (optional `blocker_turns`, or other
  speakers in the same `turns` list). Extra is clamped to the source, and
  intervals that overlap after this expansion are merged before concatenated
  or time-aligned export. Registered stems are tagged `turns:clean` or
  `turns:raw`. Canonical turns and RTTM export remain unchanged. Studio
  exposes boundary detection, cleanup, and extraction controls in that
  workflow order; its cleanup collar defaults to zero to avoid removing
  speech at close boundaries. Turn Inspector **Play**, **Download**, and
  **Save Cut** use canonical `start_s`/`end_s` with no export
  post-processing. Purity verification preview clips stay tight for speaker
  identity; purity stem export uses the same two extraction options.
- **Persistence:** Studio's diarization history and result-first verifier load
  the server-side canonical result catalog after refresh or restart. The
  history panel waits for that catalog before auto-restoring the selected
  track. Result JSON lives under `.data/diarization/results/`. The Studio
  session audio registry is persisted to `.data/studio/audio_registry.json`
  and restored on backend startup for files that still exist. Reopening a
  result re-registers its source audio when the file is still present.
  Browser storage contains viewer-only speaker labels/colors and verifier
  preferences, not the authoritative turns or source identity.
- **Experiment tab (Zero-Contamination Diarization):**
  Mounted at `/studio/#tab-experiment` with dedicated controller `static/experiment.js` and styling `static/experiment.css`.
  - `GET /api/experiment/status`: Reports available backends, compute devices, and default configuration parameters.
  - `POST /api/experiment/run`: Enqueues an asynchronous `experiment_zero_contamination` task executing the zero-contamination pipeline. Returns `task_id` with 202 Accepted. Updates report real-time SSE progress across all stages. Saves the durable `DiarizationResult` to `.data/diarization/results/`.
  - `POST /api/experiment/gemma/probe`: Pings the local or remote Unsloth/Gemma 4 endpoint and returns `{ready: bool, message: str, models: list}`.
  - `POST /api/experiment/gemma/test`: Auditions Gemma 4 direct-audio overlap classification live on the active track or workspace selection. Returns `{overlap: bool, reason: str, latency_s: float, tested_duration_s: float}`.
  - UI visualizes the pipeline's attrition funnel (retained speech duration and turn counts through Primary, Consensus, Collar Erosion, WeSpeaker Homogeneity, and Foundation Model gates) and renders an interactive table of surviving guaranteed pure turns with per-turn audio previews and NIST RTTM export.
  - Granular parameter controls provide two-way synchronized numeric input fields (allowing direct keyboard entry and fine decimal precision) alongside smooth range sliders with fine-grained step sizes across all pipeline stages: Target Speaker Onset/Offset thresholds, Competitor Tripwire Veto, Collar Inward Shave, Min Turn Duration, Transition Exclusion Gap, Handoff Risk Distance, Silence Tail Release, RMS Silence Window and Floor, WeSpeaker Cosine Floor and Sliding Window/Hop, Gemma Timeout, and VibeVoice Secondary Speech Tolerance.

### `src/web_pipeline/` (SonicPipeline API domain and frontend)
- **Role:** Large-scale channel-oriented batch engine for high-throughput
  ingestion, task queue orchestration, dataset curation, bulk separation,
  batch diarization, target filtering, benchmark evaluation, and manifests.
- **Channel endpoints:** `GET /api/channels` returns per-channel item, duration,
  separation, diarization, and target-filter coverage. `GET /api/items` accepts
  `channel_id` plus `type`, `stage`, `speaker`, `profile`, `verification`,
  `format`, dataset, tag, duration, and free-text filters. YouTube batch ingest
  groups items into `Channel · <name>`
  collections by default while retaining the video `source_id`.
- **Channel-scoped jobs:** `POST /api/jobs/batch_separation` and
  `POST /api/jobs/batch_diarization` accept optional `channel_id` alongside
  `dataset`; when item IDs are omitted, both filters are applied to resolve the
  batch. Batch diarization additionally accepts `sortformer_onset`,
  `sortformer_offset`, `sortformer_pad_onset_s`, and
  `sortformer_pad_offset_s`. Their Pipeline forms expose these settings and the
  same channel selector.
- **Frontend layout:** Flat `static/index.html` + `app.js` + `style.css`.
- **Portable registry:** Registered source audio and attached stems are owned
  by repository-root `.data/pipeline/`. `register_audio()` and `attach_stem()`
  copy external files into that tree when needed. Registry JSON and generated
  manifests store repository-relative paths and resolve them against the
  current checkout after synchronization.
- **Job progress:** Server-Sent Events on `GET /api/events` (queue/job updates
  plus telemetry heartbeats). Initial hydrate still uses `GET /api/jobs`.
- **Per-GPU queues:** Batch jobs are routed by `params.device` onto independent
  lanes (same model as Studio). Ingest/upload jobs use the `cpu` lane.
  `POST /api/queue/controls` `set_concurrency` sets **workers per GPU**
  (default 1), not a single global worker pool.
- **Target speaker filter job:** `POST /api/jobs/target_speaker_filter`
  (params: `item_ids`/`dataset`/`channel_id`, `profile`, `threshold`, `min_duration_s`,
  `exclude_overlap`, `export_cuts`, `export_pre_roll_s`, `export_post_roll_s`,
  `device`, `hf_token`) scores items with
  attached diarization against an enrolled profile, writes
  `.data/pipeline/target_speaker/<item_id>/target_speaker__<profile>.json` (kept plus all
  scored segments), optionally exports kept segments as padded wav cuts with
  overlapping padded windows merged, and attaches
  a summary (including qualified segment/duration percentages) under both the
  last-result compatibility key `metadata["target_speaker"]` and the
  per-profile map `metadata["target_speakers"]`. Pipeline-owned state is kept
  in namespaced `system_tags`; users edit only `custom_tags`.

**Telemetry payload:** The `gpu` object includes `load_percent`, host-level
`used_vram_mb`, `free_vram_mb`, and `total_vram_mb`, plus the current process's
`allocated_vram_mb` and `reserved_vram_mb`. NVIDIA telemetry also includes
`power_w`, `power_limit_w`, and `power_percent`; each entry in `devices` includes
its CUDA logical `index`, NVIDIA `physical_index`, and stable `uuid`. Logical
CUDA devices are matched to NVIDIA sensor readings by UUID because CUDA and
`nvidia-smi` can enumerate the same GPUs in different orders. GPU load, power,
and host VRAM counters are `null` when the active accelerator cannot provide
them (for example, MPS).
