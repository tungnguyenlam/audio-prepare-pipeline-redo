# 02. Source Separation & Managed Model Lifecycle

[← 01. Audio & Ingestion](01_audio_and_ingestion.md) | [Docs Index](README.md) | [Next: 03. Speaker Diarization →](03_speaker_diarization.md)

---

This module covers the source separation interfaces, supported stem-separation backends (Demucs, BS-RoFormer, Mel-RoFormer, MVSEP), and the shared **`ManagedModel`** lifecycle pattern.

```mermaid
flowchart TD
    AUDIO["Input Audio"] --> SEP["BaseSeparator.separate(audio)"]
    SEP --> STEM["Separated Audio (Vocals / Accompaniment)"]
    STEM --> PROBE["Probe WAV & write sidecar"]
    
    subgraph BACKENDS["Separation Backends"]
        HT["HTDemucs (CLI)"]
        BS["BSRoFormer (ManagedModel)"]
        MEL["MelRoFormer (ManagedModel)"]
        MV["MVSepMDX23 (ONNX CLI)"]
    end
    
    SEP -. delegates to .-> BACKENDS
```

---

## 1. The `BaseSeparator` Interface

**Defined in:** [`src/separation/BaseSeparator.py`](../src/separation/BaseSeparator.py)

Every source separator subclasses `BaseSeparator` and implements the single public entrypoint:

```python
def separate(self, audio: Audio) -> Audio:
```

### Common Contract & Invariants

1. **Path Validation:** Verifies that `audio.path` exists before starting inference.
2. **Identity Preservation:** Automatically copies `source_id`, `title`, and `native_sample_rate` from input to output.
3. **History Step Tracking:** Appends a sanitized transformation tag (e.g. `htdemucs_vocals`, `mvsep_kim_vocal`) to `audio.history`.
4. **Format Normalization:** Normalizes the output stem to the separator's target sample rate (default 44,100 Hz), mono channel layout, and PCM WAV container (`format="wav"`).
5. **Output Verification:** Probes the final WAV file on disk to guarantee valid duration and channel metadata.
6. **Error Granularity:** Raises backend-specific custom exceptions instead of generic `Exception`.

### Resource Teardown (`close()`)

```python
def close(self) -> None:
```

Default no-op teardown method. Backends override `close()` to clean up:
- `BSRoFormer` & `MelRoFormer`: Delegates to `self.unload()`.
- `HTDemucs`: Cancels and reaps any active Demucs process group.
- `MVSepMDX23`: Terminates active CLI subprocesses and cleans up scratch files.

---

## 2. Concrete Separation Backends

| Backend Class | Default Checkpoint | Stem Options | Lifecycle Requirement | Compute Device | Default `out` / `work` dirs | Notes |
|---|---|---|---|---|---|---|
| **`HTDemucs`** | `htdemucs` | `vocals`, `drums`, `bass`, `other` | No explicit `load()`. Subprocess CLI. | `cpu` default; pass `cuda`/`cuda:N` | `.data/demucs/out`, `.data/demucs/work` | Streams CLI stdout; supports `progress_callback` and `cancel()`. |
| **`BSRoFormer`** | `roformer-model-bs-roformer-sw-by-jarredou` | `vocals`, `instrumental` | Requires `load()` or `with model:`. | `auto` default (CUDA-else-CPU); `mps` on Apple Silicon | `.data/bs_roformer/out`, `.data/bs_roformer/work` | Sub-band RoFormer architecture for high-fidelity vocal isolation. Input pre-converted to 44.1 kHz stereo (`model_sample_rate=44100`). |
| **`MelRoFormer`** | `melband-roformer-kim-vocals` | `vocals`, `instrumental` | Requires `load()` or `with model:`. | `auto` default; `mps` on Apple Silicon | `.data/mel_roformer/out`, `.data/mel_roformer/work` | Mel-band RoFormer architecture. Input pre-converted to 44.1 kHz stereo. |
| **`MVSepMDX23`** | Kim ONNX (`single_onnx=True`) | `vocals`, `instrumental` | Standalone CLI runner with ONNX models. | `auto` default; `cuda:N`, `cpu`, `mps` | `.data/mvsep_mdx23/out`, `.data/mvsep_mdx23/work` (repo at `.data/mvsep_mdx23/repo`) | Fast Kim ONNX default; supports full ensemble. Requires `onnxruntime-gpu` + PyTorch CUDA when CUDA is requested. Extra knobs: `large_gpu`, `chunk_size`, `only_vocals`, `repo_dir`, `python_bin`. |

(`BaseSeparator` itself defaults to `device="cpu"`, `output_dir=.data/separated/out`, `work_dir=.data/separated/work`; each backend above overrides the directories.)

---

### `HTDemucs`

**Defined in:** [`src/separation/HTDemucs.py`](../src/separation/HTDemucs.py)

Wraps the Facebook Demucs CLI:
- Exposes `progress_callback(message: str)` receiving live progress lines from Demucs.
- Translates tqdm percent lines to clean progress telemetry.
- `cancel()` terminates the active Demucs process group immediately.
- Raises `DemucsError` on failure.

---

### `BSRoFormer` & `MelRoFormer`

**Defined in:**
- [`src/separation/BSRoFormer.py`](../src/separation/BSRoFormer.py)
- [`src/separation/MelRoFormer.py`](../src/separation/MelRoFormer.py)

Neural stem separators implementing the `ManagedModel` contract:
- Pre-trained checkpoints are loaded on demand via `model.load()` or context manager `with model:`.
- `unload()` closes the inference session and drops the reference (`_unload()`); repeated calls while unloaded are no-ops.
- Raises `BSRoFormerError` or `MelRoFormerError` if invoked prior to `load()` or if inference fails.

---

### `MVSepMDX23`

**Defined in:** [`src/separation/MVSepMDX23.py`](../src/separation/MVSepMDX23.py)

High-performance MDX23 vocal separator runner:
- **Defaults:** Configured with a resource-conscious single Kim ONNX model (`single_onnx=True`, `use_kim_model_1=False` selects Kim checkpoint 2) and `0.25` overlap for high speed and low VRAM footprint.
- **Ensemble Mode:** High-quality vocal separation can be enabled by passing `single_onnx=False`, `overlap_large=0.6`, and `overlap_small=0.5`. `large_gpu=True` opts into higher-VRAM settings; `chunk_size` overrides the CLI chunking; `only_vocals` defaults from `two_stems` (`vocals`/`instrumental` → vocals-only).
- **Chunked Processing:** Long audio files are split into bounded 10-minute WAV chunks (`max_segment_seconds=600`) to prevent OOM errors, with the separated stem concatenated afterward. Pass `None` to process in one pass.
- **Progress Tracking:** The internal CLI prints `PROGRESS: N%` lines, allowing the web task queue to report smooth percentage updates.
- **Subprocess Isolation:** Device selection (e.g. `device="cuda:1"`) is isolated using `CUDA_VISIBLE_DEVICES` so child processes strictly execute on the specified physical GPU.
- Raises `MVSepMDX23Error` if PyTorch CUDA or `onnxruntime-gpu` (`CUDAExecutionProvider`) is missing when CUDA is requested, if `max_segment_seconds <= 0`, for unsupported `two_stems`, or if the subprocess exits non-zero.

---

## 3. Managed Model Lifecycle API (`ManagedModel`)

**Defined in:** [`src/base/model.py`](../src/base/model.py)

Heavy neural models inherit `ManagedModel` to guarantee explicit, predictable memory management. Used by `BSRoFormer`, `MelRoFormer`, `PyannoteDiarizer`, `SortformerDiarizer`, `DiariZenDiarizer`, `ThreeDSpeakerDiarizer`, `ClusteringDiarizer`, `SpeakerVerifier`, and `VibeVoicePurityVerifier`.

### Properties & Methods

#### `is_loaded -> bool`
Read-only boolean property indicating whether model weights and accelerators are actively loaded in memory.

#### `load() -> None`
Executes subclass `_load()` once. Repeated calls while already loaded are no-ops.

#### `unload() -> None`
Releases neural weights and executes subclass `_unload()` once. Repeated calls while unloaded are no-ops. (Cache clearing such as `gc.collect()` / `torch.cuda.empty_cache()` lives in individual `_unload()` implementations that need it — e.g. `SpeakerVerifier` — not in the base class.)

#### Context Manager Protocol
```python
with BSRoFormer(device="cuda:0") as separator:
    vocal_audio = separator.separate(raw_audio)
# Model is automatically unloaded, freeing VRAM immediately
```

- `__enter__()`: Invokes `self.load()` and returns `self`.
- `__exit__()`: Invokes `self.unload()`, even if an unhandled exception occurred during processing.

---

## 4. Usage Example

```python
from src.utils.AudioClass import Audio
from src.separation import MVSepMDX23, BSRoFormer

input_audio = Audio.from_file(".data/yt_crawler/downloads/sample.wav")

# Example A: High-speed separation with MVSepMDX23
separator = MVSepMDX23(device="cuda:0")
vocals = separator.separate(input_audio)
print("Vocals saved at:", vocals.path)
print("History:", vocals.history)

# Example B: Managed BSRoFormer with context manager
with BSRoFormer(device="cuda:0") as bs_sep:
    bs_vocals = bs_sep.separate(input_audio)
    bs_vocals.save_to(".data/bs_roformer/out/bs_vocals.wav")
```
