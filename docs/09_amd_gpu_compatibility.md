# AMD GPU (ROCm) Compatibility & Hardware Execution Guide

This document details hardware compatibility, benchmark test results, and execution tiers for all deep learning models and heavy mathematical operations in the pipeline when running on AMD hardware, specifically tested on the **AMD Radeon RX 9060 XT (16 GB VRAM)**.

---

## 1. System & Test Environment

- **Host Hardware:** AMD Radeon RX 9060 XT (Navi 44, RDNA 4 Architecture)
- **Target ISA:** `gfx1200`
- **VRAM:** 16,384 MB (16 GB GDDR6)
- **Host OS:** Linux CachyOS x86_64 (Kernel 7.2.2)
- **Compute Stack:** ROCm 7.2 / ROCm SDK 10.0.0 (`/opt/rocm/bin/rocm-smi`, `/dev/kfd`, `/dev/dri/renderD128`)
- **PyTorch Stack:** PyTorch `2.13.0+rocm10.0.0` with `amd-torch-device-gfx1200` and `triton-3.8.0+git4cff872c.rocm10.0.0`
- **Device Target in Code:** `device="cuda:0"` (mapped directly via the HIP runtime layer)

---

## 2. Executive Summary

| Category | Total Tested | Run on AMD GPU (`cuda:0`) | Must Fall Back to CPU | Notes |
|---|---|---|---|---|
| **Separation Models** | 4 | **4** (100%) | 0 (PyTorch) / 1 (ONNX) | HTDemucs, BSRoFormer, MelRoFormer run natively on GPU. MDX23 PyTorch runs on GPU; ONNX weights fall back to CPU provider. |
| **Diarization & VAD** | 5 | **4** (80%) | 1 (Clustering backend) | Pyannote neural layers, Silero VAD, WeSpeaker, and Whisper run on GPU. Clustering & Hungarian alignment run on CPU. |
| **Heavy Math / DSP** | 12 | **9** (75%) | 3 (25%) | GEMM, Conv1D/2D, Bi-LSTM, SDPA, rocFFT STFT/iSTFT, Resampling, MFCC run on GPU. Hungarian matching, SciPy/Sklearn clustering, and libsndfile I/O run on CPU. |

---

## 3. Detailed Component Compatibility Matrix

### A. Deep Learning Separation Models

| Component / Model | Backend | GPU Support (`cuda:0`) | CPU Fallback | Performance / Test Result | Failure Mode / Special Requirement |
|---|---|---|---|---|---|
| **HTDemucs** (`HTDemucs.py`) | PyTorch (`demucs`) | ✅ **Yes** | ✅ Supported | **PASS** (38.43s full separation on test audio) | Uses MIOpen convolutions, Bi-LSTM, and cross-domain attention. |
| **BSRoFormer** (`BSRoFormer.py`) | PyTorch (`bs-roformer-infer`) | ✅ **Yes** | ✅ Supported | **PASS** (6.09s on 3s audio track) | Uses Band-Split Rotary Embeddings, SDPA Attention, and `rocFFT`. |
| **MelRoFormer** (`MelRoFormer.py`) | PyTorch (`melband-roformer-infer`) | ✅ **Yes** | ✅ Supported | **PASS** (1.79s on 3s audio track) | Uses Mel-Band Attention, `torch.amp.autocast`, and `rocFFT`. |
| **MVSepMDX23** (`MVSepMDX23.py`) | PyTorch / ONNX | ⚠️ **Partial** | ✅ Supported | PyTorch model runs on GPU; ONNX engine runs on CPU | Standard `onnxruntime-gpu` does not bundle ROCm provider on Linux PyPI wheels. |

### B. Diarization, Voice Activity & Verification Models

| Component / Model | Architecture | GPU Support (`cuda:0`) | CPU Fallback | Test Result | Failure Mode / Special Requirement |
|---|---|---|---|---|---|
| **Silero VAD** (`zero_contamination.py`) | TorchScript CNN + LSTM | ✅ **Yes** | ✅ Supported | **PASS** (< 10 ms forward pass) | Native TorchScript JIT execution on GPU. Requires 512-sample frames @ 16 kHz. |
| **Pyannote Audio** (`PyannoteDiarizer.py`) | SincNet + BiLSTM + Linear | ✅ **Yes** | ✅ Supported | **PASS** (PyanNet forward pass: `[2, 293, 4]`) | Gated weights require `HF_TOKEN`. `torchcodec` must be disabled in favor of `torchaudio`. |
| **SpeakerVerifier** (`SpeakerVerifier.py`) | WeSpeaker ResNet34 | ✅ **Yes** | ✅ Supported | **PASS** (4.20s extraction, 256-dim embedding) | Uses public HuggingFace model `pyannote/wespeaker-voxceleb-resnet34-LM`. |
| **Whisper Timestamped** (`openai-whisper`) | Transformer Encoder-Decoder | ✅ **Yes** | ✅ Supported | **PASS** (Logits shape: `[1, 1, 51865]`) | Attention accelerated via Flash/Mem-Efficient SDPA on ROCm. |
| **Clustering Diarizer** (`ClusteringDiarizer.py`) | NeMo MarbleNet + TitaNet | ⚠️ **Isolated** | ✅ Supported | Requires `.venv-sortformer` worker environment | Worker script delegates to dedicated NeMo sub-environment. |
| **Sortformer Diarizer** (`SortformerDiarizer.py`) | NeMo Transformer | ⚠️ **Isolated** | ✅ Supported | Requires `.venv-sortformer` worker environment | Worker script delegates to dedicated Sortformer sub-environment. |
| **3D-Speaker Diarizer** (`ThreeDSpeakerDiarizer.py`) | ModelScope CAM++ / ERes2Net | ⚠️ **Isolated** | ✅ Supported | Requires `.venv-3dspeaker` worker environment | Worker script delegates to dedicated 3D-Speaker sub-environment. |
| **DiariZen Diarizer** (`DiariZenDiarizer.py`) | BUT WavLM + VBx | ⚠️ **Isolated** | ✅ Supported | Requires `.venv-diarizen` worker environment (Python 3.10) | Runs on CPU fallback for RDNA 4 architectures. |
| **VibeVoice-ASR Verifier** (`VibeVoicePurityVerifier.py`) | Microsoft VibeVoice-ASR | ⚠️ **Isolated** | ✅ Supported | Requires `.venv-vibevoice` worker environment | Uses bfloat16 SDPA attention on ROCm GPU. |

---

## 4. Heavy Math & DSP Operations Breakdown

### Mathematical Operations Running on AMD GPU (ROCm)

These operations are accelerated directly through AMD's hardware compute engines (`rocBLAS`, `MIOpen`, `rocFFT`):

1. **General Matrix Multiplication (GEMM):**
   - Implemented via `torch.matmul` / `@` / `torch.bmm`.
   - Accelerated by **`rocBLAS`**.
2. **Convolutions (1D, 2D):**
   - Implemented via `torch.nn.Conv1d`, `torch.nn.Conv2d`.
   - Accelerated by **`MIOpen`** kernels.
3. **Recurrent Layers (LSTM, Bi-LSTM, GRU):**
   - Implemented via `torch.nn.LSTM`.
   - Accelerated by **`MIOpen RNN`** engines.
4. **Attention Mechanisms (SDPA):**
   - Implemented via `torch.nn.functional.scaled_dot_product_attention`.
   - Accelerated by ROCm AOTriton memory-efficient attention kernels.
5. **Time-Frequency Transforms (STFT & iSTFT):**
   - Implemented via `torch.stft`, `torch.istft`.
   - Accelerated by **`rocFFT`** / **`hipFFT`** (verified relative reconstruction difference norm: `~5.19e-5`).
6. **Spectral Audio Transforms:**
   - Implemented via `torchaudio.transforms.MelSpectrogram`, `torchaudio.transforms.MFCC`, `torchaudio.transforms.Spectrogram`, `torchaudio.transforms.SpectralCentroid`.
   - Executed entirely in GPU VRAM without host transfers.
7. **Audio Resampling:**
   - Implemented via `torchaudio.transforms.Resample` (polyphase filter bank on GPU).
8. **Pairwise Vector Distances & Similarity:**
   - Implemented via `torch.cdist`, `torch.nn.functional.cosine_similarity`.
   - Vectorized matrix operations on GPU.

### Mathematical Operations That MUST Fall Back to CPU

These operations do not have native ROCm GPU acceleration in the standard scientific Python ecosystem and run on the CPU host:

1. **Hungarian Assignment Algorithm (`_maximum_weight_assignment`):**
   - `scipy.optimize.linear_sum_assignment`: Strictly single-threaded CPU graph matching algorithm used during dual-engine consensus and DER speaker alignment.
2. **Unsupervised Speaker Clustering:**
   - `sklearn.cluster.AgglomerativeClustering`: CPU-bound linkage clustering over distance matrices.
   - `sklearn.cluster.SpectralClustering`: Scipy/LAPACK CPU eigensolvers (`scipy.sparse.linalg.eigsh`).
3. **Audio File I/O and Codec Decoding:**
   - `soundfile.read()`, `soundfile.write()`, `wave.open()`, `ffmpeg`: Decodes raw container bits (WAV, MP3, FLAC) into PCM arrays in host system RAM.
4. **Librosa DSP Operations:**
   - `librosa.effects.pitch_shift`, `librosa.stft`: Implemented via NumPy and SciPy FFTW on CPU.
5. **Separation Quality Metrics:**
   - BSS-eval, SDR, SI-SDR, SIR, SAR when calculated via `fast_bss_eval` or NumPy/SciPy on host memory.
6. **ONNX Runtime (with default packages):**
   - PyPI `onnxruntime-gpu` targets CUDA/TensorRT and lacks `ROCMExecutionProvider` on generic wheels. ONNX models fall back to `CPUExecutionProvider`.

---

## 5. Automated Hardware Detection & Setup (`start_web.sh` & `setup_worker_envs.sh`)

Running `./scripts/start_web.sh` (or `start_studio.sh` / `start_pipeline.sh`) automatically detects whether the host is equipped with an AMD GPU (ROCm) or NVIDIA GPU (CUDA), sets the required environment flags, and reconciles the virtual environment's PyTorch stack without manual intervention:

- **Primary & Worker Virtual Environments Reconciled:** On server startup, `start_web.sh` inspects the primary `.venv` and **all existing worker environments** (`.venv-sortformer`, `.venv-3dspeaker`, `.venv-vibevoice`, `.venv-diarizen`). If an environment is running mismatched wheels (e.g. CUDA wheels on an AMD machine), it automatically swaps them for the host accelerator without user intervention.
- **Dedicated Worker Provisioning Script (`setup_worker_envs.sh`):** Use `./scripts/setup_worker_envs.sh [all|sortformer|3dspeaker|vibevoice|diarizen|status]` to bootstrap any or all worker environments with auto-detected hardware configuration.
- **Automatic AMD ROCm Bootstrap:** Detects AMD GPU hardware, ensures ROCm tools are in `PATH`, sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, verifies PyTorch HIP support, and installs modular ROCm wheels if needed.
- **Automatic NVIDIA CUDA Bootstrap:** Detects NVIDIA GPU hardware and ensures CUDA-enabled PyTorch wheels are in place.
- **Hardware Telemetry Integration:** Automatically polls `rocm-smi` (for AMD) or `nvidia-smi` (for NVIDIA) to display live GPU temperature, utilization, VRAM usage, and power draw in the SonicStudio and SonicPipeline web dashboards.

### Manual RDNA 4 (`gfx1200` / RX 9060 XT) Setup Reference

If setting up a manual standalone virtual environment outside the launcher scripts:

1. **Use AMD ROCm Modular Wheels:**
   Standard PyPI `torch` defaults to NVIDIA CUDA. For ROCm with `gfx1200` support, install from AMD's official wheel repository:
   ```bash
   pip install --no-deps \
     https://stable.repo.amd.com/rocm/pytorch/whl-next/torch/torch-2.13.0%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
     https://stable.repo.amd.com/rocm/pytorch/whl-next/torchaudio/torchaudio-2.11.0.2%2Brocm10.0.0-cp313-cp313-linux_x86_64.whl \
     https://stable.repo.amd.com/rocm/pytorch/whl-next/triton/triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp313-cp313-linux_x86_64.whl
   ```
2. **Device Kernel Pack (`kpack`):**
   Ensure `amd-torch-device-gfx1200` is present so that PyTorch finds the compiled kernel binaries for Navi 44 (`torch_gfx1200.kpack`).
3. **Avoid CUDA `torchcodec`:**
   Do not install `torchcodec` from PyPI, as it requires NVIDIA's `libnvrtc.so.13`. Pyannote and torchaudio seamlessly fall back to `torchaudio` and `soundfile` without it.
4. **Environment Flag for Flash / Mem-Efficient Attention:**
   ```bash
   export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
   ```
   This enables AOTriton fused attention kernels on newer AMD architectures.

---

## 6. Device-Agnostic Process Isolation & GPU Pinning

When worker models (`SortformerWorkerDiarizer`, `ClusteringWorkerDiarizer`, `ThreeDSpeakerWorkerDiarizer`, `DiariZenWorkerDiarizer`, `VibeVoicePurityWorkerVerifier`, `MVSepMDX23`) execute in subprocesses, device allocation is completely device-agnostic:

- **AMD ROCm / HIP Device Isolation:** When a task specifies a GPU index (e.g. `cuda:1`), the worker manager sets `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES` simultaneously. This guarantees that ROCm HIP and CUDA runtime layers both expose only the target physical GPU to the child process as device 0.
- **CPU Fallback Lane:** When running in the `cpu` queue, `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and `ROCR_VISIBLE_DEVICES` are set to `""`, completely suppressing GPU runtime initialization and ensuring pure CPU execution.
- **ROCm Path & Kernel Propagation:** All worker subprocesses inherit `/opt/rocm/bin` in `PATH` and `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for optimal kernel performance across all isolated environments.
