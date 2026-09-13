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
| **DiariZen Diarizer** (`diarizen.py` / `diarizen.sh`) | BUT WavLM + VBx | ✅ **Yes** | ✅ Supported | **PASS** (9.5s on 90s audio track) | Neural segmentation & WeSpeaker embeddings run on AMD GPU (`cuda:0`); VBx/AHC clustering executes on CPU host. |
| **HF Agent & Verifier** (`scripts/agent/hf.sh`, `scripts/agent/verifier/hf.sh`) | Multimodal LLM (Gemma 4 2B/E2B-it) | ✅ **Yes** | ✅ Supported | **PASS** (8.58s on 90s audio track) | Native bfloat16 multimodal inference on AMD GPU (`cuda:0` / `--device hip`); automatic device normalization and SDPA stability guards. |
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

### 5. Automated Hardware Detection & Setup (`setup_worker_envs.sh`)
 
 Running `./envs/setup_worker_envs.sh` (or `./scripts/setup_worker_envs.sh`) automatically detects whether the host is equipped with an AMD GPU (ROCm) or NVIDIA GPU (CUDA), sets the required environment flags, and reconciles the virtual environment's PyTorch stack without manual intervention:
 
 - **Primary & Worker Virtual Environments Reconciled:** The provisioning script inspects the primary `.venvs/main` and **all worker environments** (`.venvs/sortformer`, `.venvs/3dspeaker`, `.venvs/vibevoice`, `.venvs/diarizen`). If an environment is running mismatched wheels (e.g. CUDA wheels on an AMD machine), it automatically swaps them for the host accelerator without user intervention.
 - **Dedicated Worker Provisioning Script (`setup_worker_envs.sh`):** Use `./envs/setup_worker_envs.sh [all|sortformer|3dspeaker|vibevoice|diarizen|status]` to bootstrap any or all worker environments with auto-detected hardware configuration.
 - **Automatic AMD ROCm Bootstrap:** Detects AMD GPU hardware, ensures ROCm tools are in `PATH`, sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, verifies PyTorch HIP support, and installs modular ROCm wheels if needed.
 - **Automatic NVIDIA CUDA Bootstrap:** Detects NVIDIA GPU hardware and ensures CUDA-enabled PyTorch wheels are in place.
 - **Hardware Telemetry Verification:** Verifies device availability using `rocm-smi` (for AMD) or `nvidia-smi` (for NVIDIA) and tests PyTorch GPU tensor execution.

### Manual RDNA 4 (`gfx1200` / RX 9060 XT) Setup Reference

If setting up a manual standalone virtual environment outside the launcher scripts:

1. **Use AMD ROCm Modular Wheels & SDK:**
   Standard PyPI `torch` defaults to NVIDIA CUDA. For ROCm with `gfx1200` support, install PyTorch, ROCm SDK, and device kernel pack (`amd-torch-device-gfx1200`) from AMD's official wheel repository:
   ```bash
   uv pip install \
     --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/ \
     --extra-index-url https://stable.repo.amd.com/rocm/pytorch/whl-next/ \
     --index-strategy unsafe-best-match \
     "torch==2.13.0+rocm10.0.0" \
     "torchaudio==2.11.0.2+rocm10.0.0" \
     "triton==3.8.0+git4cff872c.rocm10.0.0" \
     "rocm==10.0.0" \
     "rocm-sdk-core==10.0.0" \
     "rocm-sdk-libraries==10.0.0" \
     "rocm-sdk-device-gfx1200==10.0.0" \
     "amd-torch-device-gfx1200==2.13.0+rocm10.0.0"
   ```
2. **Device Kernel Pack (`kpack`):**
   The `amd-torch-device-gfx1200` package bundles the compiled kernel binaries for Navi 44 (`torch_gfx1200.kpack`) and `rocm-sdk-device-gfx1200`. Note: do not install `amd-torch-device-gfx1200` directly from PyPI as PyPI only hosts an empty 0.0.1 dummy package.
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
- **ROCm Path & Kernel Propagation:** All worker subprocesses inherit `/opt/rocm/bin` in `PATH`. Note that while `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` can benefit specific conv/GEMM workloads, it should **never** be enabled for multimodal audio LLM training (see Section 7.1).

---

## 7. Model Fine-Tuning & LoRA Distillation on AMD GPU

Fine-tuning compact multimodal acoustic verifiers (such as `google/gemma-4-E2B-it`) can be performed directly on the AMD Radeon RX 9060 XT (16 GB VRAM) using native ROCm HIP acceleration without falling back to CPU or remote servers.

### 7.1 LoRA vs. Full Fine-Tuning Architectural Justification

| Metric | Full Fine-Tuning (`bfloat16`) | LoRA Fine-Tuning ($r=16, \alpha=32$) |
|---|---|---|
| **Trainable Parameters** | 2,300,000,000 (100%) | 24,150,000 (0.47%) |
| **Model Weight VRAM** | 4.6 GB (`bfloat16`) | 4.6 GB (`bfloat16` frozen base) |
| **Gradients VRAM** | 4.6 GB | ~48 MB (LoRA adapters only) |
| **Optimizer States (AdamW)** | 18.4 GB ($8 \times 2.3\text{B}$) | ~96 MB ($8 \times 24.15\text{M}$) |
| **Activations (checkpointed)** | ~2.5 - 3.5 GB | ~2.5 - 3.5 GB |
| **Minimum Required VRAM** | **$\ge 27.6\text{ GB}$ (Immediate OOM)** | **~11.5 – 13.5 GB (Fits comfortably in 16 GB)** |

**Engineering Verdict:** Full fine-tuning of Gemma 4 E2B is physically impossible on 16 GB VRAM consumer GPUs. Parameter-Efficient Fine-Tuning (PEFT / LoRA) targeting projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) is mandatory, maintaining high adaptation quality while using under 14 GB peak VRAM.

### 7.2 Critical RDNA 4 (`gfx1200`) Attention Kernel Constraint

> [!WARNING]
> Setting `export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` causes PyTorch attention kernels in Gemma 4 to trigger:
> ```
> torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)
> ```
> **Rule:** Do NOT export `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for Gemma 4 training or evaluation on Navi 44 / gfx1200. Allow PyTorch to use standard SDPA / rocBLAS attention, which executes stably at 100% GPU utilization with zero crashes.

### 7.3 Recommended Configuration & Command
- **Precision:** Native `bfloat16` (`--quantization none`). Gemma 4 E2B base weights consume ~4.6 GB VRAM.
- **Quantization:** Avoid 4-bit/8-bit quantization (`bitsandbytes`) on AMD ROCm consumer cards (RDNA 4 / `gfx1200`), as upstream `bitsandbytes` wheels compile kernels specifically for NVIDIA CUDA. Native `bfloat16` is faster and avoids CUDA kernel dependency.
- **Dependency Alignment:** Ensure `torchvision` is installed from AMD's official wheel repository (`torchvision==0.28.0+rocm10.0.0`) to avoid ABI symbol incompatibilities (`operator torchvision::nms does not exist`).
- **CLI Runner:**
  ```bash
  .venv/bin/python scripts/train_verifier.py \
    --model-id google/gemma-4-E2B-it \
    --device cuda:0 \
    --quantization none \
    --epochs 3 \
    --lr 2e-4 \
    --accum-steps 4 \
    --train-file .data/distillation/train_v3.jsonl \
    --val-file .data/distillation/val_v3.jsonl \
    --output-dir .data/distillation/checkpoints_e2b_v3/best_adapter
  ```

