# Hardware notes: AMD ROCm and NVIDIA CUDA

[← Overview](../README.md) · [Provisioning](commands.md#provisioning)

## Machines

| Host | Role | Accelerator |
|---|---|---|
| `tungnl5@VF-TUNGNL5-L` | primary development | AMD Radeon RX 9060 XT, 16 GB, RDNA 4 (`gfx1200`), ROCm 10.0 / HIP, PyTorch `2.13.0+rocm10.0.0` |
| `vsf@10.148.21.12` (`server`) | model server | NVIDIA CUDA |
| `loi` (ssh alias; `loinh8@10.148.1.176`), `anhnct@10.148.21.113` | auxiliary compute | — |

`scripts/sync/{code,data}_{to,from}_{server,loi,anhnct}.sh` rsync code and
`.data/` between machines; hosts and remote paths are overridable via
`SYNC_<NAME>_HOST` / `SYNC_<NAME>_REPO`, exclusions live in
`scripts/sync/data_excludes.txt`. Credentials and runtime artifacts stay machine-local.

## What `envs/setup_worker_envs.sh` does per accelerator

- **AMD:** detected via `rocm-smi`; adds `/opt/rocm/bin` to `PATH`, exports
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, installs the ROCm 10.0 wheel set
  (`torch==2.13.0+rocm10.0.0`, `torchaudio==2.11.0.2+rocm10.0.0`, `triton`,
  `rocm-sdk-*`, `amd-torch-device-gfx1200`) from `stable.repo.amd.com` when the
  venv has non-HIP torch, and uninstalls `torchcodec` (needs CUDA `libnvrtc`).
- **NVIDIA:** detected via `nvidia-smi`; picks the PyTorch wheel index matching the
  driver's CUDA version (e.g. `cu128` for drivers < 580).
- **CPU:** default PyPI wheels.
- `status` prints each venv's torch build and device (`ROCm/HIP`, `CUDA: <name>`, `CPU`).

Manual ROCm install for reference:

```bash
uv pip install \
  --extra-index-url https://stable.repo.amd.com/rocm/core/whl-next/ \
  --extra-index-url https://stable.repo.amd.com/rocm/pytorch/whl-next/ \
  --index-strategy unsafe-best-match \
  "torch==2.13.0+rocm10.0.0" "torchaudio==2.11.0.2+rocm10.0.0" \
  "triton==3.8.0+git4cff872c.rocm10.0.0" "rocm==10.0.0" "rocm-sdk-core==10.0.0" \
  "rocm-sdk-libraries==10.0.0" "rocm-sdk-device-gfx1200==10.0.0" \
  "amd-torch-device-gfx1200==2.13.0+rocm10.0.0"
```

Do not install `amd-torch-device-gfx1200` from PyPI (empty placeholder package).
Match `torchvision` to the same AMD index (`0.28.0+rocm10.0.0`) to avoid
`operator torchvision::nms does not exist`.

## Per-backend behaviour on the AMD host (`--device cuda:0` maps to HIP)

| Backend | Status | Note |
|---|---|---|
| HTDemucs, BS-RoFormer, Mel-RoFormer | GPU | MIOpen conv/LSTM, SDPA, rocFFT |
| MVSEP-MDX23 | partial | PyTorch part on GPU; ONNX falls back to CPU (`onnxruntime-gpu` wheels lack a ROCm provider) |
| Pyannote 3.1 / Community-1, WeSpeaker scoring, whisper-timestamped | GPU | `HF_TOKEN` for gated pyannote weights; runs without `torchcodec` |
| Sortformer, clustering, 3D-Speaker, VibeVoice | GPU in isolated venvs | worker venvs are provisioned with the same wheel logic; VibeVoice `--quantization int8` / `nf4` is NVIDIA CUDA-only. 3D-Speaker's ROCm torchaudio 2.11 lacks `AudioMetaData` / `info` / `list_audio_backends` and `load` needs torchcodec; `threed_speaker.py` falls back to soundfile. |
| DiariZen | GPU | Neural segmentation & WeSpeaker embeddings run on AMD GPU; VBx/AHC clustering on CPU |
| Gemma 4 (HF verifier, LoRA training) | GPU, bf16 | Native bfloat16 inference on GPU; see constraints below |
| Hungarian matching, sklearn clustering, libsndfile / ffmpeg I/O, BSS metrics | CPU | no ROCm path in the scientific Python stack |
| DeepFilterNet, ClearVoice (MossFormer2/FRCRN), VoiceFixer | CPU on this host | isolated Python 3.11 venvs; ROCm 10.0 wheels are 3.13-only so AMD uses CPU torch. NVIDIA hosts install CUDA torch in the same venvs. |

## Constraints learned on RDNA 4 / `gfx1200`

- **Gemma 4 attention:** unset `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` before
  running or training Gemma 4 on this GPU; with it set, attention fails with
  `torch.AcceleratorError: CUDA error: invalid argument (hipErrorInvalidValue)`.
  Plain SDPA / rocBLAS attention is stable. The flag remains useful for separation
  and diarization workloads, which is why the setup script exports it.
- **Quantization:** `bitsandbytes` 4/8-bit kernels are CUDA-only; use native
  `bfloat16` (`--quantization none` in past training runs). Gemma 4 E2B bf16 base
  weights are ~4.6 GB. VibeVoice ASR/verifier `--quantization int8` / `nf4` load
  the Dubedo selective bitsandbytes checkpoints and likewise require NVIDIA CUDA;
  on this AMD host keep `--quantization none` (full BF16).
- **LoRA fits, full fine-tuning does not:** 2.3 B parameters full fine-tune needs
  ≥ 27 GB (weights + grads + AdamW); LoRA r=16/α=32 on `q/k/v/o/gate/up/down_proj`
  peaked at ~16.3 GB of 16.4 GB in past runs (≈ 0.8 s/sample).
- **GPU pinning for subprocesses:** set `HIP_VISIBLE_DEVICES`,
  `ROCR_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` together; set all three to `""`
  for a pure-CPU lane.
