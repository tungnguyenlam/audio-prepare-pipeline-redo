from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from _audio import load_audio_waveform  # noqa: E402

logger = logging.getLogger("agent.hf")


class HFAgent:
    """Raw generation client for Hugging Face multimodal audio models."""

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        device: str = "auto",
        adapter_path: str | None = None,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        hf_token: str | None = None,
    ) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # Normalize and validate device across AMD ROCm (HIP) and NVIDIA CUDA.
        # In PyTorch, AMD ROCm GPUs are exposed and addressed via the 'cuda' device API (e.g. 'cuda:0').
        raw_device = (device or "auto").strip().lower()
        if raw_device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif raw_device.startswith("hip") or raw_device.startswith("rocm"):
            suffix = raw_device[4:] if raw_device.startswith("rocm") else raw_device[3:]
            if suffix.startswith(":"):
                self.device = f"cuda{suffix}"
            elif suffix:
                self.device = f"cuda:{suffix}"
            else:
                self.device = "cuda:0" if torch.cuda.is_available() else "cuda"
        else:
            self.device = raw_device

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device is unavailable: {self.device}")

        # Check for AMD ROCm / HIP environment
        self.is_rocm = getattr(torch.version, "hip", None) is not None
        if self.is_rocm:
            # Prevent AOTriton experimental kernel crashes (hipErrorInvalidValue) on RDNA 4 (gfx1200)
            if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1":
                logger.info("Disabling TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL for stable ROCm attention on AMD GPU")
                os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"

        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        hw_type = f"AMD ROCm ({torch.version.hip})" if self.is_rocm else ("NVIDIA CUDA" if torch.cuda.is_available() else "CPU")
        dev_desc = torch.cuda.get_device_name(0) if (self.device.startswith("cuda") and torch.cuda.is_available()) else "CPU"
        logger.info("Initializing HFAgent '%s' on %s [%s: %s] (dtype=%s)...", model_id, self.device, hw_type, dev_desc, torch_dtype)

        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForMultimodalLM,
            AutoProcessor,
        )

        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=self.hf_token,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            token=self.hf_token,
        )
        self.processor_class = type(self.processor).__name__

        quant_kwargs: dict[str, Any] = {}
        if load_in_4bit or load_in_8bit:
            if self.is_rocm:
                logger.warning(
                    "BitsAndBytes quantization (4-bit/8-bit) may fail or be unsupported on AMD ROCm consumer GPUs. "
                    "Native bfloat16 is recommended for AMD ROCm."
                )
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=self.dtype,
                )
            except ImportError as exc:
                raise RuntimeError("bitsandbytes required for --load-in-4bit / --load-in-8bit") from exc

        self.model = None
        if getattr(config, "model_type", None) == "gemma4":
            # Gemma 4 audio must use the multimodal conditional-generation model.
            # Do not hide a failed multimodal load behind a text-model fallback.
            candidate_classes = [AutoModelForMultimodalLM]
        else:
            candidate_classes = [
                AutoModelForMultimodalLM,
                AutoModelForImageTextToText,
                AutoModelForCausalLM,
                AutoModel,
            ]

        load_errors: list[str] = []
        for cls in candidate_classes:
            try:
                self.model = cls.from_pretrained(
                    model_id,
                    dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=trust_remote_code,
                    token=self.hf_token,
                    **quant_kwargs,
                )
                logger.info("Successfully loaded model using %s", cls.__name__)
                break
            except TypeError:
                try:
                    self.model = cls.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                        device_map=self.device,
                        trust_remote_code=trust_remote_code,
                        token=self.hf_token,
                        **quant_kwargs,
                    )
                    logger.info("Successfully loaded model using %s", cls.__name__)
                    break
                except Exception as e:
                    load_errors.append(f"{cls.__name__}: {type(e).__name__}")
                    logger.debug(
                        "Failed loading with %s (%s)", cls.__name__, type(e).__name__
                    )
                    continue
            except Exception as e:
                load_errors.append(f"{cls.__name__}: {type(e).__name__}")
                logger.debug(
                    "Failed loading with %s (%s)", cls.__name__, type(e).__name__
                )
                continue

        if self.model is None:
            err_detail = " | ".join(load_errors)
            raise RuntimeError(f"Could not load model '{model_id}' with any supported model class. Errors: {err_detail}")
        self.model_class = type(self.model).__name__

        if adapter_path:
            from peft import PeftModel
            logger.info("Attaching LoRA adapter from '%s'...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        audio_data = load_audio_waveform(audio_path, target_sr=16000)

        audio_part = {"type": "audio", "audio": audio_data}
        prompt_part = {"type": "text", "text": prompt}
        content = [prompt_part, audio_part]
        messages = []
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append({"role": "user", "content": content})
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(self.device)

        use_cuda = self.device.startswith("cuda")
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if use_cuda
            else torch.autocast(device_type="cpu", dtype=self.dtype)
        )

        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation["temperature"] = temperature
            if top_p is not None:
                generation["top_p"] = top_p

        with torch.no_grad(), autocast_ctx:
            generated_ids = self.model.generate(**inputs, **generation)

        new_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
        output_text = self.processor.decode(new_tokens, skip_special_tokens=True)

        latency = round(time.time() - t0, 3)
        return {
            "text": output_text,
            "latency_s": latency,
            "usage": {
                "output_tokens": int(new_tokens.shape[-1]),
                "total_tokens": int(new_tokens.shape[-1]),
            },
            "provider_body": {
                "generated_text": output_text,
                "generated_token_count": int(new_tokens.shape[-1]),
            },
        }


def main() -> int:
    import argparse
    import contextlib
    from artifacts import add_prompt_arguments, load_prompts, run_agent
    from _common.files import destinations, parser

    command = parser(
        "Explore a local Hugging Face audio model without parsing its response.",
        "s4-agent",
        "hf",
    )
    add_prompt_arguments(command, top_p=True)
    command.add_argument("--model-id", default="google/gemma-4-E2B-it", help="Hugging Face model repository ID")
    command.add_argument("--device", default="auto", help='Inference device ("auto", "cpu", "cuda", or "hip")')
    command.add_argument("--adapter-path", help="Optional LoRA adapter checkpoint directory")
    command.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow executing custom code from Hugging Face model repository",
    )
    command.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="PyTorch weights dtype",
    )
    command.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit NF4 with bitsandbytes")
    command.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit with bitsandbytes")
    args = command.parse_args()

    prompt, system_prompt = load_prompts(args)
    parameters = {
        "model_id": args.model_id,
        "device": args.device,
        "adapter_path": args.adapter_path,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": args.torch_dtype,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    pairs = destinations(args, "_hf", ".txt")
    model_holder: list[HFAgent] = []

    def get_model() -> HFAgent:
        if not model_holder:
            model_holder.append(
                HFAgent(
                    model_id=args.model_id,
                    device=args.device,
                    adapter_path=args.adapter_path,
                    trust_remote_code=args.trust_remote_code,
                    torch_dtype=args.torch_dtype,
                    load_in_4bit=args.load_in_4bit,
                    load_in_8bit=args.load_in_8bit,
                )
            )
        return model_holder[0]

    def runner(audio_path: Path) -> dict[str, Any]:
        agent = get_model()
        return agent.generate(
            audio_path,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    with contextlib.ExitStack():
        return run_agent(
            args=args,
            pairs=pairs,
            backend="hf",
            parameters=parameters,
            generate=runner,
        )


if __name__ == "__main__":
    raise SystemExit(main())
