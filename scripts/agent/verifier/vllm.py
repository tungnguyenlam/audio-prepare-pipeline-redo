"""Verify audio using vLLM (offline batch inference or client mode via OpenAI-compatible endpoint)."""
from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _audio import load_audio_waveform, parse_verifier_response
from _cli import load_prompt, run_verifier
from _common.files import destinations, parser
from scripts.agent.verifier.endpoint import EndpointVerifier

logger = logging.getLogger("verifier.vllm")

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"


class VLLMServerVerifier(EndpointVerifier):
    """Queries a running vLLM server via OpenAI-compatible chat completions."""

    def __init__(
        self,
        endpoint: str = "http://localhost:8000/v1/chat/completions",
        model: str = DEFAULT_MODEL_ID,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            model=model,
            timeout_s=timeout_s,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        parsed = super().verify(audio_path, prompt)
        parsed["_engine"] = "vllm_server"
        parsed["_model"] = self.model
        return parsed


class VLLMOfflineVerifier:
    """Performs offline batch audio inference directly via vllm.LLM."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL_ID,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
        trust_remote_code: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vllm is not installed. Install vllm or run against an endpoint via --endpoint."
            ) from exc

        logger.info("Initializing offline vLLM with model '%s'...", model)
        self.model_id = model
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
        )

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        t0 = time.time()
        audio_data = load_audio_waveform(audio_path, target_sr=16000)

        # Standard multimodal prompt formatting for vLLM audio models
        prompt_with_tag = f"{prompt}\n<|audio|>"
        inputs = {
            "prompt": prompt_with_tag,
            "multi_modal_data": {"audio": (audio_data, 16000)},
        }

        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
        latency = round(time.time() - t0, 3)

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM offline generation produced no outputs")

        output_text = outputs[0].outputs[0].text
        parsed = parse_verifier_response(output_text)
        parsed["_latency_s"] = latency
        parsed["_engine"] = "vllm_offline"
        parsed["_model"] = self.model_id
        return parsed


def main() -> int:
    p = parser("Verify audio with vLLM; writes verdicts without filtering audio.", "agent/verifier", "vllm")
    p.add_argument("--prompt-file", type=Path, help="Path to custom prompt text file")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    p.add_argument("--endpoint", type=str, default=None, help="vLLM server endpoint URL (runs in server mode if set)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "auto"), help='Model weights precision ("bfloat16", "float16", or "auto")')
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Fraction of GPU memory to reserve for vLLM engine")
    p.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length in tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=512, help="Maximum generated tokens")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True, help="Allow executing custom code from Hugging Face model repository")
    args = p.parse_args()

    pairs = destinations(args, "_vllm", ".json")

    prompt = load_prompt(args.prompt_file)

    parameters = {
        "model": args.model,
        "endpoint": args.endpoint,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    with contextlib.redirect_stdout(sys.stderr):
        if args.endpoint:
            verifier = VLLMServerVerifier(
                endpoint=args.endpoint,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        else:
            verifier = VLLMOfflineVerifier(
                model=args.model,
                dtype=args.dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                trust_remote_code=args.trust_remote_code,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

    parameters["prompt"] = prompt

    return run_verifier(
        args=args, pairs=pairs, backend="vllm", parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )


if __name__ == "__main__":
    raise SystemExit(main())
