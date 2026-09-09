"""Verify audio using vLLM (offline batch inference or client mode via OpenAI-compatible endpoint)."""
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _audio import DEFAULT_ACOUSTIC_PROMPT, extract_json_payload, load_audio_waveform
from _common.files import batch, destinations, identity, parser, read_json, request, write_json

logger = logging.getLogger("verifier.vllm")

DEFAULT_MODEL_ID = "google/gemma-4-E2B-it"


class VLLMServerVerifier:
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
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        latency = round(time.time() - t0, 3)

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"vLLM server returned no choices: {data}")

        msg = choices[0].get("message", {})
        raw_content = msg.get("content", "")
        parsed = extract_json_payload(raw_content)
        parsed["_latency_s"] = latency
        parsed["_engine"] = "vllm_server"
        parsed["_model"] = self.model
        if "usage" in data:
            parsed["_usage"] = data["usage"]
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
        prompt_with_tag = f"<|audio|>\n{prompt}"
        inputs = {
            "prompt": prompt_with_tag,
            "multi_modal_data": {"audio": (audio_data, 16000)},
        }

        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
        latency = round(time.time() - t0, 3)

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM offline generation produced no outputs")

        output_text = outputs[0].outputs[0].text
        parsed = extract_json_payload(output_text)
        parsed["_latency_s"] = latency
        parsed["_engine"] = "vllm_offline"
        parsed["_model"] = self.model_id
        return parsed


def main() -> int:
    p = parser("Verify audio with vLLM; writes verdicts without filtering audio.", "verify", "vllm")
    p.add_argument("--prompt-file", type=Path, help="Path to custom prompt text file")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help="Hugging Face model ID")
    p.add_argument("--endpoint", type=str, default=None, help="vLLM server endpoint URL (runs in server mode if set)")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "auto"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    pairs = destinations(args, "_vllm", ".json")

    # Load prompt: priority is user --prompt-file -> prompts/acoustic_defect.txt -> DEFAULT_ACOUSTIC_PROMPT
    prompt = None
    if args.prompt_file and args.prompt_file.is_file():
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    elif (Path("prompts/acoustic_defect.txt")).is_file():
        prompt = Path("prompts/acoustic_defect.txt").read_text(encoding="utf-8").strip()
    else:
        prompt = DEFAULT_ACOUSTIC_PROMPT

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

    def process(src: Path, dest: Path) -> None:
        wanted = request(identity(src), "verify", parameters, "vllm")
        if dest.exists() and not args.overwrite:
            old = read_json(dest)
            if all(old.get(k) == v for k, v in wanted.items()) and "verdict" in old:
                return
            raise ValueError(f"Conflicting output: {dest}; use --overwrite")

        verdict = verifier.verify(src, prompt)
        if not isinstance(verdict, dict) or verdict.get("decision") not in {"pass", "reject"}:
            raise ValueError("Verifier did not return a pass/reject decision")
        write_json(dest, {**wanted, "verdict": verdict})

    return batch(pairs, process)


if __name__ == "__main__":
    raise SystemExit(main())
