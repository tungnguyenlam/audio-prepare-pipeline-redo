from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import logging
import os
import time
from pathlib import Path
from typing import Any

import torch

from _audio import (
    extract_json_payload,
    load_audio_waveform,
)

logger = logging.getLogger("verifier.default_hf")


class DefaultHFVerifier:
    """Default Hugging Face multimodal audio verifier (e.g., Gemma 4 E2B/E4B, standard HF models)."""

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        device: str = "auto",
        adapter_path: str | None = None,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        hf_token: str | None = None,
    ) -> None:
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device is unavailable: {self.device}")
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        logger.info("Initializing DefaultHFVerifier '%s' on %s (dtype=%s)...", model_id, self.device, torch_dtype)

        from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
        try:
            from transformers import Gemma4ForConditionalGeneration
        except ImportError:
            Gemma4ForConditionalGeneration = None

        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=self.hf_token)
        except Exception:
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, token=self.hf_token)

        self.model = None
        candidate_classes = []
        if "gemma-4" in model_id.lower() and Gemma4ForConditionalGeneration is not None:
            candidate_classes.append(Gemma4ForConditionalGeneration)
        try:
            from transformers import AutoModelForImageTextToText
            candidate_classes.append(AutoModelForImageTextToText)
        except Exception:
            pass
        candidate_classes.extend([AutoModelForCausalLM, AutoModel])

        load_errors: list[str] = []
        for cls in candidate_classes:
            try:
                self.model = cls.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    device_map=self.device,
                    trust_remote_code=trust_remote_code,
                    token=self.hf_token,
                )
                logger.info("Successfully loaded model using %s", cls.__name__)
                break
            except Exception as e:
                load_errors.append(f"{cls.__name__}: {e}")
                logger.debug("Failed loading with %s: %s", cls.__name__, e)
                continue

        if self.model is None:
            err_detail = " | ".join(load_errors)
            raise RuntimeError(f"Could not load model '{model_id}' with any supported model class. Errors: {err_detail}")

        if adapter_path:
            from peft import PeftModel
            logger.info("Attaching LoRA adapter from '%s'...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        t0 = time.time()
        audio_data = load_audio_waveform(audio_path, target_sr=16000)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_data},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=text, audio=audio_data, return_tensors="pt", sampling_rate=16000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_cuda = self.device.startswith("cuda")
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.dtype)
            if use_cuda
            else torch.autocast(device_type="cpu", dtype=self.dtype)
        )

        with torch.no_grad(), autocast_ctx:
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)

        new_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
        output_text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        latency = round(time.time() - t0, 3)
        parsed = extract_json_payload(output_text)
        parsed["_latency_s"] = latency
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _audio import DEFAULT_ACOUSTIC_PROMPT
    from _common.files import batch, destinations, identity, parser, read_json, request, write_json
    p = parser('Verify audio with hf; writes verdicts without filtering audio.', 'verify', 'hf')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--model-id', type=str, default='google/gemma-4-E2B-it')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--adapter-path', type=str, default=None)
    p.add_argument('--trust-remote-code', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--torch-dtype', type=str, default='bfloat16')
    args = p.parse_args()
    pairs = destinations(args, '_hf', '.json')
    parameters = {key: getattr(args, key) for key in ('model_id', 'device', 'adapter_path', 'trust_remote_code', 'torch_dtype')}
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else DEFAULT_ACOUSTIC_PROMPT
    with contextlib.redirect_stdout(sys.stderr):
        verifier = DefaultHFVerifier(**parameters)
    parameters['prompt'] = prompt
    # Record environment-derived model and endpoint values, never API credentials.
    for key in ('model', 'model_id', 'endpoint', 'gguf_variant', 'device'):
        if hasattr(verifier, key) and isinstance(getattr(verifier, key), (str, int, float, bool, type(None))):
            parameters[key] = getattr(verifier, key)
    def process(src, dest):
        wanted = request(identity(src), 'verify', parameters, 'hf')
        if dest.exists() and not args.overwrite:
            old = read_json(dest)
            if all(old.get(k) == v for k, v in wanted.items()) and 'verdict' in old:
                return
            raise ValueError(f'Conflicting output: {dest}; use --overwrite')
        verdict = verifier.verify(src, prompt)
        if not isinstance(verdict, dict) or verdict.get('decision') not in {'pass', 'reject'}:
            raise ValueError('Verifier did not return a pass/reject decision')
        write_json(dest, {**wanted, 'verdict': verdict})
    return batch(pairs, process)


if __name__ == '__main__':
    raise SystemExit(main())
