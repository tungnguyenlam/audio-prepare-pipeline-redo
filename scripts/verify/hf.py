from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging
import os
import time
from typing import Any

import torch

from _audio import (
    DEFAULT_ACOUSTIC_PROMPT,
    extract_json_payload,
    load_audio_waveform,
)

logger = logging.getLogger("verifier.default_hf")


class DefaultHFVerifier:
    """Default Hugging Face multimodal audio verifier (e.g., Gemma 4 E2B/E4B/12B, standard HF models)."""

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

        quant_kwargs: dict[str, Any] = {}
        if load_in_4bit or load_in_8bit:
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
                    **quant_kwargs,
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

    def generate(
        self,
        audio_path: Path,
        prompt: str,
        *,
        system_prompt: str | None = None,
        audio_position: str = "before",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        t0 = time.time()
        audio_data = load_audio_waveform(audio_path, target_sr=16000)

        audio_part = {"type": "audio", "audio": audio_data}
        prompt_part = {"type": "text", "text": prompt}
        content = (
            [audio_part, prompt_part]
            if audio_position == "before"
            else [prompt_part, audio_part]
        )
        messages = []
        if system_prompt is not None:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        messages.append({"role": "user", "content": content})
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=text, audio=audio_data, return_tensors="pt", sampling_rate=16000)
        else:
            inputs = self.processor(text=prompt, audio=audio_data, return_tensors="pt", sampling_rate=16000)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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
            "provider_body": {
                "generated_text": output_text,
                "generated_token_count": int(new_tokens.shape[-1]),
            },
        }

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        generated = self.generate(audio_path, prompt)
        parsed = extract_json_payload(generated["text"].strip())
        parsed["_latency_s"] = generated["latency_s"]
        parsed["_engine"] = "huggingface"
        parsed["_model"] = self.model_id
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _common.files import batch, destinations, identity, parser, read_json, request, write_json

    p = parser('Verify audio with hf; writes verdicts without filtering audio.', 'verify', 'hf')
    p.add_argument('--prompt-file', type=Path, help='Path to prompt text file')
    p.add_argument('--model-id', type=str, default='google/gemma-4-E2B-it', help='Hugging Face model repository ID')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--adapter-path', type=str, default=None, help='Optional LoRA adapter checkpoint directory')
    p.add_argument('--trust-remote-code', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--torch-dtype', type=str, default='bfloat16', choices=('bfloat16', 'float16', 'float32'))
    p.add_argument('--load-in-4bit', action='store_true', help='Load model in 4-bit NF4 with bitsandbytes')
    p.add_argument('--load-in-8bit', action='store_true', help='Load model in 8-bit with bitsandbytes')
    args = p.parse_args()

    pairs = destinations(args, '_hf', '.json')
    parameters = {
        key: getattr(args, key) for key in (
            'model_id', 'device', 'adapter_path', 'trust_remote_code',
            'torch_dtype', 'load_in_4bit', 'load_in_8bit'
        )
    }

    if args.prompt_file and args.prompt_file.is_file():
        prompt = args.prompt_file.read_text(encoding='utf-8').strip()
    elif Path('prompts/acoustic_defect.txt').is_file():
        prompt = Path('prompts/acoustic_defect.txt').read_text(encoding='utf-8').strip()
    else:
        prompt = DEFAULT_ACOUSTIC_PROMPT

    with contextlib.redirect_stdout(sys.stderr):
        verifier = DefaultHFVerifier(**parameters)
    parameters['prompt'] = prompt

    for key in ('model', 'model_id', 'endpoint', 'gguf_variant', 'device'):
        if hasattr(verifier, key) and isinstance(getattr(verifier, key), (str, int, float, bool, type(None))):
            parameters[key] = getattr(verifier, key)

    def process(src: Path, dest: Path) -> None:
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
