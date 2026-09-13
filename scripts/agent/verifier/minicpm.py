from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import logging
import os
import time
from pathlib import Path
from typing import Any

from _audio import (
    load_audio_waveform,
    parse_verifier_response,
)

logger = logging.getLogger("verifier.minicpm")


class MiniCPMVerifier:
    """Dedicated verifier for MiniCPM-o models (e.g., openbmb/MiniCPM-o-4_5)."""

    def __init__(
        self,
        model_id: str = "openbmb/MiniCPM-o-4_5",
        device: str = "auto",
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        adapter_path: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        import torch

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

        logger.info("Loading MiniCPM-o '%s' on %s...", model_id, self.device)

        from transformers import AutoTokenizer, AutoModel, AutoProcessor
        try:
            self.processor = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=trust_remote_code, token=self.hf_token
            )
        except Exception as e:
            logger.debug("AutoTokenizer failed for %s (%s), falling back to AutoProcessor", model_id, e)
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code, token=self.hf_token
                )
            except Exception:
                self.processor = None

        extra_model_kwargs = {
            "init_vision": False,
            "init_audio": True,
            "init_tts": False,
        }
        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                attn_implementation="sdpa",
                torch_dtype=self.dtype,
                token=self.hf_token,
                **extra_model_kwargs,
            )
        except TypeError as e:
            logger.debug("Failed with sdpa/init kwargs (%s), falling back to standard AutoModel kwargs", e)
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.dtype,
                token=self.hf_token,
                **extra_model_kwargs,
            )

        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        if self.device.startswith("cuda") and hasattr(self.model, "cuda"):
            self.model = self.model.cuda(torch.device(self.device))
        elif hasattr(self.model, "to") and self.device != "cpu":
            self.model = self.model.to(self.device)

        if hasattr(self.model, "config") and hasattr(self.model.config, "stream_input"):
            self.model.config.stream_input = False

        if adapter_path:
            from peft import PeftModel
            logger.info("Attaching LoRA adapter from '%s' to MiniCPM...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        logger.info("Successfully loaded MiniCPM-o model '%s'.", model_id)

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        t0 = time.time()
        audio_data = load_audio_waveform(audio_path, target_sr=16000)
        msgs = [{"role": "user", "content": [prompt, audio_data]}]

        try:
            res = self.model.chat(
                msgs=msgs,
                do_sample=False,
                max_new_tokens=512,
                generate_audio=False,
                enable_thinking=False,
            )
        except TypeError:
            try:
                res = self.model.chat(
                    msgs=msgs,
                    do_sample=False,
                    max_new_tokens=512,
                    generate_audio=False,
                )
            except TypeError:
                try:
                    res = self.model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=self.processor,
                        generate_audio=False,
                    )
                except TypeError:
                    res = self.model.chat(image=None, audio=audio_data, msgs=msgs, tokenizer=self.processor)

        if isinstance(res, tuple):
            output_text = res[0] if len(res) > 0 else ""
        elif isinstance(res, str):
            output_text = res
        else:
            output_text = str(res)

        latency = round(time.time() - t0, 3)
        parsed = parse_verifier_response(output_text)
        parsed["_latency_s"] = latency
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _cli import load_prompt, resolved_parameters, run_verifier
    from _common.files import destinations, parser
    p = parser('Verify audio with minicpm; writes verdicts without filtering audio.', 'agent/verifier', 'minicpm')
    p.add_argument('--prompt-file', type=Path, help='Prompt text file (default: prompts/acoustic_defect-3.txt)')
    p.add_argument('--model-id', type=str, default='openbmb/MiniCPM-o-4_5', help='MiniCPM-o model ID or local directory')
    p.add_argument('--device', type=str, default='auto', help='Inference device ("auto", "cpu", or "cuda:N")')
    p.add_argument('--trust-remote-code', action=argparse.BooleanOptionalAction, default=True, help='Allow loading custom model code from Hugging Face')
    p.add_argument('--torch-dtype', type=str, default='bfloat16', help='PyTorch weights dtype ("bfloat16", "float16", or "float32")')
    p.add_argument('--adapter-path', type=str, default=None, help='Optional LoRA adapter directory')
    args = p.parse_args()
    pairs = destinations(args, '_minicpm', '.json')
    parameters = {key: getattr(args, key) for key in ('model_id', 'device', 'trust_remote_code', 'torch_dtype', 'adapter_path')}
    prompt = load_prompt(args.prompt_file)
    with contextlib.redirect_stdout(sys.stderr):
        verifier = MiniCPMVerifier(**parameters)
    parameters['prompt'] = prompt
    parameters = resolved_parameters(parameters, verifier)
    return run_verifier(
        args=args, pairs=pairs, backend='minicpm', parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )


if __name__ == '__main__':
    raise SystemExit(main())
