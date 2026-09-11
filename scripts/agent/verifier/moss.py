from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from _audio import (
    extract_json_payload,
    load_audio_waveform,
)

logger = logging.getLogger("verifier.moss_audio")


def _import_moss_classes():
    """Import MossAudioModel and MossAudioProcessor with informative error handling."""
    try:
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor
        return MossAudioModel, MossAudioProcessor
    except ImportError:
        pass

    # Search common clone paths if running outside installed site-packages
    candidate_dirs = [
        Path.home() / "Documents" / "MOSS-Audio",
        Path("/home/nguyenlt/Documents/MOSS-Audio"),
        Path(__file__).resolve().parents[3] / ".data" / "models" / "MOSS-Audio",
    ]

    for c_dir in candidate_dirs:
        if c_dir.is_dir():
            src_dir = c_dir / "src"
            if src_dir.is_dir() and str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            if str(c_dir) not in sys.path:
                sys.path.insert(0, str(c_dir))
            if "src" in sys.modules and hasattr(sys.modules["src"], "__path__"):
                if str(src_dir) not in sys.modules["src"].__path__:
                    sys.modules["src"].__path__.append(str(src_dir))
            try:
                from src.modeling_moss_audio import MossAudioModel
                from src.processing_moss_audio import MossAudioProcessor
                return MossAudioModel, MossAudioProcessor
            except ImportError:
                try:
                    from modeling_moss_audio import MossAudioModel
                    from processing_moss_audio import MossAudioProcessor
                    return MossAudioModel, MossAudioProcessor
                except ImportError:
                    pass

    raise ImportError(
        "MOSS-Audio backend requires the official OpenMOSS/MOSS-Audio package. "
        "Please clone https://github.com/OpenMOSS/MOSS-Audio or install it in your environment via: "
        "pip install -e \".[torch-runtime]\" in a dedicated Python 3.12 environment."
    )


class MossAudioVerifier:
    """Dedicated acoustic verifier for OpenMOSS-Team/MOSS-Audio models.

    Uses official MossAudioModel and MossAudioProcessor implementations
    instead of generic AutoModel classes.
    """

    def __init__(
        self,
        model_id: str = "OpenMOSS-Team/MOSS-Audio-8B-Thinking",
        device: str = "auto",
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        adapter_path: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        import torch

        MossAudioModel, MossAudioProcessor = _import_moss_classes()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        if (device == "auto" and torch.cuda.is_available()) or device.startswith("cuda"):
            self.device_map = "cuda:0" if device == "auto" else device
        elif device == "mps" or (device == "auto" and torch.backends.mps.is_available()):
            self.device_map = "mps"
        else:
            self.device_map = "cpu"

        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

        logger.info(
            "Loading MOSS-Audio '%s' on %s (dtype=%s, trust_remote_code=%s)...",
            model_id,
            self.device_map,
            self.dtype,
            trust_remote_code,
        )

        self.processor = MossAudioProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            enable_time_marker=True,
            token=self.hf_token,
        )

        self.model = MossAudioModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            dtype=self.dtype,
            device_map=self.device_map,
            token=self.hf_token,
        )
        self.model.eval()

        if adapter_path:
            from peft import PeftModel
            logger.info("Attaching LoRA adapter from '%s' to MOSS-Audio...", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()

        logger.info("Successfully loaded MOSS-Audio model '%s'.", model_id)

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        t0 = time.time()
        # Mel sample rate expected by MOSS-Audio processor config (default 16000 or 24000)
        sr = getattr(getattr(self.processor, "config", None), "mel_sr", 16000)
        audio_data = load_audio_waveform(audio_path, target_sr=sr)

        inputs = self.processor(text=prompt, audios=[audio_data], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)

        audio_input_mask = inputs["input_ids"] == self.processor.audio_token_id
        inputs["audio_input_mask"] = audio_input_mask

        import torch
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        output_text = self.processor.decode(
            generated_ids[0, input_len:],
            skip_special_tokens=True,
        ).strip()

        latency = round(time.time() - t0, 3)
        parsed = extract_json_payload(output_text)
        parsed["_latency_s"] = latency
        return parsed


def main() -> int:
    import argparse
    import contextlib
    from _cli import load_prompt, resolved_parameters, run_verifier
    from _common.files import destinations, parser
    p = parser('Verify audio with moss; writes verdicts without filtering audio.', 'agent/verifier', 'moss')
    p.add_argument('--prompt-file', type=Path, help='Optional path to text prompt file (defaults to acoustic defect prompt)')
    p.add_argument('--model-id', type=str, default='OpenMOSS-Team/MOSS-Audio-8B-Thinking', help='MOSS-Audio model ID or local directory')
    p.add_argument('--device', type=str, default='auto', help='Inference device ("auto", "cpu", or "cuda:N")')
    p.add_argument('--trust-remote-code', action=argparse.BooleanOptionalAction, default=True, help='Allow executing custom code from Hugging Face model repository')
    p.add_argument('--torch-dtype', type=str, default='bfloat16', help='PyTorch weights dtype ("bfloat16", "float16", or "float32")')
    p.add_argument('--adapter-path', type=str, default=None, help='Optional LoRA adapter directory')
    args = p.parse_args()
    pairs = destinations(args, '_moss', '.json')
    parameters = {key: getattr(args, key) for key in ('model_id', 'device', 'trust_remote_code', 'torch_dtype', 'adapter_path')}
    prompt = load_prompt(args.prompt_file)
    with contextlib.redirect_stdout(sys.stderr):
        verifier = MossAudioVerifier(**parameters)
    parameters['prompt'] = prompt
    parameters = resolved_parameters(parameters, verifier)
    return run_verifier(
        args=args, pairs=pairs, backend='moss', parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )


if __name__ == '__main__':
    raise SystemExit(main())
