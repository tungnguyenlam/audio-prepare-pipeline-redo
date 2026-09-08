from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.diarization.verifiers.BaseVerifier import (
    BaseVerifier,
    extract_json_payload,
)

logger = logging.getLogger("verifier.kimi_audio")


class KimiAudioVerifier(BaseVerifier):
    """Dedicated verifier for Moonshot Kimi-Audio models."""

    def __init__(
        self,
        model_id: str = "moonshotai/Kimi-Audio-7B-Instruct",
        device: str = "auto",
        adapter_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_id = model_id
        self.device = (
            "cuda:0"
            if (device == "auto" and torch.cuda.is_available()) or device.startswith("cuda")
            else "cpu"
        )
        logger.info("Initializing KimiAudioVerifier '%s' on %s...", model_id, self.device)

        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        kimi_dir = repo_root / ".data" / "models" / "Kimi-Audio"
        if kimi_dir.is_dir() and str(kimi_dir) not in sys.path:
            sys.path.insert(0, str(kimi_dir))

        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError:
            try:
                from kimi_audio import KimiAudio
            except ImportError as e:
                raise ImportError(
                    f"KimiAudio could not be imported: {e}. "
                    "Please run `./scripts/setup_kimi_env.sh` to install Kimi-Audio dependencies into .venv-kimi."
                ) from e

        if self.device.startswith("cuda") and hasattr(torch.cuda, "set_device"):
            try:
                dev_idx = int(self.device.split(":")[-1]) if ":" in self.device else 0
                torch.cuda.set_device(dev_idx)
            except Exception:
                pass

        self.model = KimiAudio(model_path=model_id, load_detokenizer=False)
        logger.info("Successfully loaded KimiAudio model '%s'.", model_id)

    def verify(self, audio_path: Path, prompt: str) -> dict[str, Any]:
        t0 = time.time()
        chats = [
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
        ]
        try:
            _, text_output = self.model.generate(chats, output_type="text")
            output_text = text_output or ""
        except Exception as e:
            logger.debug("KimiAudio generate failed: %s", e)
            raise

        latency = round(time.time() - t0, 3)
        parsed = extract_json_payload(output_text)
        parsed["_latency_s"] = latency
        return parsed
