from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import logging
import sys
import time
from pathlib import Path
from typing import Any

from _audio import (
    parse_verifier_response,
)

logger = logging.getLogger("verifier.kimi_audio")


class KimiAudioVerifier:
    """Dedicated verifier for Moonshot Kimi-Audio models."""

    def __init__(
        self,
        model_id: str = "moonshotai/Kimi-Audio-7B-Instruct",
        device: str = "auto",
        adapter_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        import torch

        if adapter_path:
            raise ValueError("Kimi inference does not support --adapter-path")
        self.model_id = model_id
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device is unavailable: {self.device}")
        logger.info("Initializing KimiAudioVerifier '%s' on %s...", model_id, self.device)

        repo_root = Path(__file__).resolve().parents[3]
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
                    "Please run `./envs/setup_kimi_env.sh` to install Kimi-Audio dependencies into .venvs/kimi."
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
        parsed = parse_verifier_response(output_text)
        parsed["_latency_s"] = latency
        return parsed


def main() -> int:
    import contextlib
    from _cli import load_prompt, resolved_parameters, run_verifier
    from _common.files import destinations, parser
    p = parser('Verify audio with kimi; writes verdicts without filtering audio.', 's4-agent/verifier', 'kimi')
    p.add_argument('-pf', '-p', '--prompt-file', type=Path, help='Prompt text file (default: prompts/full-tags-prompt.md)')
    p.add_argument('-m', '--model-id', type=str, default='moonshotai/Kimi-Audio-7B-Instruct', help='Kimi-Audio model ID or local directory')
    p.add_argument('-d', '--device', type=str, default='auto', help='Inference device ("auto", "cpu", or "cuda:N")')
    p.add_argument('-ap', '--adapter-path', type=str, default=None, help='Optional LoRA adapter path')
    args = p.parse_args()
    pairs = destinations(args, '_kimi', '.json')
    parameters = {key: getattr(args, key) for key in ('model_id', 'device', 'adapter_path')}
    prompt = load_prompt(args.prompt_file)
    with contextlib.redirect_stdout(sys.stderr):
        verifier = KimiAudioVerifier(**parameters)
    parameters['prompt'] = prompt
    parameters = resolved_parameters(parameters, verifier)
    return run_verifier(
        args=args, pairs=pairs, backend='kimi', parameters=parameters,
        verify=lambda source: verifier.verify(source, prompt),
    )


if __name__ == '__main__':
    raise SystemExit(main())
