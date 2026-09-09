from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

from _audio import (
    extract_json_payload,
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
        if adapter_path:
            raise ValueError("Kimi inference does not support --adapter-path")
        self.model_id = model_id
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(f"Requested device is unavailable: {self.device}")
        logger.info("Initializing KimiAudioVerifier '%s' on %s...", model_id, self.device)

        repo_root = Path(__file__).resolve().parents[2]
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


def main() -> int:
    import argparse
    import contextlib
    from _audio import DEFAULT_ACOUSTIC_PROMPT
    from _common.files import batch, destinations, identity, parser, read_json, request, write_json
    p = parser('Verify audio with kimi; writes verdicts without filtering audio.', 'verify', 'kimi')
    p.add_argument('--prompt-file', type=Path)
    p.add_argument('--model-id', type=str, default='moonshotai/Kimi-Audio-7B-Instruct')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--adapter-path', type=str, default=None)
    args = p.parse_args()
    pairs = destinations(args, '_kimi', '.json')
    parameters = {key: getattr(args, key) for key in ('model_id', 'device', 'adapter_path')}
    prompt = args.prompt_file.read_text(encoding='utf-8') if args.prompt_file else DEFAULT_ACOUSTIC_PROMPT
    with contextlib.redirect_stdout(sys.stderr):
        verifier = KimiAudioVerifier(**parameters)
    parameters['prompt'] = prompt
    # Record environment-derived model and endpoint values, never API credentials.
    for key in ('model', 'model_id', 'endpoint', 'gguf_variant', 'device'):
        if hasattr(verifier, key) and isinstance(getattr(verifier, key), (str, int, float, bool, type(None))):
            parameters[key] = getattr(verifier, key)
    def process(src, dest):
        wanted = request(identity(src), 'verify', parameters, 'kimi')
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
