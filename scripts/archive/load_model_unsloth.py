#!/usr/bin/env python3
"""Script to load a model in Unsloth Studio and wait for it to become ready."""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_model")

API_KEY = "sk-unsloth-d5e3632a095b7e04619fd36a650a9162"
BASE_URL = "http://127.0.0.1:8889"


def load_model(model_path: str, gguf_variant: str | None = None) -> bool:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model_path": model_path,
        "force_cancel_active": True,
    }
    if gguf_variant:
        payload["gguf_variant"] = gguf_variant

    logger.info("Requesting Unsloth to load %s (variant=%s)...", model_path, gguf_variant)
    req = urllib.request.Request(f"{BASE_URL}/v1/load", data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            data = json.loads(resp.read().decode())
            logger.info("Load response: %s", data.get("status"))
    except Exception as exc:
        logger.error("Failed to call /v1/load: %s", exc)
        return False

    # Poll status
    logger.info("Waiting for model to finish loading...")
    status_req = urllib.request.Request(f"{BASE_URL}/v1/status", headers=headers)
    for attempt in range(60):
        time.sleep(2)
        try:
            with urllib.request.urlopen(status_req, timeout=10.0) as resp:
                status_data = json.loads(resp.read().decode())
                active = status_data.get("active_model")
                loaded = status_data.get("loaded", [])
                curr_status = status_data.get("status")
                logger.info("Status [%d/60]: status=%s active=%s loaded=%s", attempt + 1, curr_status, active, loaded)
                if active == model_path or model_path in loaded or curr_status == "loaded":
                    logger.info("Model %s is successfully loaded and ready for inference!", model_path)
                    return True
        except Exception as exc:
            logger.warning("Status check error (attempt %d): %s", attempt + 1, exc)

    logger.error("Timed out waiting for model to load.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Load model into Unsloth Studio.")
    parser.add_argument("--model", required=True, help="Model identifier/path.")
    parser.add_argument("--variant", default=None, help="GGUF variant (e.g. Q8_0).")
    args = parser.parse_args()

    success = load_model(args.model, args.variant)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
