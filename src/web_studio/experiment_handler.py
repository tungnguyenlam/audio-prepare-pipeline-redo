"""Dedicated REST route handlers and task executor for the Experiment tab.

Coordinates the zero-contamination high-precision diarization pipeline,
direct-audio verifier probe/test API, and hardware telemetry.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import tempfile
import time
from typing import Any

from aiohttp import web
import soundfile as sf
import torch

from src.diarization.OverlapVerifier import (
    DEFAULT_GEMMA4_MODEL_ID,
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_UNSLOTH_ENDPOINT,
    GEMINI_AUDIO_MODELS,
    OVERLAP_PROMPT,
    OverlapVerifierError,
    create_overlap_verifier,
)
from src.diarization.zero_contamination import (
    DEFAULT_COLLAR_EROSION_S,
    DEFAULT_COMPETITOR_ONSET,
    DEFAULT_ENERGY_SEARCH_WINDOW_S,
    DEFAULT_ENERGY_VALLEY_FLOOR_DB,
    DEFAULT_HANDOFF_RISK_DISTANCE_S,
    DEFAULT_HOMOGENEITY_HOP_S,
    DEFAULT_HOMOGENEITY_WINDOW_S,
    DEFAULT_MIN_HOMOGENEITY_SIMILARITY,
    DEFAULT_MIN_TURN_DURATION_S,
    DEFAULT_SILENCE_TAIL_BUFFER_S,
    DEFAULT_TARGET_OFFSET,
    DEFAULT_TARGET_ONSET,
    DEFAULT_TRANSITION_EXCLUSION_S,
    ZeroContaminationConfig,
    run_zero_contamination_pipeline,
)
from src.utils.AudioClass import Audio

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / ".data"
DIARIZATION_RESULTS_DIR = DATA_DIR / "diarization" / "results"
DIARIZATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ExperimentRouteHandler:
    """Encapsulates Experiment tab routes with injected task manager and audio registry."""

    def __init__(self, task_manager: Any, registry: Any) -> None:
        self.task_manager = task_manager
        self.registry = registry

    def _resolve_device(self, req_device: str | None) -> str:
        if req_device and req_device != "auto":
            return req_device
        if torch.cuda.is_available():
            best_idx = max(
                range(torch.cuda.device_count()),
                key=lambda i: torch.cuda.get_device_properties(i).total_memory,
            )
            return f"cuda:{best_idx}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    async def handle_status(self, request: web.Request) -> web.Response:
        """Report available backends and default experiment configurations."""
        del request
        default_dev = self._resolve_device("auto")
        
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")

        backends = {
            "sortformer": True,
            "diarizen": True,
            "pyannote": True,
            "wespeaker_homogeneity": True,
            "whisper_timestamped": True,
            "vibevoice": True,
            "gemma4": True,
            "gemini": True,
        }
        defaults = {
            "primary_backend": "sortformer",
            "primary_device": default_dev,
            "target_onset": DEFAULT_TARGET_ONSET,
            "target_offset": DEFAULT_TARGET_OFFSET,
            "competitor_onset": DEFAULT_COMPETITOR_ONSET,
            "enable_consensus": True,
            "secondary_backend": "diarizen",
            "secondary_device": "same",
            "enable_collar_erosion": True,
            "boundary_collar_s": DEFAULT_COLLAR_EROSION_S,
            "min_turn_duration_s": DEFAULT_MIN_TURN_DURATION_S,
            "transition_exclusion_s": DEFAULT_TRANSITION_EXCLUSION_S,
            # Syllable & Boundary Integrity Gate
            "enable_context_collar": True,
            "handoff_risk_distance_s": DEFAULT_HANDOFF_RISK_DISTANCE_S,
            "silence_tail_buffer_s": DEFAULT_SILENCE_TAIL_BUFFER_S,
            "enable_syllable_alignment": True,
            "aligner_engine": "whisper_timestamped",
            "aligner_model": "vinai/PhoWhisper-large",
            "aligner_language": "vi",
            "aligner_device": "same",
            "aligner_endpoint": "",
            "enable_energy_snapping": False,
            "energy_search_window_s": DEFAULT_ENERGY_SEARCH_WINDOW_S,
            "energy_valley_floor_db": DEFAULT_ENERGY_VALLEY_FLOOR_DB,
            # Homogeneity
            "enable_homogeneity": True,
            "homogeneity_device": "same",
            "homogeneity_window_s": DEFAULT_HOMOGENEITY_WINDOW_S,
            "homogeneity_hop_s": DEFAULT_HOMOGENEITY_HOP_S,
            "min_homogeneity_similarity": DEFAULT_MIN_HOMOGENEITY_SIMILARITY,
            # Foundation Models
            "enable_gemma": True,
            "gemma_backend": "gemini",
            "gemma_endpoint": DEFAULT_UNSLOTH_ENDPOINT,
            "gemma_model": DEFAULT_GEMMA4_MODEL_ID,
            "gemini_model": DEFAULT_GEMINI_MODEL_ID,
            "gemini_models": list(GEMINI_AUDIO_MODELS),
            "gemma_prompt": OVERLAP_PROMPT,
            "gemma_max_output_tokens": 256,
            "enable_vibevoice": False,
            "vibevoice_model_id": "Dubedo/VibeVoice-ASR-HF-INT8",
            "vibevoice_device": "same",
            "device": default_dev,
        }
        return web.json_response({
            "status": "ready",
            "device": default_dev,
            "devices": devices,
            "backends": backends,
            "defaults": defaults,
        })

    async def handle_gemma_probe(self, request: web.Request) -> web.Response:
        """Report readiness for the selected direct-audio verifier."""
        try:
            body = await request.json() if request.can_read_body else {}
        except Exception:
            body = {}
        backend = str(body.get("backend") or "gemma4")
        model = body.get("model") or (
            DEFAULT_GEMINI_MODEL_ID if backend == "gemini" else DEFAULT_GEMMA4_MODEL_ID
        )
        config: dict[str, Any] = {"backend": backend, "model": model}
        if backend == "gemma4":
            config["endpoint"] = body.get("endpoint") or DEFAULT_UNSLOTH_ENDPOINT
        verifier = create_overlap_verifier(config)
        status = verifier.check_ready(timeout_s=4.0)
        status["backend"] = backend
        status["model"] = model
        return web.json_response(status)

    async def handle_gemma_test(self, request: web.Request) -> web.Response:
        """Run the selected direct-audio verifier on a file or selected range."""
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"Invalid JSON payload: {exc}"}, status=400)

        audio_id = body.get("audio_id")
        if not audio_id:
            return web.json_response({"error": "audio_id is required"}, status=400)

        audio = self.registry.get_audio(audio_id)
        if not audio or not Path(audio.path).is_file():
            return web.json_response({"error": "Audio track not found"}, status=404)

        backend = str(body.get("backend") or "gemma4")
        model = body.get("model") or (
            DEFAULT_GEMINI_MODEL_ID if backend == "gemini" else DEFAULT_GEMMA4_MODEL_ID
        )
        prompt = body.get("prompt") or None
        start_s = float(body.get("start_s", 0.0))
        end_s = float(body.get("end_s", 0.0))

        config: dict[str, Any] = {
            "backend": backend,
            "model": model,
            "prompt": prompt or OVERLAP_PROMPT,
            "max_output_tokens": int(body.get("max_output_tokens", 256)),
        }
        if backend == "gemma4":
            config["endpoint"] = body.get("endpoint") or DEFAULT_UNSLOTH_ENDPOINT
        verifier = create_overlap_verifier(config)

        loop = asyncio.get_running_loop()

        def do_test() -> dict[str, Any]:
            t_start = time.time()
            source_path = Path(audio.path)

            with tempfile.TemporaryDirectory(prefix="gemma-test-") as tmpdir:
                if end_s > start_s + 0.1:
                    waveform, sr = sf.read(str(source_path), dtype="float32", always_2d=False)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=1)
                    s_idx = max(0, int(round(start_s * sr)))
                    e_idx = min(len(waveform), int(round(end_s * sr)))
                    clip = waveform[s_idx:e_idx]
                    clip_path = Path(tmpdir) / "test_clip.wav"
                    sf.write(clip_path, clip, sr, subtype="PCM_16")
                    test_audio = Audio.from_file(clip_path)
                else:
                    test_audio = audio

                res = verifier.verify(test_audio)
                elapsed = time.time() - t_start
                return {
                    "overlap": res.get("overlap"),
                    "decision": res.get("decision"),
                    "speaker_purity": res.get("speaker_purity"),
                    "word_completeness": res.get("word_completeness"),
                    "boundary_issue": res.get("boundary_issue"),
                    "failure_codes": res.get("failure_codes"),
                    "reason": res.get("reason"),
                    "usage": res.get("usage"),
                    "cost": res.get("cost"),
                    "latency_s": round(elapsed, 2),
                    "tested_duration_s": round(test_audio.duration_s, 2),
                }

        try:
            result = await loop.run_in_executor(None, do_test)
            return web.json_response({"status": "ok", "result": result})
        except OverlapVerifierError as exc:
            return web.json_response({"error": str(exc), "readiness": exc.readiness}, status=502)
        except Exception as exc:
            logger.exception("Direct-audio test invocation failed")
            return web.json_response({"error": str(exc)}, status=500)

    async def handle_run_experiment(self, request: web.Request) -> web.Response:
        """Queue and run the zero-contamination diarization experiment in background."""
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"Invalid JSON payload: {exc}"}, status=400)

        audio_id = body.get("audio_id")
        if not audio_id:
            return web.json_response({"error": "audio_id is required"}, status=400)

        audio = self.registry.get_audio(audio_id)
        if not audio or not Path(audio.path).is_file():
            return web.json_response({"error": "Audio track not found"}, status=404)

        # Resolve per-step devices
        device = self._resolve_device(body.get("primary_device") or body.get("device"))
        primary_device = device

        sec_dev_req = body.get("secondary_device")
        secondary_device = (
            self._resolve_device(sec_dev_req)
            if sec_dev_req and sec_dev_req != "same"
            else primary_device
        )

        align_dev_req = body.get("aligner_device")
        aligner_device = (
            self._resolve_device(align_dev_req)
            if align_dev_req and align_dev_req != "same"
            else "cpu"
        )

        homo_dev_req = body.get("homogeneity_device")
        homo_device = (
            self._resolve_device(homo_dev_req)
            if homo_dev_req and homo_dev_req != "same"
            else primary_device
        )

        vv_dev_req = body.get("vibevoice_device")
        vibevoice_device = (
            self._resolve_device(vv_dev_req)
            if vv_dev_req and vv_dev_req != "same"
            else primary_device
        )

        token = body.get("token")

        config = ZeroContaminationConfig(
            primary_backend=body.get("primary_backend", "sortformer"),
            primary_device=primary_device,
            target_onset=float(body.get("target_onset", DEFAULT_TARGET_ONSET)),
            target_offset=float(body.get("target_offset", DEFAULT_TARGET_OFFSET)),
            competitor_onset=float(body.get("competitor_onset", DEFAULT_COMPETITOR_ONSET)),
            enable_consensus=bool(body.get("enable_consensus", True)),
            secondary_backend=body.get("secondary_backend", "diarizen"),
            secondary_device=secondary_device,
            enable_collar_erosion=bool(body.get("enable_collar_erosion", True)),
            boundary_collar_s=float(body.get("boundary_collar_s", DEFAULT_COLLAR_EROSION_S)),
            min_turn_duration_s=float(body.get("min_turn_duration_s", DEFAULT_MIN_TURN_DURATION_S)),
            transition_exclusion_s=float(body.get("transition_exclusion_s", DEFAULT_TRANSITION_EXCLUSION_S)),
            allow_gap_merge=bool(body.get("allow_gap_merge", False)),
            # Stage 3: Syllable & Boundary Integrity Gate
            enable_context_collar=bool(body.get("enable_context_collar", True)),
            handoff_risk_distance_s=float(body.get("handoff_risk_distance_s", DEFAULT_HANDOFF_RISK_DISTANCE_S)),
            silence_tail_buffer_s=float(body.get("silence_tail_buffer_s", DEFAULT_SILENCE_TAIL_BUFFER_S)),
            enable_energy_snapping=bool(body.get("enable_energy_snapping", False)),
            energy_search_window_s=float(body.get("energy_search_window_s", DEFAULT_ENERGY_SEARCH_WINDOW_S)),
            energy_valley_floor_db=float(body.get("energy_valley_floor_db", DEFAULT_ENERGY_VALLEY_FLOOR_DB)),
            enable_syllable_alignment=bool(body.get("enable_syllable_alignment", False)),
            aligner_engine=body.get("aligner_engine", "whisper_timestamped"),
            aligner_model=body.get("aligner_model", "vinai/PhoWhisper-small"),
            aligner_language=body.get("aligner_language", "vi"),
            aligner_endpoint=body.get("aligner_endpoint") or None,
            aligner_device=aligner_device,
            # Stage 4: Homogeneity
            enable_homogeneity=bool(body.get("enable_homogeneity", False)),
            homogeneity_device=homo_device,
            homogeneity_window_s=float(body.get("homogeneity_window_s", DEFAULT_HOMOGENEITY_WINDOW_S)),
            homogeneity_hop_s=float(body.get("homogeneity_hop_s", DEFAULT_HOMOGENEITY_HOP_S)),
            min_homogeneity_similarity=float(body.get("min_homogeneity_similarity", DEFAULT_MIN_HOMOGENEITY_SIMILARITY)),
            # Stage 5: Foundation Models
            enable_gemma=bool(body.get("enable_gemma", False)),
            gemma_backend=str(body.get("gemma_backend") or "gemma4"),
            gemma_endpoint=body.get("gemma_endpoint") or DEFAULT_UNSLOTH_ENDPOINT,
            gemma_model=body.get("gemma_model") or None,
            gemma_prompt=body.get("gemma_prompt") or None,
            # Credentials stay server-side and resolve from GEMINI_API_KEY or
            # UNSLOTH_API_KEY in the repository-root .env.
            gemma_api_key=None,
            gemma_timeout_s=float(body.get("gemma_timeout_s", 120.0)),
            gemma_max_output_tokens=int(body.get("gemma_max_output_tokens", 256)),
            enable_vibevoice=bool(body.get("enable_vibevoice", False)),
            vibevoice_model_id=body.get("vibevoice_model_id", "Dubedo/VibeVoice-ASR-HF-INT8"),
            vibevoice_device=vibevoice_device,
            vibevoice_endpoint=body.get("vibevoice_endpoint") or None,
            max_secondary_speech_s=float(body.get("max_secondary_speech_s", 0.0)),
            device=primary_device,
            token=token,
        )

        models_list = [f"{config.primary_backend} ({primary_device})"]
        if config.enable_consensus:
            models_list.append(f"{config.secondary_backend} ∩ ({secondary_device})")
        if config.enable_syllable_alignment:
            eng_lbl = "Whisper-VI" if "whisper" in config.aligner_engine.lower() else "MMS-FA"
            models_list.append(f"{eng_lbl} ({aligner_device})")
        if config.enable_energy_snapping:
            models_list.append("RMS-Snap")
        if config.enable_homogeneity:
            models_list.append(f"WeSpeaker ({homo_device})")
        if config.enable_gemma:
            models_list.append(
                f"{config.gemma_model or config.gemma_backend} direct audio"
            )
        if config.enable_vibevoice:
            models_list.append(f"VibeVoice ({vibevoice_device})")
        model_display = " + ".join(models_list)

        task_id = self.task_manager.create_task(
            "experiment_zero_contamination",
            {
                "title": f"Zero-Contamination: {audio.title or audio.source_id}",
                "model": model_display,
                "model_type": model_display,
                "backend": "zero_contamination",
                "audio_id": audio_id,
                "audio_title": audio.title or audio.source_id,
                "device": device,
                "queue_device": device,
                "config": config.to_dict(),
            },
        )

        async def run_pipeline_task():
            self.task_manager.update_task(
                task_id,
                status="running",
                progress=0.02,
                progress_known=True,
                message=f"Starting Zero-Contamination Experiment on {device}...",
            )
            loop = asyncio.get_running_loop()

            def progress_callback(progress: float, message: str) -> None:
                self.task_manager.update_task(
                    task_id,
                    progress=round(progress, 3),
                    progress_known=True,
                    message=message,
                )

            def execute() -> dict[str, Any]:
                exp_res = run_zero_contamination_pipeline(
                    audio, config, progress_callback=progress_callback
                )
                # Save DiarizationResult for durability
                saved_path = exp_res.diarization.save(DIARIZATION_RESULTS_DIR)
                res_dict = exp_res.to_dict()
                res_dict["saved_path"] = str(saved_path)
                res_dict["audio_id"] = audio_id
                res_dict["session_audio_id"] = audio_id
                return res_dict

            try:
                result_payload = await loop.run_in_executor(None, execute)
                self.task_manager.update_task(
                    task_id,
                    status="completed",
                    progress=1.0,
                    progress_known=True,
                    message="Zero-Contamination Experiment Complete!",
                    result=result_payload,
                )
            except Exception as exc:
                logger.exception("Experiment execution failed")
                self.task_manager.update_task(
                    task_id,
                    status="failed",
                    error=str(exc),
                    message=f"Experiment failed: {exc}",
                )

        self.task_manager.enqueue(task_id, run_pipeline_task)
        return web.json_response(
            {"task_id": task_id, "task": self.task_manager.get_task(task_id)},
            status=202,
        )


def register_experiment_routes(
    app: web.Application,
    task_manager: Any,
    registry: Any,
) -> None:
    """Mount all Experiment tab endpoints to the studio application."""
    handler = ExperimentRouteHandler(task_manager, registry)
    app.router.add_get("/api/experiment/status", handler.handle_status)
    app.router.add_post("/api/experiment/run", handler.handle_run_experiment)
    app.router.add_post("/api/experiment/direct-audio/probe", handler.handle_gemma_probe)
    app.router.add_post("/api/experiment/direct-audio/test", handler.handle_gemma_test)
    app.router.add_post("/api/experiment/gemma/probe", handler.handle_gemma_probe)
    app.router.add_post("/api/experiment/gemma/test", handler.handle_gemma_test)
    logger.info("Mounted dedicated Experiment tab routes at /api/experiment/*")
