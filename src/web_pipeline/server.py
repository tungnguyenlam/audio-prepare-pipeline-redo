"""Fast, Asynchronous REST & SSE API Server for SonicPipeline."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import mimetypes
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
import soundfile as sf
import torch
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".data" / "huggingface"))

from src.yt_crawler.YtCrawlerClass import parse_crawl_sample_rate
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio
from src.web_pipeline.batch_processors import register_all_handlers
from src.web_pipeline.dataset_manager import dataset_manager
from src.web_pipeline.hardware_monitor import hardware_monitor
from src.web_pipeline.queue_manager import queue_manager

logger = logging.getLogger("pipeline_server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = ROOT_DIR / ".data" / "pipeline"
UPLOADS_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = DATA_DIR / "exports"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def json_response(data: Any, status: int = 200) -> web.Response:
    """Helper to return JSON response."""
    return web.json_response(data, status=status)


def json_error(message: str, status: int = 400) -> web.Response:
    """Helper to return JSON error response."""
    return web.json_response({"error": message, "status": status}, status=status)


# -------------------------------------------------------------------------
# Telemetry & Real-Time SSE Stream
# -------------------------------------------------------------------------

async def handle_events_sse(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events (SSE) stream for real-time dashboard events."""
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
    await response.prepare(request)

    q = queue_manager.subscribe()
    try:
        # Initial telemetry snapshot
        init_payload = json.dumps({
            "event": "telemetry",
            "data": hardware_monitor.get_system_telemetry(),
            "timestamp": time.time(),
        })
        await response.write(f"data: {init_payload}\n\n".encode("utf-8"))

        while True:
            try:
                # Wait for broadcast event or 3-second heartbeat telemetry
                msg = await asyncio.wait_for(q.get(), timeout=3.0)
                payload = json.dumps(msg)
                await response.write(f"data: {payload}\n\n".encode("utf-8"))
            except asyncio.TimeoutError:
                # Telemetry heartbeat
                tele_payload = json.dumps({
                    "event": "telemetry",
                    "data": hardware_monitor.get_system_telemetry(),
                    "timestamp": time.time(),
                })
                await response.write(f"data: {tele_payload}\n\n".encode("utf-8"))
    except (asyncio.CancelledError, ConnectionResetError, aiohttp.ClientConnectionResetError):
        pass
    finally:
        queue_manager.unsubscribe(q)

    return response


# -------------------------------------------------------------------------
# Dataset & Audio Item Handlers
# -------------------------------------------------------------------------

async def handle_list_datasets(request: web.Request) -> web.Response:
    """List all dataset collections."""
    return json_response(dataset_manager.list_datasets())


async def handle_list_channels(request: web.Request) -> web.Response:
    """List channel-scoped audio groups and processing coverage."""
    return json_response({"channels": dataset_manager.list_channels()})


async def handle_create_dataset(request: web.Request) -> web.Response:
    """Create a new dataset collection."""
    try:
        body = await request.json()
        name = body.get("name", "").strip()
        desc = body.get("description", "").strip()
        tags = body.get("tags", [])
        ds = dataset_manager.create_dataset(name, desc, tags)
        queue_manager.broadcast("datasets_updated", {"action": "create", "dataset": ds})
        return json_response(ds)
    except Exception as e:
        return json_error(str(e))


async def handle_delete_dataset(request: web.Request) -> web.Response:
    """Delete a dataset collection."""
    name = request.match_info["name"]
    delete_items = request.query.get("delete_items", "false").lower() == "true"
    success = dataset_manager.delete_dataset(name, delete_items=delete_items)
    if success:
        queue_manager.broadcast("datasets_updated", {"action": "delete", "name": name})
        return json_response({"success": True})
    return json_error("Dataset not found", 404)


async def handle_list_items(request: web.Request) -> web.Response:
    """Search and filter audio items."""
    dataset = request.query.get("dataset")
    query = request.query.get("query")
    tag = request.query.get("tag")
    channel_id = request.query.get("channel_id")
    has_stems_param = request.query.get("has_stems")
    has_diar_param = request.query.get("has_diarization")
    min_dur = float(request.query.get("min_duration")) if request.query.get("min_duration") else None
    max_dur = float(request.query.get("max_duration")) if request.query.get("max_duration") else None
    sort_by = request.query.get("sort_by", "created_at")
    sort_desc = request.query.get("sort_desc", "true").lower() == "true"
    limit = int(request.query.get("limit", 200))
    offset = int(request.query.get("offset", 0))

    has_stems = None
    if has_stems_param is not None:
        has_stems = has_stems_param.lower() == "true"

    has_diar = None
    if has_diar_param is not None:
        has_diar = has_diar_param.lower() == "true"

    res = dataset_manager.list_items(
        dataset=dataset,
        query=query,
        tag=tag,
        channel_id=channel_id,
        has_stems=has_stems,
        has_diarization=has_diar,
        min_duration=min_dur,
        max_duration=max_dur,
        sort_by=sort_by,
        sort_desc=sort_desc,
        limit=limit,
        offset=offset,
    )
    return json_response(res)


async def handle_get_item(request: web.Request) -> web.Response:
    """Get single audio item detail."""
    item_id = request.match_info["id"]
    item = dataset_manager.get_item(item_id)
    if not item:
        return json_error("Item not found", 404)
    return json_response(item.to_dict())


async def handle_update_item(request: web.Request) -> web.Response:
    """Update item metadata or tags."""
    item_id = request.match_info["id"]
    try:
        body = await request.json()
        success = dataset_manager.update_item(item_id, body)
        if success:
            updated = dataset_manager.get_item(item_id)
            queue_manager.broadcast("item_updated", updated.to_dict() if updated else {})
            return json_response({"success": True, "item": updated.to_dict() if updated else None})
        return json_error("Item not found", 404)
    except Exception as e:
        return json_error(str(e))


async def handle_delete_items(request: web.Request) -> web.Response:
    """Bulk delete audio items."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids", [])
        delete_files = bool(body.get("delete_files", False))
        count = dataset_manager.delete_items(item_ids, delete_files=delete_files)
        queue_manager.broadcast("items_deleted", {"deleted_count": count, "item_ids": item_ids})
        return json_response({"deleted_count": count})
    except Exception as e:
        return json_error(str(e))


async def handle_bulk_tag(request: web.Request) -> web.Response:
    """Bulk tag items."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids", [])
        add_tags = body.get("add_tags", [])
        remove_tags = body.get("remove_tags", [])
        count = dataset_manager.bulk_tag(item_ids, add_tags, remove_tags)
        queue_manager.broadcast("items_tagged", {"affected": count})
        return json_response({"affected": count})
    except Exception as e:
        return json_error(str(e))


async def handle_bulk_dataset(request: web.Request) -> web.Response:
    """Bulk move items to dataset."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids", [])
        target_dataset = body.get("dataset", "Default")
        count = dataset_manager.bulk_assign_dataset(item_ids, target_dataset)
        queue_manager.broadcast("items_moved", {"affected": count, "target_dataset": target_dataset})
        return json_response({"affected": count})
    except Exception as e:
        return json_error(str(e))


async def handle_stream_audio(request: web.Request) -> web.StreamResponse:
    """Stream audio with HTTP range support for playback in UI."""
    item_id = request.match_info["id"]
    item = dataset_manager.get_item(item_id)
    if not item or not Path(item.path).exists():
        return web.Response(status=404, text="Audio file not found")

    file_path = Path(item.path)
    return web.FileResponse(file_path)


async def handle_stream_stem(request: web.Request) -> web.StreamResponse:
    """Stream separated stem audio file."""
    item_id = request.match_info["id"]
    model_name = request.match_info["model"]
    stem_name = request.match_info["stem"]

    item = dataset_manager.get_item(item_id)
    if not item:
        return web.Response(status=404, text="Audio item not found")

    model_stems = item.stems.get(model_name, {})
    stem_path_str = model_stems.get(stem_name)
    if not stem_path_str or not Path(stem_path_str).exists():
        return web.Response(status=404, text="Stem not found")

    return web.FileResponse(Path(stem_path_str))


async def handle_download_audio(request: web.Request) -> web.StreamResponse:
    """Download master audio file."""
    item_id = request.match_info["id"]
    item = dataset_manager.get_item(item_id)
    if not item or not Path(item.path).exists():
        return web.Response(status=404, text="Audio file not found")

    file_path = Path(item.path)
    return web.FileResponse(
        file_path,
        headers={"Content-Disposition": f'attachment; filename="{file_path.name}"'},
    )


async def handle_download_stem(request: web.Request) -> web.StreamResponse:
    """Download separated stem audio file."""
    item_id = request.match_info["id"]
    model_name = request.match_info["model"]
    stem_name = request.match_info["stem"]

    item = dataset_manager.get_item(item_id)
    if not item:
        return web.Response(status=404, text="Audio item not found")

    model_stems = item.stems.get(model_name, {})
    stem_path_str = model_stems.get(stem_name)
    if not stem_path_str or not Path(stem_path_str).exists():
        return web.Response(status=404, text="Stem not found")

    target_path = Path(stem_path_str)
    return web.FileResponse(
        target_path,
        headers={"Content-Disposition": f'attachment; filename="{target_path.name}"'},
    )


# -------------------------------------------------------------------------
# Job Queue Endpoints
# -------------------------------------------------------------------------

async def handle_list_jobs(request: web.Request) -> web.Response:
    """List queue jobs."""
    status_filter = request.query.get("status")
    limit = int(request.query.get("limit", 50))
    return json_response(queue_manager.list_jobs(limit=limit, status_filter=status_filter))


async def handle_get_job(request: web.Request) -> web.Response:
    """Get single job status and logs."""
    job_id = request.match_info["id"]
    job = queue_manager.get_job(job_id)
    if not job:
        return json_error("Job not found", 404)
    return json_response(job.to_dict())


async def handle_cancel_job(request: web.Request) -> web.Response:
    """Cancel a running or pending job."""
    job_id = request.match_info["id"]
    success = queue_manager.cancel_job(job_id)
    if success:
        return json_response({"success": True})
    return json_error("Could not cancel job (it may already be finished)", 400)


async def handle_delete_job(request: web.Request) -> web.Response:
    """Delete a job record."""
    job_id = request.match_info["id"]
    success = queue_manager.delete_job(job_id)
    if success:
        return json_response({"success": True})
    return json_error("Job not found", 404)


async def handle_queue_controls(request: web.Request) -> web.Response:
    """Control queue execution (pause, resume, concurrency)."""
    try:
        body = await request.json()
        action = body.get("action")
        if action == "pause":
            queue_manager.pause_queue()
        elif action == "resume":
            queue_manager.resume_queue()
        elif action == "set_concurrency":
            concurrency = int(body.get("concurrency", 1))
            queue_manager.set_concurrency(concurrency)

        return json_response({
            "is_paused": queue_manager.is_paused,
            "max_concurrency": queue_manager.max_concurrency,
            "workers_per_device": queue_manager.workers_per_device,
            "device_queues": queue_manager.status().get("device_queues", {}),
        })
    except Exception as e:
        return json_error(str(e))


# -------------------------------------------------------------------------
# Job Submission Endpoints
# -------------------------------------------------------------------------

async def handle_submit_ingest_yt(request: web.Request) -> web.Response:
    """Submit YouTube batch ingestion job."""
    try:
        body = await request.json()
        urls = body.get("urls", [])
        dataset = body.get("dataset", "Default")
        group_by_channel = bool(body.get("group_by_channel", True))
        tags = body.get("tags", [])
        try:
            sample_rate = parse_crawl_sample_rate(body.get("sample_rate", DEFAULT_SAMPLE_RATE))
        except (TypeError, ValueError):
            return json_error("sample_rate must be 'native', 16000, or 44100")
        if sample_rate not in (None, 16000, DEFAULT_SAMPLE_RATE):
            return json_error("sample_rate must be 'native', 16000, or 44100")
        max_duration = body.get("max_duration_seconds")

        if isinstance(urls, str):
            urls = [u.strip() for u in urls.replace(",", "\n").splitlines() if u.strip()]

        if not urls:
            return json_error("No YouTube URLs provided")

        rate_label = "native" if sample_rate is None else f"{sample_rate}Hz"
        title = f"YouTube Batch Ingest ({len(urls)} items, {rate_label}) -> {dataset}"
        job = queue_manager.submit_job(
            job_type="batch_ingest_yt",
            title=title,
            params={
                "urls": urls,
                "dataset": dataset,
                "group_by_channel": group_by_channel,
                "tags": tags,
                "sample_rate": sample_rate,
                "max_duration_seconds": max_duration,
            },
            total_items=len(urls),
        )
        return json_response(job.to_dict())
    except Exception as e:
        return json_error(str(e))


async def handle_submit_ingest_files(request: web.Request) -> web.Response:
    """Submit local file scan batch ingestion job."""
    try:
        body = await request.json()
        scan_dir = body.get("scan_directory")
        file_paths = body.get("file_paths", [])
        dataset = body.get("dataset", "Default")
        tags = body.get("tags", [])
        sample_rate = int(body.get("sample_rate", DEFAULT_SAMPLE_RATE))

        if not scan_dir and not file_paths:
            return json_error("Must specify scan_directory or file_paths")

        title = f"Local Batch Ingest ({scan_dir or len(file_paths)} files) -> {dataset}"
        job = queue_manager.submit_job(
            job_type="batch_ingest_files",
            title=title,
            params={
                "scan_directory": scan_dir,
                "file_paths": file_paths,
                "dataset": dataset,
                "tags": tags,
                "sample_rate": sample_rate,
            },
            total_items=len(file_paths) if file_paths else 1,
        )
        return json_response(job.to_dict())
    except Exception as e:
        return json_error(str(e))


async def handle_upload_batch_files(request: web.Request) -> web.Response:
    """Handle multipart file uploads and trigger batch ingestion."""
    reader = await request.multipart()
    saved_files: List[str] = []
    dataset = "Default"
    tags: List[str] = []

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "dataset":
            dataset = (await part.text()).strip() or "Default"
        elif part.name == "tags":
            tag_str = await part.text()
            tags = [t.strip() for t in tag_str.split(",") if t.strip()]
        elif part.filename:
            safe_name = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{part.filename}"
            target_path = UPLOADS_DIR / safe_name
            with open(target_path, "wb") as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)
            saved_files.append(str(target_path))

    if not saved_files:
        return json_error("No audio files were uploaded")

    job = queue_manager.submit_job(
        job_type="batch_ingest_files",
        title=f"Uploaded Batch ({len(saved_files)} files) -> {dataset}",
        params={
            "file_paths": saved_files,
            "dataset": dataset,
            "tags": tags,
        },
        total_items=len(saved_files),
    )
    return json_response(job.to_dict())


async def handle_submit_separation(request: web.Request) -> web.Response:
    """Submit batch separation job."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids", [])
        dataset = body.get("dataset")
        channel_id = body.get("channel_id")
        model = body.get("model", "BSRoFormer")
        device = body.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if not item_ids and (dataset or channel_id or "dataset" in body):
            target_ds = None if (not dataset or dataset == "all") else dataset
            matched = dataset_manager.list_items(
                dataset=target_ds,
                channel_id=None if channel_id in (None, "", "all") else channel_id,
                limit=10000,
            )
            item_ids = [it["id"] for it in matched["items"]]

        if not item_ids:
            return json_error("No audio items selected for separation")

        title = f"Batch Separation ({len(item_ids)} items) [{model}]"
        job = queue_manager.submit_job(
            job_type="batch_separation",
            title=title,
            params={
                "item_ids": item_ids,
                "model": model,
                "device": device,
            },
            total_items=len(item_ids),
        )
        return json_response(job.to_dict())
    except Exception as e:
        return json_error(str(e))


async def handle_submit_diarization(request: web.Request) -> web.Response:
    """Submit batch diarization job."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids", [])
        dataset = body.get("dataset")
        channel_id = body.get("channel_id")
        backend = body.get("backend", "sortformer")
        device = body.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        num_speakers = body.get("num_speakers")
        min_speakers = body.get("min_speakers")
        max_speakers = body.get("max_speakers")
        hf_token = body.get("hf_token")
        include_overlap = bool(body.get("include_overlap", False))
        vad_onset = body.get("vad_onset", 0.5)
        vad_offset = body.get("vad_offset", 0.3)
        chunk_duration_s = body.get("chunk_duration_s", 1.5)
        chunk_step_s = body.get("chunk_step_s", 0.75)

        if not item_ids and (dataset or channel_id or "dataset" in body):
            target_ds = None if (not dataset or dataset == "all") else dataset
            matched = dataset_manager.list_items(
                dataset=target_ds,
                channel_id=None if channel_id in (None, "", "all") else channel_id,
                limit=10000,
            )
            item_ids = [it["id"] for it in matched["items"]]

        if not item_ids:
            return json_error("No audio items selected for diarization")

        title = f"Batch Diarization ({len(item_ids)} items) [{backend}]"
        job = queue_manager.submit_job(
            job_type="batch_diarization",
            title=title,
            params={
                "item_ids": item_ids,
                "backend": backend,
                "device": device,
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "hf_token": hf_token,
                "include_overlap": include_overlap,
                "vad_onset": vad_onset,
                "vad_offset": vad_offset,
                "chunk_duration_s": chunk_duration_s,
                "chunk_step_s": chunk_step_s,
            },
            total_items=len(item_ids),
        )
        return json_response(job.to_dict())
    except Exception as e:
        return json_error(str(e))


async def handle_submit_target_speaker(request: web.Request) -> web.Response:
    """Submit a target speaker verification / filtering job."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids", [])
        dataset = body.get("dataset")
        channel_id = body.get("channel_id")
        profile = body.get("profile")
        threshold = float(body.get("threshold", 0.6))
        min_duration_s = float(body.get("min_duration_s", 1.5))
        exclude_overlap = bool(body.get("exclude_overlap", True))
        export_cuts = bool(body.get("export_cuts", False))
        device = body.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        hf_token = body.get("hf_token")

        if not profile:
            return json_error("A speaker profile name is required")

        if not item_ids and (dataset or channel_id or "dataset" in body):
            target_ds = None if (not dataset or dataset == "all") else dataset
            matched = dataset_manager.list_items(
                dataset=target_ds,
                channel_id=None if channel_id in (None, "", "all") else channel_id,
                limit=10000,
            )
            item_ids = [it["id"] for it in matched["items"]]

        if not item_ids:
            return json_error("No audio items selected for target speaker filtering")

        title = f"Target Speaker Filter ({len(item_ids)} items) [{profile}]"
        job = queue_manager.submit_job(
            job_type="target_speaker_filter",
            title=title,
            params={
                "item_ids": item_ids,
                "profile": profile,
                "channel_id": channel_id,
                "threshold": threshold,
                "min_duration_s": min_duration_s,
                "exclude_overlap": exclude_overlap,
                "export_cuts": export_cuts,
                "device": device,
                "hf_token": hf_token,
            },
            total_items=len(item_ids),
        )
        return json_response(job.to_dict())
    except Exception as e:
        return json_error(str(e))


async def handle_list_speaker_profiles(request: web.Request) -> web.Response:
    """List globally reusable enrolled speaker profiles."""
    from src.diarization.SpeakerVerifier import SpeakerVerifier, SpeakerVerifierError

    verifier = SpeakerVerifier()
    profiles = []
    for name in verifier.list_profiles():
        try:
            p = verifier.load_profile(name)
            profiles.append(
                {
                    "name": p.name,
                    "num_clips": len(p.clip_paths),
                    "created_at": p.created_at,
                    "updated_at": p.updated_at,
                    "channel_id": p.channel_id,
                    "channel_name": p.channel_name,
                    "channel_url": p.channel_url,
                }
            )
        except SpeakerVerifierError:
            continue
    return json_response({"profiles": profiles})


async def handle_submit_benchmark(request: web.Request) -> web.Response:
    """Submit separation benchmark job across speech and music pools."""
    try:
        body = await request.json()
        speech_item_ids = body.get("speech_item_ids", [])
        music_item_ids = body.get("music_item_ids", [])
        speech_dataset = body.get("speech_dataset")
        music_dataset = body.get("music_dataset")
        models = body.get("models", ["BSRoFormer", "HTDemucs"])
        snr_levels = body.get("snr_levels", [0.0, 6.0, 12.0])
        device = body.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if not speech_item_ids and speech_dataset:
            sp_res = dataset_manager.list_items(dataset=speech_dataset, limit=500)
            speech_item_ids = [x["id"] for x in sp_res["items"]]

        if not music_item_ids and music_dataset:
            mus_res = dataset_manager.list_items(dataset=music_dataset, limit=500)
            music_item_ids = [x["id"] for x in mus_res["items"]]

        if not speech_item_ids or not music_item_ids:
            return json_error("Must provide at least one speech item and one music item for benchmark")

        title = f"Benchmark Matrix: {len(speech_item_ids)} speech x {len(music_item_ids)} music ({len(models)} models)"
        job = queue_manager.submit_job(
            job_type="batch_benchmark",
            title=title,
            params={
                "speech_item_ids": speech_item_ids,
                "music_item_ids": music_item_ids,
                "models": models,
                "snr_levels": snr_levels,
                "device": device,
            },
            total_items=len(speech_item_ids) * len(music_item_ids) * len(snr_levels) * len(models),
        )
        return json_response(job.to_dict())
    except Exception as e:
        return json_error(str(e))


async def handle_generate_manifest(request: web.Request) -> web.Response:
    """Generate manifest string (JSONL or CSV)."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids")
        dataset = body.get("dataset")
        fmt = body.get("format", "jsonl").lower()

        manifest_content = dataset_manager.generate_manifest(
            item_ids=item_ids,
            dataset=dataset,
            format_type=fmt,
        )

        content_type = "text/csv" if fmt == "csv" else "application/x-ndjson"
        return web.Response(
            text=manifest_content,
            content_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=manifest.{fmt}",
            },
        )
    except Exception as e:
        return json_error(str(e))


async def handle_create_export(request: web.Request) -> web.Response:
    """Create a ZIP bundle of dataset items, stems, and manifests."""
    try:
        body = await request.json()
        item_ids = body.get("item_ids")
        dataset = body.get("dataset")
        include_stems = bool(body.get("include_stems", True))
        include_manifests = bool(body.get("include_manifests", True))

        zip_path = dataset_manager.create_export_bundle(
            item_ids=item_ids,
            dataset=dataset,
            include_stems=include_stems,
            include_manifests=include_manifests,
        )

        return json_response({
            "export_id": zip_path.name,
            "download_url": f"/api/exports/download/{zip_path.name}",
            "size_mb": round(os.path.getsize(zip_path) / (1024 * 1024), 2),
        })
    except Exception as e:
        return json_error(str(e))


async def handle_download_export(request: web.Request) -> web.StreamResponse:
    """Download export ZIP bundle."""
    filename = request.match_info["filename"]
    target_file = (EXPORTS_DIR / filename).resolve()
    try:
        target_file.relative_to(EXPORTS_DIR.resolve())
    except ValueError:
        return web.Response(status=403, text="Invalid export path")
    if not target_file.exists():
        return web.Response(status=404, text="Export archive not found")
    return web.FileResponse(target_file)


# -------------------------------------------------------------------------
# Benchmark Reports
# -------------------------------------------------------------------------

async def handle_list_benchmarks(request: web.Request) -> web.Response:
    """List saved benchmark evaluation reports."""
    reports = []
    for f in BENCHMARK_DIR.glob("*_report.json"):
        try:
            with open(f, "r", encoding="utf-8") as rf:
                data = json.load(rf)
                reports.append({
                    "job_id": data.get("job_id"),
                    "timestamp": data.get("timestamp"),
                    "models": data.get("models", []),
                    "leaderboard": data.get("leaderboard", {}),
                    "total_runs": len(data.get("runs", [])),
                })
        except Exception:
            pass
    reports.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return json_response(reports)


async def handle_get_benchmark(request: web.Request) -> web.Response:
    """Get single benchmark report."""
    job_id = request.match_info["id"]
    report_file = (BENCHMARK_DIR / f"{job_id}_report.json").resolve()
    try:
        report_file.relative_to(BENCHMARK_DIR.resolve())
    except ValueError:
        return json_error("Invalid benchmark ID", 400)
    if not report_file.exists():
        return json_error("Benchmark report not found", 404)
    with open(report_file, "r", encoding="utf-8") as f:
        return json_response(json.load(f))


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Static File & Single-Page App
# -------------------------------------------------------------------------

@web.middleware
async def no_cache_middleware(request: web.Request, handler):
    """Ensure static assets and HTML are never cached stale during development."""
    response = await handler(request)
    if request.path.startswith("/static/") or request.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


async def handle_index(request: web.Request) -> web.Response:
    """Serve the SonicPipeline frontend index.html."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return web.Response(status=404, text="Static frontend not found")
    return web.Response(
        text=index_path.read_text(encoding="utf-8"),
        content_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"},
    )


def register_lifecycle(app: web.Application) -> None:
    """Register SonicPipeline background services on an application.

    Args:
        app: Aiohttp application that owns the shared backend.
    """
    # Register batch job execution handlers
    register_all_handlers(queue_manager)

    # Lifecycle hooks
    async def on_startup(app: web.Application) -> None:
        await queue_manager.start()
        logger.info("SonicPipeline server initialized and queue manager active.")

    async def on_shutdown(app: web.Application) -> None:
        await queue_manager.stop()
        logger.info("SonicPipeline server stopped.")

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)


def register_api_routes(app: web.Application) -> None:
    """Register SonicPipeline API routes on an application.

    Args:
        app: Aiohttp application that owns the shared backend.
    """
    app.router.add_get("/api/events", handle_events_sse)

    # Dataset & item routes
    app.router.add_get("/api/datasets", handle_list_datasets)
    app.router.add_get("/api/channels", handle_list_channels)
    app.router.add_post("/api/datasets", handle_create_dataset)
    app.router.add_delete("/api/datasets/{name}", handle_delete_dataset)
    app.router.add_get("/api/items", handle_list_items)
    app.router.add_get("/api/items/{id}", handle_get_item)
    app.router.add_patch("/api/items/{id}", handle_update_item)
    app.router.add_post("/api/items/delete", handle_delete_items)
    app.router.add_post("/api/items/bulk_tag", handle_bulk_tag)
    app.router.add_post("/api/items/bulk_dataset", handle_bulk_dataset)
    app.router.add_get("/api/items/{id}/stream", handle_stream_audio)
    app.router.add_get("/api/items/{id}/download", handle_download_audio)
    app.router.add_get("/api/items/{id}/stems/{model}/{stem}/stream", handle_stream_stem)
    app.router.add_get("/api/items/{id}/stems/{model}/{stem}/download", handle_download_stem)

    # Job queue routes
    app.router.add_get("/api/jobs", handle_list_jobs)
    app.router.add_get("/api/jobs/{id}", handle_get_job)
    app.router.add_post("/api/jobs/{id}/cancel", handle_cancel_job)
    app.router.add_delete("/api/jobs/{id}", handle_delete_job)
    app.router.add_post("/api/queue/controls", handle_queue_controls)

    # Job submission routes
    app.router.add_post("/api/jobs/batch_ingest_yt", handle_submit_ingest_yt)
    app.router.add_post("/api/jobs/batch_ingest_files", handle_submit_ingest_files)
    app.router.add_post("/api/jobs/batch_upload", handle_upload_batch_files)
    app.router.add_post("/api/jobs/batch_separation", handle_submit_separation)
    app.router.add_post("/api/jobs/batch_diarization", handle_submit_diarization)
    app.router.add_post("/api/jobs/target_speaker_filter", handle_submit_target_speaker)
    app.router.add_get("/api/speaker-profiles", handle_list_speaker_profiles)
    app.router.add_post("/api/jobs/batch_benchmark", handle_submit_benchmark)

    # Manifests and exports
    app.router.add_post("/api/manifests/generate", handle_generate_manifest)
    app.router.add_post("/api/exports/create", handle_create_export)
    app.router.add_get("/api/exports/download/{filename}", handle_download_export)

    # Benchmark reports
    app.router.add_get("/api/benchmarks", handle_list_benchmarks)
    app.router.add_get("/api/benchmarks/{id}", handle_get_benchmark)


def create_app() -> web.Application:
    """Create a standalone SonicPipeline application for compatibility."""
    app = web.Application(
        client_max_size=2048 * 1024 * 1024,  # 2GB upload limit
        middlewares=[no_cache_middleware],
    )

    register_lifecycle(app)
    register_api_routes(app)

    # Static files and root route
    app.router.add_get("/", handle_index)
    app.router.add_static("/static/", path=STATIC_DIR, name="static")

    return app
