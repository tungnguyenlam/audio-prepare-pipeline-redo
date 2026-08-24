"""High-Throughput Batch Processors for Ingestion, Separation, Diarization, and Benchmark."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch

from src.benchmark.separation.mixer import AudioMixer
from src.diarization import (
    ClusteringDiarizer,
    ClusteringWorkerDiarizer,
    PyannoteDiarizer,
    SortformerWorkerDiarizer,
    SpeakerVerifier,
    ThreeDSpeakerDiarizer,
    ThreeDSpeakerWorkerDiarizer,
)
from src.diarization.schemas import DiarizationResult, Speaker, SpeakerTurn
from src.separation import BSRoFormer, HTDemucs, MelRoFormer, MVSepMDX23
from src.utils.AudioClass import DEFAULT_SAMPLE_RATE, Audio
from src.web_pipeline.dataset_manager import dataset_manager
from src.web_pipeline.hardware_monitor import hardware_monitor
from src.web_pipeline.queue_manager import PipelineJob, JobQueueManager
from src.yt_crawler.YtCrawlerClass import YtCrawler

def _bind_job_cancel(queue: JobQueueManager, job_id: str, worker: Any) -> None:
    """Attach a backend cancel hook when the worker supports it."""
    cancel = getattr(worker, "cancel", None)
    if callable(cancel):
        queue.set_job_cancel_callback(job_id, cancel)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / ".data" / "pipeline"
INGEST_DIR = DATA_DIR / "ingest"
STEMS_DIR = DATA_DIR / "stems"
DIARIZATION_DIR = DATA_DIR / "diarization"
TARGET_SPEAKER_DIR = DATA_DIR / "target_speaker"
BENCHMARK_DIR = DATA_DIR / "benchmarks"

INGEST_DIR.mkdir(parents=True, exist_ok=True)
STEMS_DIR.mkdir(parents=True, exist_ok=True)
DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
TARGET_SPEAKER_DIR.mkdir(parents=True, exist_ok=True)

def _setup_device(device_str: str) -> tuple[str, Optional[float]]:
    """Configure torch active device if CUDA is selected and query current wattage."""
    if device_str in ("auto", "cuda") and torch.cuda.is_available():
        best_index = max(
            range(torch.cuda.device_count()),
            key=lambda index: torch.cuda.get_device_properties(index).total_memory,
        )
        device_str = f"cuda:{best_index}"
    elif device_str == "auto":
        device_str = "cpu"

    if device_str.startswith("cuda:") and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(device_str.split(":")[1]))
        except Exception:
            pass

    power_w = None
    try:
        gpu_info = hardware_monitor.get_gpu_info()
        if gpu_info and gpu_info.get("available"):
            if device_str.startswith("cuda:"):
                try:
                    idx = int(device_str.split(":")[1])
                    for dev in gpu_info.get("devices", []):
                        if dev.get("index") == idx:
                            power_w = dev.get("power_w")
                            break
                except (ValueError, IndexError):
                    pass
            if power_w is None:
                power_w = gpu_info.get("power_w")
    except Exception:
        pass
    return device_str, power_w


def si_sdr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB."""
    length = min(len(estimate), len(reference))
    if length == 0:
        return 0.0
    est = np.asarray(estimate[:length], dtype=np.float64)
    ref = np.asarray(reference[:length], dtype=np.float64)

    est = est - np.mean(est)
    ref = ref - np.mean(ref)

    ref_energy = np.dot(ref, ref) + 1e-12
    target = (np.dot(est, ref) / ref_energy) * ref
    noise = est - target

    target_energy = np.dot(target, target) + 1e-12
    noise_energy = np.dot(noise, noise) + 1e-12
    return float(10.0 * np.log10(target_energy / noise_energy))


def load_mono_waveform(path: str | Path, target_sr: int = 44100) -> np.ndarray:
    """Load audio file as mono float64 array."""
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    mono = np.mean(data, axis=1)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
    return mono


def normalize_ingested_audio(audio: Audio, target_sr: int) -> Audio:
    """Resample a locally ingested file and return a new file-backed Audio."""
    if not target_sr or audio.sample_rate == target_sr:
        return audio

    data, source_sr = sf.read(str(audio.path), dtype="float32", always_2d=True)
    resampled = librosa.resample(data, orig_sr=source_sr, target_sr=target_sr, axis=0)
    output_path = audio.path.with_name(f"{audio.path.stem}_{target_sr}hz.wav")
    sf.write(output_path, resampled, target_sr)
    return Audio.from_file(
        output_path,
        source_id=audio.source_id,
        title=audio.title,
        history=(*audio.history, f"resampled_{target_sr}hz"),
    )


async def process_batch_ingest_yt(job: PipelineJob, queue: JobQueueManager) -> None:
    """Ingest batch of YouTube videos or playlists."""
    event_loop = asyncio.get_running_loop()
    urls = job.params.get("urls", [])
    target_dataset = job.params.get("dataset", "Default")
    tags = list(job.params.get("tags", []))
    raw_sr = job.params.get("sample_rate", DEFAULT_SAMPLE_RATE)
    if raw_sr in (None, "", "native", 0, "0"):
        target_sr = None
    else:
        target_sr = int(raw_sr)

    if isinstance(urls, str):
        urls = [u.strip() for u in urls.replace(",", "\n").splitlines() if u.strip()]

    resolved_entries: List[Dict[str, Any]] = [{"type": "url", "url": url} for url in urls]
    rate_label = "native" if target_sr is None else f"{target_sr}Hz"
    job.add_log(f"Resolving {len(urls)} input URL(s) at {rate_label}...", "info")
    queue.update_job_progress(job.id, current_step="Preparing YouTube ingest...")

    def report_yt_progress(message: str) -> None:
        event_loop.call_soon_threadsafe(
            lambda current_message=message: queue.update_job_progress(
                job.id,
                current_step=current_message[:180],
                log_message=current_message,
            )
        )

    crawler = YtCrawler(
        output_dir=str(INGEST_DIR / "downloads"),
        work_dir=str(INGEST_DIR / "work"),
        sample_rate=target_sr,
        progress_callback=report_yt_progress,
    )
    _bind_job_cancel(queue, job.id, crawler)

    job.total_items = len(resolved_entries)
    job.add_log(f"Total resolved items to download and ingest: {job.total_items}", "info")

    processed = 0
    failed = 0

    for idx, entry in enumerate(resolved_entries, start=1):
        if queue.is_cancelled(job.id):
            job.add_log("Batch ingestion cancelled by user", "warning")
            return

        step_title = entry.get("title") or entry.get("url")
        queue.update_job_progress(
            job.id,
            processed_items=processed,
            failed_items=failed,
            current_step=f"[{idx}/{job.total_items}] Ingesting {step_title}",
        )

        t0 = time.time()
        try:
            audio = await asyncio.to_thread(crawler.download, entry["url"])

            item = dataset_manager.register_audio(
                audio=audio,
                dataset=target_dataset,
                tags=tags + ["youtube"],
                metadata={"original_url": entry.get("url", "")},
            )
            wall_time = time.time() - t0
            hardware_monitor.record_item_processed(item.duration, wall_time)

            job.item_results.append({
                "item_id": item.id,
                "title": item.title,
                "duration": item.duration,
                "sample_rate": item.sample_rate,
                "status": "success",
            })
            processed += 1
            job.add_log(f"Successfully ingested: {item.title} ({round(item.duration, 1)}s)", "info")
        except Exception as e:
            failed += 1
            job.item_results.append({
                "url": entry.get("url", ""),
                "title": step_title,
                "status": "failed",
                "error": str(e),
            })
            job.add_log(f"Failed ingesting {step_title}: {e}", "error")

        queue.update_job_progress(job.id, processed_items=processed, failed_items=failed)


async def process_batch_ingest_files(job: PipelineJob, queue: JobQueueManager) -> None:
    """Ingest multiple local audio files or scan directory."""
    file_paths: List[str] = job.params.get("file_paths", [])
    scan_directory = job.params.get("scan_directory")
    target_dataset = job.params.get("dataset", "Default")
    tags = list(job.params.get("tags", []))
    target_sr = int(job.params.get("sample_rate", DEFAULT_SAMPLE_RATE))

    all_files: List[Path] = []
    for fp in file_paths:
        p = Path(fp)
        if p.is_file():
            all_files.append(p)

    if scan_directory:
        scan_p = Path(scan_directory)
        if scan_p.is_dir():
            for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff"):
                all_files.extend(scan_p.rglob(f"*{ext}"))
                all_files.extend(scan_p.rglob(f"*{ext.upper()}"))

    # Deduplicate
    all_files = list(dict.fromkeys(all_files))
    job.total_items = len(all_files)
    job.add_log(f"Discovered {len(all_files)} audio files for ingestion", "info")

    processed = 0
    failed = 0

    for idx, fpath in enumerate(all_files, start=1):
        if queue.is_cancelled(job.id):
            return

        queue.update_job_progress(
            job.id,
            processed_items=processed,
            failed_items=failed,
            current_step=f"[{idx}/{job.total_items}] Ingesting {fpath.name}",
        )

        t0 = time.time()
        try:
            # Copy to pipeline storage
            out_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{fpath.name}"
            dest_path = INGEST_DIR / out_name
            shutil.copy2(fpath, dest_path)

            audio = Audio.from_file(dest_path, source_id=fpath.stem, title=fpath.stem)
            audio = await asyncio.to_thread(normalize_ingested_audio, audio, target_sr)

            item = dataset_manager.register_audio(
                audio=audio,
                dataset=target_dataset,
                tags=tags + ["local_import"],
            )
            wall_time = time.time() - t0
            hardware_monitor.record_item_processed(item.duration, wall_time)

            job.item_results.append({
                "item_id": item.id,
                "title": item.title,
                "duration": item.duration,
                "sample_rate": item.sample_rate,
                "status": "success",
            })
            processed += 1
            job.add_log(f"Ingested: {item.title}", "info")
        except Exception as e:
            failed += 1
            job.item_results.append({
                "file": str(fpath),
                "status": "failed",
                "error": str(e),
            })
            job.add_log(f"Error ingesting {fpath.name}: {e}", "error")

        queue.update_job_progress(job.id, processed_items=processed, failed_items=failed)


async def process_batch_separation(job: PipelineJob, queue: JobQueueManager) -> None:
    """Batch stem separation across audio items."""
    event_loop = asyncio.get_running_loop()
    item_ids: List[str] = job.params.get("item_ids", [])
    model_name: str = job.params.get("model", "BSRoFormer")
    device: str = job.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    target_sr: int = int(job.params.get("sample_rate", DEFAULT_SAMPLE_RATE))

    # Setup device and query wattage
    device, cur_pwr = _setup_device(device)
    pwr_str = f" (⚡ {cur_pwr} W)" if cur_pwr is not None else ""

    # Retrieve valid audio items
    items = [dataset_manager.get_item(iid) for iid in item_ids if dataset_manager.get_item(iid)]
    job.total_items = len(items)
    job.add_log(f"Initializing {model_name} on device '{device}'{pwr_str} for {len(items)} audio items", "info")
    queue.update_job_progress(job.id, current_step=f"Initializing {model_name} on {device}...")

    # Instantiate model
    separator = None
    try:
        if model_name == "BSRoFormer":
            separator = BSRoFormer(device=device)
        elif model_name == "MelRoFormer":
            separator = MelRoFormer(device=device)
        elif model_name == "HTDemucs":
            def report_cli_progress(message: str) -> None:
                event_loop.call_soon_threadsafe(
                    lambda current_message=message: queue.update_job_progress(
                        job.id,
                        current_step=current_message[:180],
                        log_message=current_message,
                    )
                )

            separator = HTDemucs(device=device, progress_callback=report_cli_progress)
        elif model_name == "MVSepMDX23":
            def report_mvsep_progress(message: str) -> None:
                event_loop.call_soon_threadsafe(
                    lambda current_message=message: queue.update_job_progress(
                        job.id,
                        current_step=current_message[:180],
                        log_message=f"MVSep-MDX23: {current_message}",
                    )
                )

            separator = MVSepMDX23(
                device=device,
                progress_callback=report_mvsep_progress,
            )
        else:
            raise ValueError(f"Unsupported separation backend: {model_name}")

        # Check if ManagedModel needs load
        if hasattr(separator, "load"):
            await asyncio.to_thread(separator.load)
        _bind_job_cancel(queue, job.id, separator)
    except Exception as e:
        job.add_log(f"Failed to initialize separator {model_name}: {e}", "error")
        raise

    processed = 0
    failed = 0

    try:
        for idx, it in enumerate(items, start=1):
            if queue.is_cancelled(job.id):
                job.add_log("Separation batch cancelled by user", "warning")
                break

            queue.update_job_progress(
                job.id,
                processed_items=processed,
                failed_items=failed,
                current_step=f"[{idx}/{job.total_items}] Separating {it.title} ({model_name} on {device})",
            )

            t0 = time.time()
            try:
                audio = it.to_audio()
                out_audio = await asyncio.to_thread(separator.separate, audio)

                # Collect separated stems
                out_p = Path(out_audio.path)
                stems_found: Dict[str, str] = {}

                # Destination dir for stems
                dest_dir = STEMS_DIR / model_name.lower() / it.id
                dest_dir.mkdir(parents=True, exist_ok=True)

                if out_p.is_dir():
                    for f in out_p.glob("*.wav"):
                        stem_name = f.stem
                        dest_stem = dest_dir / f"{stem_name}.wav"
                        shutil.copy2(f, dest_stem)
                        stems_found[stem_name] = str(dest_stem)
                        dataset_manager.attach_stem(it.id, model_name, stem_name, str(dest_stem))
                elif out_p.is_file():
                    dest_stem = dest_dir / "vocals.wav"
                    shutil.copy2(out_p, dest_stem)
                    stems_found["vocals"] = str(dest_stem)
                    dataset_manager.attach_stem(it.id, model_name, "vocals", str(dest_stem))

                wall_time = time.time() - t0
                hardware_monitor.record_item_processed(it.duration, wall_time)
                _, item_pwr = _setup_device(device)

                job.item_results.append({
                    "item_id": it.id,
                    "title": it.title,
                    "model": model_name,
                    "stems": stems_found,
                    "device": device,
                    "power_w": item_pwr,
                    "wall_time": round(wall_time, 2),
                    "status": "success",
                })
                processed += 1
                pwr_tag = f" • ⚡ {item_pwr}W" if item_pwr is not None else ""
                job.add_log(f"[{idx}/{job.total_items}] Completed {it.title} in {round(wall_time, 1)}s (Device: {device}{pwr_tag}, Stems: {list(stems_found.keys())})", "info")

            except Exception as e:
                failed += 1
                job.item_results.append({
                    "item_id": it.id,
                    "title": it.title,
                    "model": model_name,
                    "status": "failed",
                    "error": str(e),
                })
                job.add_log(f"Separation failed for {it.title}: {e}", "error")

            finally:
                # Periodic VRAM flush
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            queue.update_job_progress(job.id, processed_items=processed, failed_items=failed)

    finally:
        queue.set_job_cancel_callback(job.id, None)
        if (
            separator
            and hasattr(separator, "close")
            and not queue.is_cancelled(job.id)
        ):
            await asyncio.to_thread(separator.close)


async def process_batch_diarization(job: PipelineJob, queue: JobQueueManager) -> None:
    """Batch speaker diarization across audio items."""
    item_ids: List[str] = job.params.get("item_ids", [])
    backend: str = job.params.get("backend", "sortformer")
    device: str = job.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model_id: Optional[str] = job.params.get("model_id")
    num_speakers: Optional[int] = job.params.get("num_speakers")
    min_speakers: Optional[int] = job.params.get("min_speakers")
    max_speakers: Optional[int] = job.params.get("max_speakers")
    hf_token: Optional[str] = job.params.get("hf_token")
    include_overlap: bool = bool(job.params.get("include_overlap", False))
    vad_onset = float(job.params.get("vad_onset", 0.5)) if job.params.get("vad_onset") is not None else 0.5
    vad_offset = float(job.params.get("vad_offset", 0.3)) if job.params.get("vad_offset") is not None else 0.3
    chunk_duration_s = float(job.params.get("chunk_duration_s", 1.5))
    chunk_step_s = float(job.params.get("chunk_step_s", 0.75))
    backend_key = backend.lower()

    if backend_key == "sortformer" and any(
        value is not None for value in (num_speakers, min_speakers, max_speakers)
    ):
        job.add_log(
            "Speaker-count bounds are not supported by Sortformer; using backend defaults.",
            "warning",
        )

    # Setup device and query wattage
    device, cur_pwr = _setup_device(device)
    pwr_str = f" (⚡ {cur_pwr} W)" if cur_pwr is not None else ""

    items = [dataset_manager.get_item(iid) for iid in item_ids if dataset_manager.get_item(iid)]
    job.total_items = len(items)
    job.add_log(f"Initializing Diarizer '{backend}' on device '{device}'{pwr_str} for {len(items)} audio items", "info")
    queue.update_job_progress(job.id, current_step=f"Initializing {backend} diarizer on {device}...")

    diarizer = None
    try:
        if backend_key == "sortformer":
            diarizer = SortformerWorkerDiarizer(device=device)
        elif backend_key in {"clustering", "nemo-clustering"}:
            oracle_speakers, max_num_speakers = ClusteringDiarizer.resolve_speaker_settings(
                num_speakers,
                min_speakers,
                max_speakers,
            )
            diarizer = ClusteringWorkerDiarizer(
                device=device,
                num_speakers=oracle_speakers,
                max_num_speakers=max_num_speakers,
                vad_onset=vad_onset,
                vad_offset=vad_offset,
            )
        elif backend_key in {"3d_speaker", "3d-speaker", "threed_speaker", "speakerlab"}:
            oracle_speakers = ThreeDSpeakerDiarizer.resolve_speaker_settings(
                num_speakers,
                min_speakers,
                max_speakers,
            )
            diarizer = ThreeDSpeakerWorkerDiarizer(
                device=device,
                num_speakers=oracle_speakers,
                include_overlap=include_overlap,
                chunk_duration_s=chunk_duration_s,
                chunk_step_s=chunk_step_s,
                token=hf_token if include_overlap else None,
            )
        elif backend_key in {"pyannote_31", "pyannote_3", "pyannote_3.1"}:
            diarizer = PyannoteDiarizer(
                model_id=model_id or "pyannote/speaker-diarization-3.1",
                device=device,
                token=hf_token,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        else:
            diarizer = PyannoteDiarizer(
                model_id=model_id or "pyannote/speaker-diarization-community-1",
                device=device,
                token=hf_token,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        _bind_job_cancel(queue, job.id, diarizer)
        if hasattr(diarizer, "load"):
            await asyncio.to_thread(diarizer.load)
    except Exception as e:
        job.add_log(f"Failed to initialize diarizer {backend}: {e}", "error")
        raise

    processed = 0
    failed = 0

    try:
        for idx, it in enumerate(items, start=1):
            if queue.is_cancelled(job.id):
                break

            queue.update_job_progress(
                job.id,
                processed_items=processed,
                failed_items=failed,
                current_step=f"[{idx}/{job.total_items}] Diarizing {it.title} ({backend} on {device})",
            )

            t0 = time.time()
            try:
                audio = it.to_audio()
                res = await asyncio.to_thread(diarizer.diarize, audio)

                # Persist diarization RTTM and JSON
                item_diar_dir = DIARIZATION_DIR / it.id
                item_diar_dir.mkdir(parents=True, exist_ok=True)
                json_path = item_diar_dir / "diarization.json"

                turns_data = [
                    {
                        "speaker_id": turn.speaker_id,
                        "start_s": round(turn.start_s, 3),
                        "end_s": round(turn.end_s, 3),
                        "duration_s": round(turn.end_s - turn.start_s, 3),
                    }
                    for turn in res.turns
                ]

                diar_summary = {
                    "backend": backend,
                    "speaker_count": len(res.speakers),
                    "num_turns": len(res.turns),
                    "total_speech_duration": round(sum(t["duration_s"] for t in turns_data), 2),
                    "turns": turns_data,
                }

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(diar_summary, f, indent=2)

                dataset_manager.attach_diarization(it.id, diar_summary)

                wall_time = time.time() - t0
                hardware_monitor.record_item_processed(it.duration, wall_time)
                _, item_pwr = _setup_device(device)

                job.item_results.append({
                    "item_id": it.id,
                    "title": it.title,
                    "speaker_count": len(res.speakers),
                    "num_turns": len(res.turns),
                    "device": device,
                    "power_w": item_pwr,
                    "wall_time": round(wall_time, 2),
                    "status": "success",
                })
                processed += 1
                pwr_tag = f" • ⚡ {item_pwr}W" if item_pwr is not None else ""
                job.add_log(f"[{idx}/{job.total_items}] Diarized {it.title}: {len(res.speakers)} speakers, {len(res.turns)} turns (Device: {device}{pwr_tag})", "info")

            except Exception as e:
                failed += 1
                job.item_results.append({
                    "item_id": it.id,
                    "title": it.title,
                    "status": "failed",
                    "error": str(e),
                })
                job.add_log(f"Diarization error for {it.title}: {e}", "error")

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            queue.update_job_progress(job.id, processed_items=processed, failed_items=failed)

    finally:
        queue.set_job_cancel_callback(job.id, None)
        if (
            diarizer
            and hasattr(diarizer, "close")
            and not queue.is_cancelled(job.id)
        ):
            await asyncio.to_thread(diarizer.close)


async def process_target_speaker_filter(job: PipelineJob, queue: JobQueueManager) -> None:
    """Score diarized items against a speaker profile and keep confident matches.

    Requires items with attached diarization (``batch_diarization`` first).
    Writes per-item ``target_speaker.json`` (all scored segments plus the kept
    subset) and optionally exports kept segments as wav cuts.
    """
    item_ids: List[str] = job.params.get("item_ids", [])
    profile_name: str = job.params.get("profile", "")
    threshold: float = float(job.params.get("threshold", 0.6))
    min_duration_s: float = float(job.params.get("min_duration_s", 1.5))
    exclude_overlap: bool = bool(job.params.get("exclude_overlap", True))
    export_cuts: bool = bool(job.params.get("export_cuts", False))
    device: str = job.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    hf_token: Optional[str] = job.params.get("hf_token")

    if not profile_name:
        raise ValueError("Target speaker filter requires a profile name")

    device, cur_pwr = _setup_device(device)
    pwr_str = f" (⚡ {cur_pwr} W)" if cur_pwr is not None else ""

    items = [dataset_manager.get_item(iid) for iid in item_ids if dataset_manager.get_item(iid)]
    job.total_items = len(items)
    job.add_log(
        f"Target speaker filter '{profile_name}' on '{device}'{pwr_str} for "
        f"{len(items)} items (threshold={threshold}, min_dur={min_duration_s}s, "
        f"exclude_overlap={exclude_overlap})",
        "info",
    )
    queue.update_job_progress(job.id, current_step=f"Loading speaker verifier on {device}...")

    verifier = SpeakerVerifier(device=device, token=hf_token)
    profile = verifier.load_profile(profile_name)
    await asyncio.to_thread(verifier.load)

    processed = 0
    failed = 0

    try:
        for idx, it in enumerate(items, start=1):
            if queue.is_cancelled(job.id):
                break

            queue.update_job_progress(
                job.id,
                processed_items=processed,
                failed_items=failed,
                current_step=f"[{idx}/{job.total_items}] Verifying {it.title} vs '{profile_name}'",
            )

            t0 = time.time()
            try:
                diar = it.diarization
                if not diar or not diar.get("turns"):
                    raise ValueError("Item has no attached diarization; run batch diarization first")

                turns = [
                    SpeakerTurn(
                        speaker_id=str(t["speaker_id"]),
                        start_s=float(t["start_s"]),
                        end_s=float(t["end_s"]),
                    )
                    for t in diar["turns"]
                ]
                diarization = DiarizationResult(
                    schema_version="1.0",
                    audio_id=it.source_id or it.id,
                    speakers=[
                        Speaker(speaker_id=spk)
                        for spk in sorted({t.speaker_id for t in turns})
                    ],
                    turns=turns,
                )

                audio = it.to_audio()
                scored = await asyncio.to_thread(verifier.score, audio, diarization, profile)
                kept = SpeakerVerifier.filter(
                    scored,
                    threshold=threshold,
                    min_duration_s=min_duration_s,
                    exclude_overlap=exclude_overlap,
                )

                item_dir = TARGET_SPEAKER_DIR / it.id
                item_dir.mkdir(parents=True, exist_ok=True)
                json_path = item_dir / "target_speaker.json"

                def _segment_dict(seg) -> Dict[str, Any]:
                    return {
                        "speaker_id": seg.speaker_id,
                        "start_s": round(seg.start_s, 3),
                        "end_s": round(seg.end_s, 3),
                        "duration_s": round(seg.end_s - seg.start_s, 3),
                        "similarity": round(seg.similarity, 4),
                        "overlaps_other_speaker": seg.overlaps_other_speaker,
                    }

                kept_duration = sum(seg.end_s - seg.start_s for seg in kept.segments)
                report = {
                    "profile": profile_name,
                    "model_id": scored.model.model_id if scored.model else None,
                    "threshold": threshold,
                    "min_duration_s": min_duration_s,
                    "exclude_overlap": exclude_overlap,
                    "num_scored": len(scored.segments),
                    "num_kept": len(kept.segments),
                    "kept_duration_s": round(kept_duration, 2),
                    "kept_segments": [_segment_dict(seg) for seg in kept.segments],
                    "all_segments": [_segment_dict(seg) for seg in scored.segments],
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)

                exported_paths: List[str] = []
                if export_cuts and kept.segments:
                    from src.utils.AudioCutter import AudioCutter

                    cutter = AudioCutter(output_dir=item_dir / "segments")
                    for seg in kept.segments:
                        clip = await asyncio.to_thread(
                            cutter.cut, audio, seg.start_s, seg.end_s
                        )
                        exported_paths.append(str(clip.path))

                summary = {key: value for key, value in report.items() if key != "all_segments"}
                summary["json_path"] = str(json_path)
                if exported_paths:
                    summary["exported_cuts"] = exported_paths
                dataset_manager.attach_target_speaker(it.id, summary)

                wall_time = time.time() - t0
                hardware_monitor.record_item_processed(it.duration, wall_time)

                job.item_results.append({
                    "item_id": it.id,
                    "title": it.title,
                    "num_scored": len(scored.segments),
                    "num_kept": len(kept.segments),
                    "kept_duration_s": round(kept_duration, 2),
                    "device": device,
                    "wall_time": round(wall_time, 2),
                    "status": "success",
                })
                processed += 1
                job.add_log(
                    f"[{idx}/{job.total_items}] {it.title}: kept "
                    f"{len(kept.segments)}/{len(scored.segments)} segments "
                    f"({kept_duration:.1f}s of '{profile_name}' speech)",
                    "info",
                )

            except Exception as e:
                failed += 1
                job.item_results.append({
                    "item_id": it.id,
                    "title": it.title,
                    "status": "failed",
                    "error": str(e),
                })
                job.add_log(f"Target speaker filter error for {it.title}: {e}", "error")

            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            queue.update_job_progress(job.id, processed_items=processed, failed_items=failed)

    finally:
        await asyncio.to_thread(verifier.unload)


async def process_batch_benchmark(job: PipelineJob, queue: JobQueueManager) -> None:
    """Execute separation benchmark suite across speech and music datasets."""
    event_loop = asyncio.get_running_loop()
    speech_ids: List[str] = job.params.get("speech_item_ids", [])
    music_ids: List[str] = job.params.get("music_item_ids", [])
    models_to_test: List[str] = job.params.get("models", ["BSRoFormer", "HTDemucs"])
    snr_levels: List[float] = [float(s) for s in job.params.get("snr_levels", [0.0, 6.0, 12.0])]
    device: str = job.params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    device, cur_pwr = _setup_device(device)
    pwr_str = f" (⚡ {cur_pwr} W)" if cur_pwr is not None else ""

    speech_items = [dataset_manager.get_item(iid) for iid in speech_ids if dataset_manager.get_item(iid)]
    music_items = [dataset_manager.get_item(iid) for iid in music_ids if dataset_manager.get_item(iid)]

    if not speech_items or not music_items:
        raise ValueError("Benchmark requires at least 1 speech item and 1 music item")

    # Generate mix matrix
    benchmark_tasks: List[Dict[str, Any]] = []
    for spk in speech_items:
        for mus in music_items:
            for snr in snr_levels:
                benchmark_tasks.append({
                    "speech": spk,
                    "music": mus,
                    "snr": snr,
                    "sample_id": f"bench_{spk.id}_{mus.id}_snr{str(snr).replace('-', 'm').replace('.', 'p')}",
                })

    job.total_items = len(benchmark_tasks) * len(models_to_test)
    job.add_log(
        f"Separation Benchmark initialized on device '{device}'{pwr_str}: {len(speech_items)} speech x {len(music_items)} music x {len(snr_levels)} SNRs = {len(benchmark_tasks)} mixtures across {len(models_to_test)} models ({job.total_items} runs total)",
        "info",
    )

    mixer = AudioMixer(sample_rate=DEFAULT_SAMPLE_RATE, channels=1)
    evaluated_records: List[Dict[str, Any]] = []

    processed = 0
    failed = 0

    for model_name in models_to_test:
        if queue.is_cancelled(job.id):
            break

        job.add_log(f"Loading separation model {model_name} on {device} for benchmark...", "info")
        separator = None
        try:
            if model_name == "BSRoFormer":
                separator = BSRoFormer(device=device)
            elif model_name == "MelRoFormer":
                separator = MelRoFormer(device=device)
            elif model_name == "HTDemucs":
                def report_cli_progress(message: str) -> None:
                    event_loop.call_soon_threadsafe(
                        lambda current_message=message: queue.update_job_progress(
                            job.id,
                            current_step=current_message[:180],
                            log_message=current_message,
                        )
                    )

                separator = HTDemucs(device=device, progress_callback=report_cli_progress)
            elif model_name == "MVSepMDX23":
                def report_mvsep_progress(message: str) -> None:
                    event_loop.call_soon_threadsafe(
                        lambda current_message=message: queue.update_job_progress(
                            job.id,
                            current_step=current_message[:180],
                            log_message=f"MVSep-MDX23: {current_message}",
                        )
                    )

                separator = MVSepMDX23(
                    device=device,
                    progress_callback=report_mvsep_progress,
                )
            else:
                raise ValueError(f"Unsupported separation backend: {model_name}")
            if hasattr(separator, "load"):
                await asyncio.to_thread(separator.load)
            _bind_job_cancel(queue, job.id, separator)
        except Exception as e:
            job.add_log(f"Failed to load model {model_name}: {e}", "error")
            failed += len(benchmark_tasks)
            continue

        try:
            for b_task in benchmark_tasks:
                if queue.is_cancelled(job.id):
                    break

                spk = b_task["speech"]
                mus = b_task["music"]
                snr = b_task["snr"]
                sample_id = b_task["sample_id"]

                queue.update_job_progress(
                    job.id,
                    processed_items=processed,
                    failed_items=failed,
                    current_step=f"[{processed + failed + 1}/{job.total_items}] {model_name} on {spk.title} + {mus.title} ({device}, SNR {snr}dB)",
                )

                t0 = time.time()
                try:
                    # 1. Render mix
                    mix_result = await asyncio.to_thread(
                        mixer.mix,
                        spk.to_audio(),
                        mus.to_audio(),
                        target_smr_db=snr,
                        seed=42,
                        output_dir=BENCHMARK_DIR / "mixtures" / sample_id,
                    )

                    # 2. Run model separation on mixture
                    separated_audio = await asyncio.to_thread(separator.separate, mix_result.mixture)

                    # Find vocals stem
                    sep_p = Path(separated_audio.path)
                    pred_vocal_path = None
                    if sep_p.is_file():
                        pred_vocal_path = sep_p
                    elif sep_p.is_dir():
                        v_candidates = list(sep_p.glob("*vocal*.wav")) or list(sep_p.glob("*.wav"))
                        if v_candidates:
                            pred_vocal_path = v_candidates[0]

                    if not pred_vocal_path or not pred_vocal_path.exists():
                        raise RuntimeError("Model did not produce vocal stem output")

                    # 3. Compute SI-SDR metrics
                    ref_mono = load_mono_waveform(mix_result.speech_reference.path)
                    mix_mono = load_mono_waveform(mix_result.mixture.path)
                    pred_mono = load_mono_waveform(pred_vocal_path)

                    mix_si_sdr = si_sdr_db(mix_mono, ref_mono)
                    pred_si_sdr = si_sdr_db(pred_mono, ref_mono)
                    si_sdri = pred_si_sdr - mix_si_sdr

                    wall_time = time.time() - t0
                    hardware_monitor.record_item_processed(spk.duration, wall_time)
                    _, item_pwr = _setup_device(device)

                    record = {
                        "model": model_name,
                        "sample_id": sample_id,
                        "speech_title": spk.title,
                        "music_title": mus.title,
                        "target_snr_db": snr,
                        "mixture_si_sdr_db": round(mix_si_sdr, 2),
                        "vocals_si_sdr_db": round(pred_si_sdr, 2),
                        "si_sdri_db": round(si_sdri, 2),
                        "device": device,
                        "power_w": item_pwr,
                        "wall_time": round(wall_time, 2),
                        "status": "success",
                    }
                    evaluated_records.append(record)
                    job.item_results.append(record)
                    processed += 1
                    pwr_tag = f" • ⚡ {item_pwr}W" if item_pwr is not None else ""
                    job.add_log(f"Benchmark [{model_name} / {device}{pwr_tag} / SNR {snr}dB]: SI-SDRi = {round(si_sdri, 2)} dB (time {round(wall_time, 1)}s)", "info")

                except Exception as e:
                    failed += 1
                    job.item_results.append({
                        "model": model_name,
                        "sample_id": sample_id,
                        "status": "failed",
                        "error": str(e),
                    })
                    job.add_log(f"Benchmark run failed for {sample_id} with {model_name}: {e}", "error")

                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                queue.update_job_progress(job.id, processed_items=processed, failed_items=failed)

        finally:
            queue.set_job_cancel_callback(job.id, None)
            if (
                separator
                and hasattr(separator, "close")
                and not queue.is_cancelled(job.id)
            ):
                await asyncio.to_thread(separator.close)

    # Aggregate Benchmark Leaderboard
    leaderboard: Dict[str, Any] = {}
    for m in models_to_test:
        model_records = [r for r in evaluated_records if r["model"] == m and r.get("status") == "success"]
        if model_records:
            sdri_vals = [r["si_sdri_db"] for r in model_records]
            vocals_vals = [r["vocals_si_sdr_db"] for r in model_records]
            leaderboard[m] = {
                "model": m,
                "samples_count": len(model_records),
                "mean_si_sdri_db": round(float(np.mean(sdri_vals)), 2),
                "median_si_sdri_db": round(float(np.median(sdri_vals)), 2),
                "min_si_sdri_db": round(float(np.min(sdri_vals)), 2),
                "max_si_sdri_db": round(float(np.max(sdri_vals)), 2),
                "mean_vocals_si_sdr_db": round(float(np.mean(vocals_vals)), 2),
                "avg_speed_sec": round(float(np.mean([r["wall_time"] for r in model_records])), 2),
            }

    # Save benchmark report file
    report_file = BENCHMARK_DIR / f"{job.id}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "job_id": job.id,
                "timestamp": time.time(),
                "models": models_to_test,
                "leaderboard": leaderboard,
                "runs": evaluated_records,
            },
            f,
            indent=2,
        )

    job.params["leaderboard"] = leaderboard
    job.params["report_path"] = str(report_file)
    job.add_log(f"Benchmark finished. Leaderboard: {leaderboard}", "info")


def register_all_handlers(queue: JobQueueManager) -> None:
    """Register all coroutine handlers with queue manager."""
    queue.register_handler("batch_ingest_yt", process_batch_ingest_yt)
    queue.register_handler("batch_ingest_files", process_batch_ingest_files)
    queue.register_handler("batch_separation", process_batch_separation)
    queue.register_handler("batch_diarization", process_batch_diarization)
    queue.register_handler("target_speaker_filter", process_target_speaker_filter)
    queue.register_handler("batch_benchmark", process_batch_benchmark)
