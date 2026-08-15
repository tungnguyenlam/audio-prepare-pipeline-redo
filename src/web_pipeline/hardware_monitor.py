"""Hardware Telemetry and Pipeline Performance Monitor."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / ".data"


class HardwareMonitor:
    """Collects real-time hardware telemetry and pipeline throughput statistics."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.total_processed_items = 0
        self.total_processed_audio_seconds = 0.0
        self.total_processing_wall_time = 0.0
        self._last_cpu_times = psutil.cpu_percent(interval=None)

    def record_item_processed(self, audio_duration_seconds: float, wall_time_seconds: float) -> None:
        """Record completed processing of an audio item for throughput calculation."""
        self.total_processed_items += 1
        self.total_processed_audio_seconds += max(0.0, audio_duration_seconds)
        self.total_processing_wall_time += max(0.001, wall_time_seconds)

    @staticmethod
    def _query_nvidia_smi(device_index: int) -> Dict[str, Optional[float]]:
        """Read host-level GPU load and VRAM counters for one CUDA device."""
        if not shutil.which("nvidia-smi"):
            return {}

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {}

            values = [value.strip() for value in result.stdout.splitlines()[0].split(",")]

            def parse(value: str) -> Optional[float]:
                try:
                    return float(value)
                except ValueError:
                    return None

            if len(values) < 5:
                return {}
            return {
                "utilization_percent": parse(values[0]),
                "used_vram_mb": parse(values[1]),
                "total_vram_mb": parse(values[2]),
                "free_vram_mb": parse(values[3]),
                "temperature_c": parse(values[4]),
            }
        except Exception:
            return {}

    def get_gpu_info(self) -> Dict[str, Any]:
        """Query GPU telemetry via PyTorch and nvidia-smi if present."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(current_device)
            smi_info = self._query_nvidia_smi(current_device)
            total_vram_mb = round(
                smi_info["total_vram_mb"]
                if smi_info.get("total_vram_mb") is not None
                else props.total_memory / (1024 * 1024),
                1,
            )
            allocated_vram_mb = round(torch.cuda.memory_allocated(current_device) / (1024 * 1024), 1)
            reserved_vram_mb = round(torch.cuda.memory_reserved(current_device) / (1024 * 1024), 1)
            used_vram_mb = round(
                smi_info["used_vram_mb"]
                if smi_info.get("used_vram_mb") is not None
                else reserved_vram_mb,
                1,
            )
            free_vram_mb = round(
                smi_info.get("free_vram_mb")
                if smi_info.get("free_vram_mb") is not None
                else max(0.0, total_vram_mb - used_vram_mb),
                1,
            )
            vram_percent = round((used_vram_mb / total_vram_mb) * 100, 1) if total_vram_mb > 0 else 0.0
            gpu_util_percent = smi_info.get("utilization_percent")
            gpu_temp_c = smi_info.get("temperature_c")

            return {
                "available": True,
                "type": "cuda",
                "name": props.name,
                "device_count": device_count,
                "total_vram_mb": total_vram_mb,
                "used_vram_mb": used_vram_mb,
                "allocated_vram_mb": allocated_vram_mb,
                "reserved_vram_mb": reserved_vram_mb,
                "free_vram_mb": free_vram_mb,
                "vram_percent": vram_percent,
                "load_percent": gpu_util_percent,
                "utilization_percent": gpu_util_percent,
                "temperature_c": gpu_temp_c,
            }

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "available": True,
                "type": "mps",
                "name": "Apple Silicon (MPS)",
                "device_count": 1,
                "total_vram_mb": None,
                "used_vram_mb": None,
                "allocated_vram_mb": None,
                "reserved_vram_mb": None,
                "free_vram_mb": None,
                "vram_percent": None,
                "load_percent": None,
                "utilization_percent": None,
                "temperature_c": None,
            }

        return {
            "available": False,
            "type": "cpu",
            "name": "No GPU (CPU Mode)",
            "device_count": 0,
            "total_vram_mb": 0,
            "used_vram_mb": 0,
            "allocated_vram_mb": 0,
            "reserved_vram_mb": 0,
            "free_vram_mb": 0,
            "vram_percent": 0,
            "load_percent": None,
            "utilization_percent": None,
            "temperature_c": None,
        }

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Collect current system metrics and throughput stats."""
        # CPU
        cpu_overall = psutil.cpu_percent(interval=None)
        cpu_count_logical = psutil.cpu_count(logical=True) or 1
        cpu_count_physical = psutil.cpu_count(logical=False) or 1
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = round(cpu_freq.current, 1) if cpu_freq else None

        # Memory (RAM)
        mem = psutil.virtual_memory()
        ram_total_gb = round(mem.total / (1024**3), 2)
        ram_used_gb = round(mem.used / (1024**3), 2)
        ram_available_gb = round(mem.available / (1024**3), 2)
        ram_percent = mem.percent

        # Disk Storage
        try:
            disk = psutil.disk_usage(str(DATA_DIR if DATA_DIR.exists() else ROOT_DIR))
            disk_total_gb = round(disk.total / (1024**3), 2)
            disk_used_gb = round(disk.used / (1024**3), 2)
            disk_free_gb = round(disk.free / (1024**3), 2)
            disk_percent = disk.percent
        except Exception:
            disk_total_gb = disk_used_gb = disk_free_gb = disk_percent = 0.0

        # Data directory size
        data_size_bytes = 0
        if DATA_DIR.exists():
            for root, _, files in os.walk(DATA_DIR):
                for f in files:
                    try:
                        data_size_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        data_size_mb = round(data_size_bytes / (1024 * 1024), 2)

        # Throughput Calculations
        uptime_seconds = round(time.time() - self.start_time, 1)
        speedup_factor = (
            round(self.total_processed_audio_seconds / self.total_processing_wall_time, 2)
            if self.total_processing_wall_time > 0
            else 0.0
        )
        audio_hours = round(self.total_processed_audio_seconds / 3600.0, 3)

        return {
            "timestamp": time.time(),
            "uptime_seconds": uptime_seconds,
            "cpu": {
                "utilization_percent": cpu_overall,
                "logical_cores": cpu_count_logical,
                "physical_cores": cpu_count_physical,
                "frequency_mhz": cpu_freq_mhz,
            },
            "ram": {
                "total_gb": ram_total_gb,
                "used_gb": ram_used_gb,
                "available_gb": ram_available_gb,
                "percent": ram_percent,
            },
            "disk": {
                "total_gb": disk_total_gb,
                "used_gb": disk_used_gb,
                "free_gb": disk_free_gb,
                "percent": disk_percent,
                "pipeline_data_mb": data_size_mb,
            },
            "gpu": self.get_gpu_info(),
            "throughput": {
                "processed_items": self.total_processed_items,
                "processed_audio_seconds": round(self.total_processed_audio_seconds, 1),
                "processed_audio_hours": audio_hours,
                "speedup_factor": speedup_factor,
            },
            "runtime": {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "pid": os.getpid(),
            },
        }


# Singleton instance
hardware_monitor = HardwareMonitor()
