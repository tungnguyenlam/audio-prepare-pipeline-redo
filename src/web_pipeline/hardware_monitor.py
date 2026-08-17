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
    def _query_all_nvidia_gpus() -> Dict[int, Dict[str, Any]]:
        """Read host-level GPU load and VRAM counters for all CUDA devices."""
        if not shutil.which("nvidia-smi"):
            return {}

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,memory.free,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {}

            def parse_num(value: str) -> Optional[float]:
                try:
                    return float(value)
                except ValueError:
                    return None

            smi_data: Dict[int, Dict[str, Any]] = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                pwr_w = parse_num(parts[7]) if len(parts) > 7 else None
                pwr_limit_w = parse_num(parts[8]) if len(parts) > 8 else None
                pwr_pct = (
                    round((pwr_w / pwr_limit_w) * 100, 1)
                    if (pwr_w is not None and pwr_limit_w and pwr_limit_w > 0)
                    else None
                )
                smi_data[idx] = {
                    "index": idx,
                    "id": f"cuda:{idx}",
                    "name": parts[1],
                    "utilization_percent": parse_num(parts[2]),
                    "used_vram_mb": parse_num(parts[3]),
                    "total_vram_mb": parse_num(parts[4]),
                    "free_vram_mb": parse_num(parts[5]),
                    "temperature_c": parse_num(parts[6]),
                    "power_w": pwr_w,
                    "power_draw_w": pwr_w,
                    "power_limit_w": pwr_limit_w,
                    "power_percent": pwr_pct,
                }
            return smi_data
        except Exception:
            return {}

    @classmethod
    def _query_nvidia_smi(cls, device_index: int) -> Dict[str, Optional[float]]:
        """Read host-level GPU load and VRAM counters for one CUDA device."""
        smi_all = cls._query_all_nvidia_gpus()
        return smi_all.get(device_index, {})

    def get_gpu_info(self) -> Dict[str, Any]:
        """Query GPU telemetry via PyTorch and nvidia-smi if present."""
        smi_devices = self._query_all_nvidia_gpus()

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices: List[Dict[str, Any]] = []

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                smi_info = smi_devices.get(i, {})

                total_vram_mb = round(
                    smi_info.get("total_vram_mb")
                    if smi_info.get("total_vram_mb") is not None
                    else props.total_memory / (1024 * 1024),
                    1,
                )
                allocated_vram_mb = round(torch.cuda.memory_allocated(i) / (1024 * 1024), 1)
                reserved_vram_mb = round(torch.cuda.memory_reserved(i) / (1024 * 1024), 1)
                used_vram_mb = round(
                    smi_info.get("used_vram_mb")
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
                power_w = smi_info.get("power_w")
                power_limit_w = smi_info.get("power_limit_w")
                power_percent = smi_info.get("power_percent")

                devices.append({
                    "index": i,
                    "id": f"cuda:{i}",
                    "name": props.name,
                    "total_vram_mb": total_vram_mb,
                    "used_vram_mb": used_vram_mb,
                    "allocated_vram_mb": allocated_vram_mb,
                    "reserved_vram_mb": reserved_vram_mb,
                    "free_vram_mb": free_vram_mb,
                    "vram_percent": vram_percent,
                    "load_percent": gpu_util_percent,
                    "utilization_percent": gpu_util_percent,
                    "temperature_c": gpu_temp_c,
                    "power_w": power_w,
                    "power_draw_w": power_w,
                    "power_limit_w": power_limit_w,
                    "power_percent": power_percent,
                })

            primary = devices[0] if devices else {}
            tot_vram = round(sum(d["total_vram_mb"] for d in devices), 1) if devices else 0.0
            tot_used_vram = round(sum(d["used_vram_mb"] for d in devices), 1) if devices else 0.0
            tot_vram_pct = round((tot_used_vram / tot_vram) * 100, 1) if tot_vram > 0 else 0.0
            valid_loads = [d["load_percent"] for d in devices if d.get("load_percent") is not None]
            avg_load = round(sum(valid_loads) / len(valid_loads), 1) if valid_loads else primary.get("load_percent")

            valid_powers = [d["power_w"] for d in devices if d.get("power_w") is not None]
            tot_power_w = round(sum(valid_powers), 1) if valid_powers else None
            valid_limits = [d["power_limit_w"] for d in devices if d.get("power_limit_w") is not None]
            tot_power_limit_w = round(sum(valid_limits), 1) if valid_limits else None
            tot_power_pct = (
                round((tot_power_w / tot_power_limit_w) * 100, 1)
                if (tot_power_w is not None and tot_power_limit_w and tot_power_limit_w > 0)
                else None
            )

            return {
                "available": True,
                "type": "cuda",
                "name": primary.get("name", "CUDA GPU"),
                "device_count": device_count,
                "devices": devices,
                "total_vram_mb": primary.get("total_vram_mb", 0.0),
                "used_vram_mb": primary.get("used_vram_mb", 0.0),
                "allocated_vram_mb": primary.get("allocated_vram_mb", 0.0),
                "reserved_vram_mb": primary.get("reserved_vram_mb", 0.0),
                "free_vram_mb": primary.get("free_vram_mb", 0.0),
                "vram_percent": primary.get("vram_percent", 0.0),
                "load_percent": primary.get("load_percent"),
                "utilization_percent": primary.get("utilization_percent"),
                "temperature_c": primary.get("temperature_c"),
                "power_w": primary.get("power_w"),
                "power_draw_w": primary.get("power_w"),
                "power_limit_w": primary.get("power_limit_w"),
                "power_percent": primary.get("power_percent"),
                "aggregate": {
                    "total_vram_mb": tot_vram,
                    "used_vram_mb": tot_used_vram,
                    "vram_percent": tot_vram_pct,
                    "avg_load_percent": avg_load,
                    "total_power_w": tot_power_w,
                    "total_power_limit_w": tot_power_limit_w,
                    "power_percent": tot_power_pct,
                },
            }

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {
                "available": True,
                "type": "mps",
                "name": "Apple Silicon (MPS)",
                "device_count": 1,
                "devices": [{
                    "index": 0,
                    "id": "mps",
                    "name": "Apple Silicon (MPS)",
                    "total_vram_mb": None,
                    "used_vram_mb": None,
                    "allocated_vram_mb": None,
                    "reserved_vram_mb": None,
                    "free_vram_mb": None,
                    "vram_percent": None,
                    "load_percent": None,
                    "utilization_percent": None,
                    "temperature_c": None,
                    "power_w": None,
                    "power_draw_w": None,
                    "power_limit_w": None,
                    "power_percent": None,
                }],
                "total_vram_mb": None,
                "used_vram_mb": None,
                "allocated_vram_mb": None,
                "reserved_vram_mb": None,
                "free_vram_mb": None,
                "vram_percent": None,
                "load_percent": None,
                "utilization_percent": None,
                "temperature_c": None,
                "power_w": None,
                "power_draw_w": None,
                "power_limit_w": None,
                "power_percent": None,
            }

        # If smi found standalone GPUs even if torch is CPU build
        if smi_devices:
            dev_list = list(smi_devices.values())
            primary = dev_list[0]
            valid_powers = [d["power_w"] for d in dev_list if d.get("power_w") is not None]
            tot_power_w = round(sum(valid_powers), 1) if valid_powers else None
            return {
                "available": True,
                "type": "cuda",
                "name": primary.get("name", "NVIDIA GPU"),
                "device_count": len(dev_list),
                "devices": dev_list,
                "total_vram_mb": primary.get("total_vram_mb", 0.0),
                "used_vram_mb": primary.get("used_vram_mb", 0.0),
                "allocated_vram_mb": 0.0,
                "reserved_vram_mb": 0.0,
                "free_vram_mb": primary.get("free_vram_mb", 0.0),
                "vram_percent": round((primary.get("used_vram_mb", 0.0) / primary.get("total_vram_mb", 1.0)) * 100, 1) if primary.get("total_vram_mb") else 0.0,
                "load_percent": primary.get("utilization_percent"),
                "utilization_percent": primary.get("utilization_percent"),
                "temperature_c": primary.get("temperature_c"),
                "power_w": primary.get("power_w"),
                "power_draw_w": primary.get("power_w"),
                "power_limit_w": primary.get("power_limit_w"),
                "power_percent": primary.get("power_percent"),
                "aggregate": {
                    "total_power_w": tot_power_w,
                },
            }

        return {
            "available": False,
            "type": "cpu",
            "name": "No GPU (CPU Mode)",
            "device_count": 0,
            "devices": [],
            "total_vram_mb": 0,
            "used_vram_mb": 0,
            "allocated_vram_mb": 0,
            "reserved_vram_mb": 0,
            "free_vram_mb": 0,
            "vram_percent": 0,
            "load_percent": None,
            "utilization_percent": None,
            "temperature_c": None,
            "power_w": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "power_percent": None,
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

    def get_telemetry(self) -> Dict[str, Any]:
        """Alias for get_system_telemetry."""
        return self.get_system_telemetry()


# Singleton instance
hardware_monitor = HardwareMonitor()
