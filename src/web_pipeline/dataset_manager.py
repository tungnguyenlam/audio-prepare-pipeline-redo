"""Dataset Collection, Audio Registry, and Manifest Manager for Large-Scale Audio."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import soundfile as sf

from src.utils.AudioClass import Audio

logger = logging.getLogger("dataset_manager")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / ".data" / "pipeline"
REGISTRY_FILE = DATA_DIR / "dataset_registry.json"
DATASETS_FILE = DATA_DIR / "datasets.json"
EXPORTS_DIR = DATA_DIR / "exports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AudioItem:
    """Representation of an audio asset tracked in the pipeline."""

    id: str
    source_id: str
    title: str
    path: str
    dataset: str
    duration: float
    sample_rate: int
    channels: int
    native_sample_rate: int
    format: str
    source_url: Optional[str] = None
    channel_id: Optional[str] = None
    channel_name: Optional[str] = None
    channel_url: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    stems: Dict[str, Dict[str, str]] = field(default_factory=dict)  # model -> {stem_name: path}
    diarization: Optional[Dict[str, Any]] = None  # {speaker_count, turns: [...], rttm_path}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AudioItem:
        return cls(
            id=data["id"],
            source_id=data.get("source_id", ""),
            title=data.get("title", "Untitled"),
            path=data["path"],
            dataset=data.get("dataset", "Default"),
            duration=float(data.get("duration", 0.0)),
            sample_rate=int(data.get("sample_rate", 44100)),
            channels=int(data.get("channels", 1)),
            native_sample_rate=int(data.get("native_sample_rate", data.get("sample_rate", 44100))),
            format=data.get("format", "WAV"),
            source_url=data.get("source_url") or data.get("metadata", {}).get("original_url"),
            channel_id=data.get("channel_id") or data.get("metadata", {}).get("channel_id"),
            channel_name=data.get("channel_name") or data.get("metadata", {}).get("channel_name"),
            channel_url=data.get("channel_url") or data.get("metadata", {}).get("channel_url"),
            tags=list(data.get("tags", [])),
            created_at=float(data.get("created_at", time.time())),
            stems=dict(data.get("stems", {})),
            diarization=data.get("diarization"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_audio(self) -> Audio:
        """Instantiate a file-backed Audio class."""
        return Audio(
            path=self.path,
            source_id=self.source_id,
            title=self.title,
            sample_rate=self.sample_rate,
            duration_s=self.duration,
            channels=self.channels,
            format=self.format,
            native_sample_rate=self.native_sample_rate,
            source_url=self.source_url,
            channel_id=self.channel_id,
            channel_name=self.channel_name,
            channel_url=self.channel_url,
        )


class DatasetManager:
    """Manages audio registries, dataset grouping, metadata, and manifest exports."""

    def __init__(self) -> None:
        self._items: Dict[str, AudioItem] = {}
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load items and datasets from disk."""
        if DATASETS_FILE.exists():
            try:
                with open(DATASETS_FILE, "r", encoding="utf-8") as f:
                    self._datasets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading datasets.json: {e}")
                self._datasets = {}

        if not self._datasets:
            self._datasets = {
                "Default": {
                    "name": "Default",
                    "description": "General ingestion pool",
                    "created_at": time.time(),
                    "tags": ["raw"],
                },
                "Speech Corpus": {
                    "name": "Speech Corpus",
                    "description": "Voice, spoken word, podcast recordings",
                    "created_at": time.time(),
                    "tags": ["speech", "vocal"],
                },
                "Music BGM": {
                    "name": "Music BGM",
                    "description": "Background music, instrumentals, soundscapes",
                    "created_at": time.time(),
                    "tags": ["music", "instrumental"],
                },
                "Benchmark Eval": {
                    "name": "Benchmark Eval",
                    "description": "Evaluated mixtures and reference stems",
                    "created_at": time.time(),
                    "tags": ["benchmark", "mix"],
                },
            }
            self._save_datasets()

        if REGISTRY_FILE.exists():
            try:
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    raw_items = json.load(f)
                    for item_id, item_data in raw_items.items():
                        # Verify file still exists on disk
                        if Path(item_data.get("path", "")).exists():
                            self._items[item_id] = AudioItem.from_dict(item_data)
            except Exception as e:
                logger.error(f"Error loading registry file: {e}")

    def _save_items(self) -> None:
        """Persist items to disk."""
        try:
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump({k: v.to_dict() for k, v in self._items.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def _save_datasets(self) -> None:
        """Persist datasets metadata to disk."""
        try:
            with open(DATASETS_FILE, "w", encoding="utf-8") as f:
                json.dump(self._datasets, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving datasets: {e}")

    def register_audio(
        self,
        audio: Audio,
        dataset: str = "Default",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AudioItem:
        """Register an existing Audio file into a dataset collection."""
        item_id = f"aud_{uuid.uuid4().hex[:10]}"
        tags_list = list(tags) if tags else []

        if dataset not in self._datasets:
            self.create_dataset(dataset)

        item = AudioItem(
            id=item_id,
            source_id=audio.source_id or item_id,
            title=audio.title or Path(audio.path).stem,
            path=str(Path(audio.path).resolve()),
            dataset=dataset,
            duration=round(audio.duration_s or 0.0, 3),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            native_sample_rate=audio.native_sample_rate or audio.sample_rate,
            format=audio.format,
            source_url=audio.source_url,
            channel_id=audio.channel_id,
            channel_name=audio.channel_name,
            channel_url=audio.channel_url,
            tags=tags_list,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._items[item_id] = item
        self._save_items()
        return item

    def get_item(self, item_id: str) -> Optional[AudioItem]:
        """Retrieve an audio item by ID."""
        return self._items.get(item_id)

    def list_items(
        self,
        dataset: Optional[str] = None,
        query: Optional[str] = None,
        tag: Optional[str] = None,
        channel_id: Optional[str] = None,
        has_stems: Optional[bool] = None,
        has_diarization: Optional[bool] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Query and filter registered audio items with pagination."""
        results = list(self._items.values())

        if dataset and dataset != "all":
            results = [x for x in results if x.dataset == dataset]

        if query:
            q_lower = query.lower()
            results = [
                x for x in results
                if q_lower in x.title.lower()
                or q_lower in x.source_id.lower()
                or q_lower in (x.channel_name or "").lower()
                or q_lower in (x.channel_id or "").lower()
                or any(q_lower in t.lower() for t in x.tags)
            ]

        if tag:
            results = [x for x in results if tag in x.tags]

        if channel_id and channel_id != "all":
            results = [x for x in results if x.channel_id == channel_id]

        if has_stems is not None:
            results = [x for x in results if (bool(x.stems) == has_stems)]

        if has_diarization is not None:
            results = [x for x in results if (bool(x.diarization) == has_diarization)]

        if min_duration is not None:
            results = [x for x in results if x.duration >= min_duration]

        if max_duration is not None:
            results = [x for x in results if x.duration <= max_duration]

        # Sorting
        if sort_by == "duration":
            results.sort(key=lambda x: x.duration, reverse=sort_desc)
        elif sort_by == "title":
            results.sort(key=lambda x: x.title.lower(), reverse=sort_desc)
        elif sort_by == "sample_rate":
            results.sort(key=lambda x: x.sample_rate, reverse=sort_desc)
        else:
            results.sort(key=lambda x: x.created_at, reverse=sort_desc)

        total_count = len(results)
        total_duration = sum(x.duration for x in results)
        paginated = results[offset : offset + limit]

        return {
            "total": total_count,
            "total_duration_seconds": round(total_duration, 1),
            "total_duration_hours": round(total_duration / 3600.0, 3),
            "offset": offset,
            "limit": limit,
            "items": [x.to_dict() for x in paginated],
        }

    def update_item(self, item_id: str, updates: Dict[str, Any]) -> bool:
        """Update properties of an item (title, tags, dataset, metadata)."""
        item = self._items.get(item_id)
        if not item:
            return False

        if "title" in updates and updates["title"]:
            item.title = str(updates["title"])
        if "dataset" in updates and updates["dataset"]:
            new_ds = str(updates["dataset"])
            if new_ds not in self._datasets:
                self.create_dataset(new_ds)
            item.dataset = new_ds
        if "tags" in updates and isinstance(updates["tags"], list):
            item.tags = list(set(updates["tags"]))
        if "metadata" in updates and isinstance(updates["metadata"], dict):
            item.metadata.update(updates["metadata"])

        self._save_items()
        return True

    def attach_stem(self, item_id: str, model_name: str, stem_name: str, stem_path: str) -> None:
        """Attach separated stem path to an audio item."""
        item = self._items.get(item_id)
        if not item:
            return
        if model_name not in item.stems:
            item.stems[model_name] = {}
        item.stems[model_name][stem_name] = str(Path(stem_path).resolve())
        tag_name = f"sep_{model_name.lower()}"
        if tag_name not in item.tags:
            item.tags.append(tag_name)
        self._save_items()

    def attach_diarization(self, item_id: str, diarization_result: Dict[str, Any]) -> None:
        """Attach speaker diarization outcome to an audio item."""
        item = self._items.get(item_id)
        if not item:
            return
        item.diarization = diarization_result
        if "diarized" not in item.tags:
            item.tags.append("diarized")
        self._save_items()

    def attach_target_speaker(self, item_id: str, summary: Dict[str, Any]) -> None:
        """Attach a target-speaker verification summary to an audio item.

        Stored under ``metadata["target_speaker"]`` and tagged
        ``target:<profile>`` so manifest exports can filter verified items.
        """
        item = self._items.get(item_id)
        if not item:
            return
        item.metadata["target_speaker"] = summary
        profiles = item.metadata.setdefault("target_speakers", {})
        profile = summary.get("profile")
        if profile:
            profiles[str(profile)] = summary
            tag_name = f"target:{profile}"
            if tag_name not in item.tags:
                item.tags.append(tag_name)
        self._save_items()

    def delete_items(self, item_ids: List[str], delete_files: bool = False) -> int:
        """Delete multiple audio items from registry, optionally deleting source files."""
        deleted = 0
        for item_id in item_ids:
            item = self._items.pop(item_id, None)
            if item:
                deleted += 1
                if delete_files:
                    try:
                        p = Path(item.path)
                        if p.exists():
                            p.unlink()
                        # Also delete stem files
                        for model_stems in item.stems.values():
                            for stem_path in model_stems.values():
                                sp = Path(stem_path)
                                if sp.exists():
                                    sp.unlink()
                    except Exception as e:
                        logger.warning(f"Failed deleting file for {item_id}: {e}")
        if deleted > 0:
            self._save_items()
        return deleted

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all dataset collections with item counts, duration, and tag stats."""
        stats: Dict[str, Dict[str, Any]] = {}
        for ds_name, ds_info in self._datasets.items():
            stats[ds_name] = {
                "name": ds_name,
                "description": ds_info.get("description", ""),
                "created_at": ds_info.get("created_at", time.time()),
                "tags": ds_info.get("tags", []),
                "item_count": 0,
                "total_duration_seconds": 0.0,
                "separated_count": 0,
                "diarized_count": 0,
            }

        for item in self._items.values():
            ds_name = item.dataset
            if ds_name not in stats:
                stats[ds_name] = {
                    "name": ds_name,
                    "description": "",
                    "created_at": time.time(),
                    "tags": [],
                    "item_count": 0,
                    "total_duration_seconds": 0.0,
                    "separated_count": 0,
                    "diarized_count": 0,
                }
            s = stats[ds_name]
            s["item_count"] += 1
            s["total_duration_seconds"] += item.duration
            if item.stems:
                s["separated_count"] += 1
            if item.diarization:
                s["diarized_count"] += 1

        for s in stats.values():
            s["total_duration_seconds"] = round(s["total_duration_seconds"], 1)
            s["total_duration_hours"] = round(s["total_duration_seconds"] / 3600.0, 3)

        return list(stats.values())

    def list_channels(self) -> List[Dict[str, Any]]:
        """List YouTube channels with aggregate processing coverage."""
        channels: Dict[str, Dict[str, Any]] = {}
        for item in self._items.values():
            if not item.channel_id and not item.channel_name:
                continue
            key = item.channel_id or item.channel_name or "unknown_channel"
            summary = channels.setdefault(
                key,
                {
                    "channel_id": item.channel_id,
                    "channel_name": item.channel_name or item.channel_id or "Unknown channel",
                    "channel_url": item.channel_url,
                    "item_count": 0,
                    "total_duration_seconds": 0.0,
                    "separated_count": 0,
                    "diarized_count": 0,
                    "target_filtered_count": 0,
                },
            )
            summary["item_count"] += 1
            summary["total_duration_seconds"] += item.duration
            summary["separated_count"] += int(bool(item.stems))
            summary["diarized_count"] += int(bool(item.diarization))
            summary["target_filtered_count"] += int(bool(item.metadata.get("target_speaker")))

        for summary in channels.values():
            summary["total_duration_seconds"] = round(summary["total_duration_seconds"], 1)
        return sorted(channels.values(), key=lambda x: x["channel_name"].lower())

    def create_dataset(self, name: str, description: str = "", tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new dataset collection."""
        name = name.strip()
        if not name:
            raise ValueError("Dataset name cannot be empty")
        self._datasets[name] = {
            "name": name,
            "description": description,
            "created_at": time.time(),
            "tags": list(tags) if tags else [],
        }
        self._save_datasets()
        return self._datasets[name]

    def delete_dataset(self, name: str, delete_items: bool = False) -> bool:
        """Delete a dataset collection."""
        if name in self._datasets:
            del self._datasets[name]
            self._save_datasets()
            if delete_items:
                to_delete = [k for k, v in self._items.items() if v.dataset == name]
                self.delete_items(to_delete, delete_files=True)
            else:
                # Reassign items to Default
                for item in self._items.values():
                    if item.dataset == name:
                        item.dataset = "Default"
                self._save_items()
            return True
        return False

    def bulk_tag(self, item_ids: List[str], add_tags: List[str], remove_tags: List[str]) -> int:
        """Add or remove tags across multiple items."""
        affected = 0
        add_set = set(t.strip() for t in add_tags if t.strip())
        rem_set = set(t.strip() for t in remove_tags if t.strip())

        for item_id in item_ids:
            item = self._items.get(item_id)
            if item:
                current = set(item.tags)
                new_tags = (current | add_set) - rem_set
                if new_tags != current:
                    item.tags = list(new_tags)
                    affected += 1
        if affected > 0:
            self._save_items()
        return affected

    def bulk_assign_dataset(self, item_ids: List[str], target_dataset: str) -> int:
        """Move multiple items into a designated dataset."""
        target_dataset = target_dataset.strip()
        if not target_dataset:
            return 0
        if target_dataset not in self._datasets:
            self.create_dataset(target_dataset)

        affected = 0
        for item_id in item_ids:
            item = self._items.get(item_id)
            if item and item.dataset != target_dataset:
                item.dataset = target_dataset
                affected += 1
        if affected > 0:
            self._save_items()
        return affected

    def generate_manifest(
        self,
        item_ids: Optional[List[str]] = None,
        dataset: Optional[str] = None,
        format_type: str = "jsonl",
    ) -> str:
        """Generate formatted manifest string (JSONL or CSV)."""
        if item_ids:
            items = [self._items[iid] for iid in item_ids if iid in self._items]
        elif dataset:
            items = [x for x in self._items.values() if x.dataset == dataset]
        else:
            items = list(self._items.values())

        if format_type.lower() == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "id",
                "source_id",
                "title",
                "audio_filepath",
                "duration",
                "sample_rate",
                "channels",
                "dataset",
                "tags",
                "stems_count",
                "speaker_count",
            ])
            for it in items:
                spk_count = it.diarization.get("speaker_count", 0) if it.diarization else 0
                writer.writerow([
                    it.id,
                    it.source_id,
                    it.title,
                    it.path,
                    it.duration,
                    it.sample_rate,
                    it.channels,
                    it.dataset,
                    "|".join(it.tags),
                    len(it.stems),
                    spk_count,
                ])
            return output.getvalue()

        # Default JSONL (standard ML audio manifest format)
        lines = []
        for it in items:
            record = {
                "id": it.id,
                "audio_filepath": it.path,
                "duration": it.duration,
                "sample_rate": it.sample_rate,
                "channels": it.channels,
                "title": it.title,
                "source_id": it.source_id,
                "dataset": it.dataset,
                "tags": it.tags,
                "stems": it.stems,
                "diarization": it.diarization,
                "metadata": it.metadata,
            }
            lines.append(json.dumps(record))
        return "\n".join(lines)

    def create_export_bundle(
        self,
        item_ids: Optional[List[str]] = None,
        dataset: Optional[str] = None,
        include_stems: bool = True,
        include_manifests: bool = True,
    ) -> Path:
        """Package audio files, separated stems, and manifest into a ZIP file."""
        if item_ids:
            items = [self._items[iid] for iid in item_ids if iid in self._items]
        elif dataset:
            items = [x for x in self._items.values() if x.dataset == dataset]
        else:
            items = list(self._items.values())

        bundle_id = f"export_{dataset or 'selected'}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        zip_path = EXPORTS_DIR / f"{bundle_id}.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            if include_manifests:
                jsonl_data = self.generate_manifest(item_ids=[it.id for it in items], format_type="jsonl")
                zf.writestr("manifest.jsonl", jsonl_data)
                csv_data = self.generate_manifest(item_ids=[it.id for it in items], format_type="csv")
                zf.writestr("manifest.csv", csv_data)

            for it in items:
                src_p = Path(it.path)
                if src_p.exists():
                    zf.write(src_p, arcname=f"audio/{src_p.name}")

                if include_stems and it.stems:
                    for model_name, stem_dict in it.stems.items():
                        for stem_name, stem_p_str in stem_dict.items():
                            stem_p = Path(stem_p_str)
                            if stem_p.exists():
                                zf.write(stem_p, arcname=f"stems/{model_name}/{it.id}_{stem_name}{stem_p.suffix}")

        return zip_path


# Singleton dataset manager
dataset_manager = DatasetManager()
