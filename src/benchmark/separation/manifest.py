"""Load and resolve JSON manifests for the separation benchmark.

Manifest paths are repository-relative by convention.  The default repository
root is inferred from this module, and callers working with another checkout
can pass ``repository_root`` to :meth:`SeparationBenchmarkManifest.load` or to
an entry's ``to_source`` method.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.benchmark.separation.schemas import (
    BenchmarkDefinition,
    Difficulty,
    MusicCategory,
    MusicSource,
    SpeechSource,
)
from src.utils.AudioClass import Audio


PathLike = str | Path
DEFAULT_MANIFEST_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "separation" / "manifests"


class ManifestError(ValueError):
    """Raised when a separation benchmark manifest is malformed or inconsistent."""


def _repository_root(repository_root: PathLike | None) -> Path:
    if repository_root is not None:
        return Path(repository_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def _resolve_source_path(path: str, repository_root: PathLike | None) -> Path:
    source_path = Path(path).expanduser()
    if source_path.is_absolute():
        return source_path.resolve()
    return (_repository_root(repository_root) / source_path).resolve()


def _read_manifest(path: PathLike) -> list[Mapping[str, Any]]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Could not read manifest {manifest_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise ManifestError(f"Manifest {manifest_path} must contain a JSON array")

    entries: list[Mapping[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ManifestError(
                f"Manifest {manifest_path} entry {index} must be a JSON object"
            )
        entries.append(item)
    return entries


def _required_string(item: Mapping[str, Any], field: str, context: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{context} field '{field}' must be a non-empty string")
    return value


def _optional_string(item: Mapping[str, Any], field: str, context: str) -> str | None:
    value = item.get(field)
    if value is not None and not isinstance(value, str):
        raise ManifestError(f"{context} field '{field}' must be a string or null")
    return value


def _required_number(item: Mapping[str, Any], field: str, context: str) -> float:
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{context} field '{field}' must be a number")
    return float(value)


def _required_int(item: Mapping[str, Any], field: str, context: str) -> int:
    value = item.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{context} field '{field}' must be an integer")
    return value


def _enum_value(enum_type: type[MusicCategory] | type[Difficulty], item: Mapping[str, Any], field: str, context: str):
    value = _required_string(item, field, context)
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ManifestError(
            f"{context} field '{field}' has unsupported value {value!r}; expected one of: {allowed}"
        ) from exc


@dataclass(frozen=True)
class SpeechManifestEntry:
    """JSON-facing metadata for one speech source."""

    speech_id: str
    path: str
    language: str
    speaker_id: str | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any], *, context: str) -> SpeechManifestEntry:
        return cls(
            speech_id=_required_string(item, "speech_id", context),
            path=_required_string(item, "path", context),
            language=_required_string(item, "language", context),
            speaker_id=_optional_string(item, "speaker_id", context),
            notes=_optional_string(item, "notes", context),
        )

    def to_source(self, *, repository_root: PathLike | None = None) -> SpeechSource:
        """Load this entry's file using the existing :class:`Audio` loader."""
        audio = Audio.from_file(
            _resolve_source_path(self.path, repository_root),
            source_id=self.speech_id,
        )
        return SpeechSource(
            speech_id=self.speech_id,
            audio=audio,
            language=self.language,
            speaker_id=self.speaker_id,
            notes=self.notes,
        )


@dataclass(frozen=True)
class MusicManifestEntry:
    """JSON-facing metadata for one music source."""

    music_id: str
    path: str
    category: MusicCategory
    notes: str | None = None

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any], *, context: str) -> MusicManifestEntry:
        return cls(
            music_id=_required_string(item, "music_id", context),
            path=_required_string(item, "path", context),
            category=_enum_value(MusicCategory, item, "category", context),
            notes=_optional_string(item, "notes", context),
        )

    def to_source(self, *, repository_root: PathLike | None = None) -> MusicSource:
        """Load this entry's file using the existing :class:`Audio` loader."""
        audio = Audio.from_file(
            _resolve_source_path(self.path, repository_root),
            source_id=self.music_id,
        )
        return MusicSource(
            music_id=self.music_id,
            audio=audio,
            category=self.category,
            notes=self.notes,
        )


def _load_entries(path: PathLike, entry_type: type[SpeechManifestEntry] | type[MusicManifestEntry]) -> list[Any]:
    manifest_path = Path(path)
    entries = [
        entry_type.from_mapping(item, context=f"{manifest_path} entry {index}")
        for index, item in enumerate(_read_manifest(manifest_path))
    ]
    id_field = "speech_id" if entry_type is SpeechManifestEntry else "music_id"
    seen: set[str] = set()
    for entry in entries:
        entry_id = getattr(entry, id_field)
        if entry_id in seen:
            raise ManifestError(
                f"Manifest {manifest_path} contains duplicate {id_field} {entry_id!r}"
            )
        seen.add(entry_id)
    return entries


def load_speech_manifest(path: PathLike) -> list[SpeechManifestEntry]:
    """Load speech metadata entries from a JSON array."""
    return _load_entries(path, SpeechManifestEntry)


def load_music_manifest(path: PathLike) -> list[MusicManifestEntry]:
    """Load music metadata entries from a JSON array."""
    return _load_entries(path, MusicManifestEntry)


def _load_benchmark_definitions(path: PathLike) -> list[BenchmarkDefinition]:
    manifest_path = Path(path)
    definitions: list[BenchmarkDefinition] = []
    for index, item in enumerate(_read_manifest(manifest_path)):
        context = f"{manifest_path} entry {index}"
        definitions.append(
            BenchmarkDefinition(
                sample_id=_required_string(item, "sample_id", context),
                speech_id=_required_string(item, "speech_id", context),
                music_id=_required_string(item, "music_id", context),
                music_category=_enum_value(MusicCategory, item, "music_category", context),
                difficulty=_enum_value(Difficulty, item, "difficulty", context),
                target_smr_db=_required_number(item, "target_smr_db", context),
                seed=_required_int(item, "seed", context),
            )
        )

    seen: set[str] = set()
    for definition in definitions:
        if definition.sample_id in seen:
            raise ManifestError(
                f"Manifest {manifest_path} contains duplicate sample_id {definition.sample_id!r}"
            )
        seen.add(definition.sample_id)
    return definitions


def load_benchmark_manifest(path: PathLike) -> list[BenchmarkDefinition]:
    """Load planned benchmark definitions from a JSON array."""
    return _load_benchmark_definitions(path)


@dataclass
class SeparationBenchmarkManifest:
    """The three manifests and their cross-reference validation context."""

    speech_entries: list[SpeechManifestEntry]
    music_entries: list[MusicManifestEntry]
    definitions: list[BenchmarkDefinition]
    repository_root: Path

    @classmethod
    def load(
        cls,
        speech_path: PathLike | None = None,
        music_path: PathLike | None = None,
        benchmark_path: PathLike | None = None,
        *,
        repository_root: PathLike | None = None,
        validate_files: bool = False,
    ) -> SeparationBenchmarkManifest:
        """Load all manifests and validate their references.

        If paths are omitted, the standard files under
        ``benchmarks/separation/manifests`` are used.  Relative source paths
        are resolved against ``repository_root`` when sources are resolved.
        """
        root = _repository_root(repository_root)
        manifest_dir = root / "benchmarks" / "separation" / "manifests"
        manifest = cls(
            speech_entries=load_speech_manifest(
                speech_path or manifest_dir / "speech.json"
            ),
            music_entries=load_music_manifest(
                music_path or manifest_dir / "music.json"
            ),
            definitions=load_benchmark_manifest(
                benchmark_path or manifest_dir / "benchmark.json"
            ),
            repository_root=root,
        )
        manifest.validate()
        if validate_files:
            manifest.validate_files()
        return manifest

    def validate(self) -> None:
        """Validate benchmark references and declared music categories."""
        speech_ids = {entry.speech_id for entry in self.speech_entries}
        music_by_id = {entry.music_id: entry for entry in self.music_entries}
        for definition in self.definitions:
            if definition.speech_id not in speech_ids:
                raise ManifestError(
                    f"Benchmark definition {definition.sample_id!r} references unknown speech_id "
                    f"{definition.speech_id!r}"
                )
            music = music_by_id.get(definition.music_id)
            if music is None:
                raise ManifestError(
                    f"Benchmark definition {definition.sample_id!r} references unknown music_id "
                    f"{definition.music_id!r}"
                )
            if definition.music_category != music.category:
                raise ManifestError(
                    f"Benchmark definition {definition.sample_id!r} declares music_category "
                    f"{definition.music_category.value!r}, but music source "
                    f"{definition.music_id!r} has category {music.category.value!r}"
                )

    def validate_files(self) -> None:
        """Check that all manifest-referenced audio paths exist as files."""
        missing: list[str] = []
        for entry in self.speech_entries:
            resolved = _resolve_source_path(entry.path, self.repository_root)
            if not resolved.is_file():
                missing.append(f"speech_id={entry.speech_id!r}: {resolved}")
        for entry in self.music_entries:
            resolved = _resolve_source_path(entry.path, self.repository_root)
            if not resolved.is_file():
                missing.append(f"music_id={entry.music_id!r}: {resolved}")
        if missing:
            details = "\n".join(f"- {item}" for item in missing)
            raise FileNotFoundError(f"Manifest audio files do not exist:\n{details}")

    def resolve_speech_source(self, speech_id: str) -> SpeechSource:
        """Resolve one speech entry to a loaded :class:`SpeechSource`."""
        for entry in self.speech_entries:
            if entry.speech_id == speech_id:
                return entry.to_source(repository_root=self.repository_root)
        raise KeyError(f"Unknown speech_id {speech_id!r}")

    def resolve_music_source(self, music_id: str) -> MusicSource:
        """Resolve one music entry to a loaded :class:`MusicSource`."""
        for entry in self.music_entries:
            if entry.music_id == music_id:
                return entry.to_source(repository_root=self.repository_root)
        raise KeyError(f"Unknown music_id {music_id!r}")


def _write_manifest(path: PathLike, payload: Iterable[Mapping[str, Any]]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(list(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_speech_manifest(path: PathLike, entries: Iterable[SpeechManifestEntry]) -> None:
    """Write speech entries as human-readable UTF-8 JSON."""
    _write_manifest(
        path,
        (
            {
                "speech_id": entry.speech_id,
                "path": entry.path,
                "language": entry.language,
                "speaker_id": entry.speaker_id,
                "notes": entry.notes,
            }
            for entry in entries
        ),
    )


def save_music_manifest(path: PathLike, entries: Iterable[MusicManifestEntry]) -> None:
    """Write music entries as human-readable UTF-8 JSON."""
    _write_manifest(
        path,
        (
            {
                "music_id": entry.music_id,
                "path": entry.path,
                "category": entry.category.value,
                "notes": entry.notes,
            }
            for entry in entries
        ),
    )


def save_benchmark_manifest(path: PathLike, definitions: Iterable[BenchmarkDefinition]) -> None:
    """Write benchmark definitions as human-readable UTF-8 JSON."""
    _write_manifest(
        path,
        (
            {
                "sample_id": definition.sample_id,
                "speech_id": definition.speech_id,
                "music_id": definition.music_id,
                "music_category": definition.music_category.value,
                "difficulty": definition.difficulty.value,
                "target_smr_db": definition.target_smr_db,
                "seed": definition.seed,
            }
            for definition in definitions
        ),
    )


__all__ = [
    "DEFAULT_MANIFEST_DIR",
    "ManifestError",
    "MusicManifestEntry",
    "SeparationBenchmarkManifest",
    "SpeechManifestEntry",
    "load_benchmark_manifest",
    "load_music_manifest",
    "load_speech_manifest",
    "save_benchmark_manifest",
    "save_music_manifest",
    "save_speech_manifest",
]
