"""Diarization system factories for the ViYT-Diar benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from src.diarization.BaseDiarizer import BaseDiarizer
from src.diarization.ClusteringWorkerDiarizer import ClusteringWorkerDiarizer
from src.diarization.DiariZenWorkerDiarizer import DiariZenWorkerDiarizer
from src.diarization.PyannoteDiarizer import PyannoteDiarizer
from src.diarization.SortformerWorkerDiarizer import SortformerWorkerDiarizer
from src.diarization.ThreeDSpeakerWorkerDiarizer import ThreeDSpeakerWorkerDiarizer


@dataclass(frozen=True)
class SystemSpec:
    """One named diarization system under evaluation."""

    key: str
    label: str
    env_hint: str
    factory: Callable[..., BaseDiarizer]


def _pyannote_community(*, device: str, token: str | None, **_: Any) -> BaseDiarizer:
    return PyannoteDiarizer(
        model_id="pyannote/speaker-diarization-community-1",
        device=device,
        token=token,
    )


def _pyannote_31(*, device: str, token: str | None, **_: Any) -> BaseDiarizer:
    return PyannoteDiarizer(
        model_id="pyannote/speaker-diarization-3.1",
        device=device,
        token=token,
    )


def _sortformer(*, device: str, **_: Any) -> BaseDiarizer:
    return SortformerWorkerDiarizer(device=device)


def _clustering(*, device: str, **_: Any) -> BaseDiarizer:
    return ClusteringWorkerDiarizer(device=device)


def _diarizen(
    *, device: str, token: str | None, **_: Any
) -> BaseDiarizer:
    return DiariZenWorkerDiarizer(device=device, token=token)


def _three_d_speaker(
    *,
    device: str,
    token: str | None,
    include_overlap: bool = False,
    **_: Any,
) -> BaseDiarizer:
    return ThreeDSpeakerWorkerDiarizer(
        device=device,
        include_overlap=include_overlap,
        token=token if include_overlap else None,
    )


# Baseline first: primary-env Pyannote Community-1.
SYSTEM_REGISTRY: dict[str, SystemSpec] = {
    "pyannote_community": SystemSpec(
        key="pyannote_community",
        label="Pyannote Community-1 (baseline)",
        env_hint="primary .venv",
        factory=_pyannote_community,
    ),
    "pyannote_31": SystemSpec(
        key="pyannote_31",
        label="Pyannote 3.1",
        env_hint="primary .venv",
        factory=_pyannote_31,
    ),
    "sortformer": SystemSpec(
        key="sortformer",
        label="NeMo Sortformer",
        env_hint=".venv-sortformer",
        factory=_sortformer,
    ),
    "clustering": SystemSpec(
        key="clustering",
        label="NeMo Clustering",
        env_hint=".venv-sortformer",
        factory=_clustering,
    ),
    "diarizen": SystemSpec(
        key="diarizen",
        label="DiariZen Large s80-v2",
        env_hint=".venv-diarizen",
        factory=_diarizen,
    ),
    "3d_speaker": SystemSpec(
        key="3d_speaker",
        label="3D-Speaker",
        env_hint=".venv-3dspeaker",
        factory=_three_d_speaker,
    ),
}

BASELINE_SYSTEM = "pyannote_community"
ALL_SYSTEM_KEYS: tuple[str, ...] = tuple(SYSTEM_REGISTRY.keys())


def resolve_systems(names: list[str] | None) -> list[SystemSpec]:
    """Resolve CLI system keys into ``SystemSpec`` entries.

    Args:
        names: Explicit system keys, ``["all"]``, or ``None`` / empty for
            the baseline only. Order is preserved; duplicates are dropped.

    Returns:
        Ordered system specs.

    Raises:
        ValueError: If an unknown system key is requested.
    """
    if not names:
        return [SYSTEM_REGISTRY[BASELINE_SYSTEM]]

    resolved: list[SystemSpec] = []
    seen: set[str] = set()
    for raw in names:
        key = raw.strip().lower()
        if not key or key in seen:
            continue
        if key == "all":
            for registry_key in ALL_SYSTEM_KEYS:
                if registry_key in seen:
                    continue
                resolved.append(SYSTEM_REGISTRY[registry_key])
                seen.add(registry_key)
            continue
        if key not in SYSTEM_REGISTRY:
            known = ", ".join((*SYSTEM_REGISTRY, "all"))
            raise ValueError(f"Unknown system {raw!r}. Choose from: {known}")
        resolved.append(SYSTEM_REGISTRY[key])
        seen.add(key)
    if not resolved:
        return [SYSTEM_REGISTRY[BASELINE_SYSTEM]]
    return resolved


def build_diarizer(
    spec: SystemSpec,
    *,
    device: str,
    token: str | None = None,
    **kwargs: Any,
) -> BaseDiarizer:
    """Instantiate a diarizer for ``spec``."""
    return spec.factory(device=device, token=token, **kwargs)


__all__ = [
    "ALL_SYSTEM_KEYS",
    "BASELINE_SYSTEM",
    "SYSTEM_REGISTRY",
    "SystemSpec",
    "build_diarizer",
    "resolve_systems",
]
