from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_EXTS = {'.wav', '.flac', '.ogg'}


def iter_audio(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def resolve_output(src: Path, input_root: Path, output: Path, suffix: str = '') -> Path:
    src = Path(src)
    input_root = Path(input_root)
    output = Path(output)
    if input_root.is_file():
        if output.suffix:
            return output
        return output / f'{src.stem}{suffix}.wav'
    rel = src.relative_to(input_root)
    return output / rel.parent / f'{rel.stem}{suffix}.wav'


def read_mono(path: Path, dtype: str = 'float32') -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype=dtype, always_2d=True)
    x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float32), int(sr)


def resample_audio(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return np.asarray(x, dtype=np.float32)
    g = math.gcd(int(src_sr), int(dst_sr))
    y = resample_poly(x, dst_sr // g, src_sr // g)
    return np.asarray(y, dtype=np.float32)


def safe_write(path: Path, audio: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.999
    sf.write(str(path), y, int(sr), subtype='FLOAT')
