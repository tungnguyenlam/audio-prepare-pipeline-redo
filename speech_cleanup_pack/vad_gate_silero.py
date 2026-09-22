#!/usr/bin/env python3
"""Mute non-speech regions using Silero VAD with fade edges.

This removes isolated music/SFX/noise when nobody is speaking. It does NOT remove
noise or SFX that overlap speech. Use after source separation/enhancement and before
diarization when you want a speech-only timeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_utils import iter_audio, read_mono, resolve_output, safe_write


def _raised_cosine(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(1, n), dtype=np.float32)
    t = np.linspace(0.0, np.pi, n, dtype=np.float32)
    return (0.5 - 0.5 * np.cos(t)).astype(np.float32)


def make_mask(n: int, sr: int, timestamps: list[dict], pad_ms: float, fade_ms: float) -> np.ndarray:
    mask = np.zeros(n, dtype=np.float32)
    pad = int(round(pad_ms * sr / 1000.0))
    fade = int(round(fade_ms * sr / 1000.0))
    for ts in timestamps:
        s = max(0, int(round(float(ts['start']) * sr)) - pad)
        e = min(n, int(round(float(ts['end']) * sr)) + pad)
        if e <= s:
            continue
        mask[s:e] = 1.0
    if fade > 0:
        # Smooth all 0->1 / 1->0 transitions to avoid clicks.
        edges = np.diff(np.pad(mask, (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        ramp = _raised_cosine(fade)
        for s in starts:
            a, b = max(0, s - fade), s
            if b > a:
                mask[a:b] = np.maximum(mask[a:b], ramp[-(b-a):])
        for e in ends:
            a, b = e, min(n, e + fade)
            if b > a:
                mask[a:b] = np.maximum(mask[a:b], ramp[:b-a][::-1])
    return np.clip(mask, 0.0, 1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True)
    p.add_argument('-o', '--output', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--min-speech-ms', type=int, default=150)
    p.add_argument('--min-silence-ms', type=int, default=150)
    p.add_argument('--speech-pad-ms', type=int, default=80)
    p.add_argument('--fade-ms', type=float, default=8.0)
    p.add_argument('--onnx', action='store_true')
    p.add_argument('--suffix', default='_vad')
    p.add_argument('--write-json', action='store_true', help='Write speech timestamps next to output wav')
    args = p.parse_args()

    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    model = load_silero_vad(onnx=args.onnx)
    files = iter_audio(args.input)
    if not files:
        raise SystemExit('No audio files found')

    for idx, src in enumerate(files, 1):
        dest = resolve_output(src, args.input, args.output, args.suffix)
        x, sr = read_mono(src)
        vad_wav = read_audio(str(src), sampling_rate=16000)
        ts = get_speech_timestamps(
            vad_wav,
            model,
            sampling_rate=16000,
            threshold=args.threshold,
            min_speech_duration_ms=args.min_speech_ms,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            return_seconds=True,
        )
        mask = make_mask(len(x), sr, ts, pad_ms=0.0, fade_ms=args.fade_ms)
        safe_write(dest, x * mask, sr)
        if args.write_json:
            dest.with_suffix('.speech.json').write_text(json.dumps(ts, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[{idx}/{len(files)}] speech_regions={len(ts)} {src.name} -> {dest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
