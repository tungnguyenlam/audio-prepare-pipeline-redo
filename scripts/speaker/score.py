"""Score diarization turns against an enrolled speaker profile using Pyannote embeddings."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, infer_audio_family, probe, progress, read_json, safe_name, write_json
from _common.segments import normalize_turns, source_path

DEFAULT_EMBEDDING_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_EMBEDDING_DURATION_S = 0.15


def load_profile_clips(profile_name_or_dir: str, profiles_dir: Path) -> tuple[str, list[Path]]:
    p_path = Path(profile_name_or_dir)
    if p_path.is_dir() and (p_path / "profile.json").is_file():
        profile_dir = p_path
    else:
        profile_dir = profiles_dir / safe_name(profile_name_or_dir)
    manifest_path = profile_dir / "profile.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Speaker profile not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = data.get("name", profile_dir.name)
    clips = [profile_dir / "clips" / c for c in data.get("clips", [])]
    existing = [c for c in clips if c.is_file()]
    if not existing:
        raise ValueError(f"No existing clips found for profile {name!r} in {profile_dir / 'clips'}")
    return name, existing


def get_embedder(model_id: str, device: str = "auto"):
    import torch
    from pyannote.audio import Inference, Model
    token = os.getenv("HF_TOKEN")
    with contextlib.redirect_stdout(sys.stderr):
        model = Model.from_pretrained(model_id, token=token)
        if model is None:
            raise RuntimeError(f"Could not load embedding model {model_id}")
        target_device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        inference = Inference(model, window="whole")
        if target_device.type != "cpu":
            inference.to(target_device)
    return inference


def embed_audio_interval(inference, path: Path, start_s: float | None = None, end_s: float | None = None):
    import numpy as np
    import soundfile as sf
    import torch

    info = sf.info(str(path))
    start_frame = 0
    stop_frame = info.frames
    if start_s is not None and end_s is not None:
        start_frame = max(0, round(start_s * info.samplerate))
        stop_frame = min(info.frames, round(end_s * info.samplerate))
    if stop_frame <= start_frame:
        raise ValueError(f"Empty audio interval: {start_s} - {end_s}")
    waveform, sample_rate = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32", always_2d=True)
    if waveform.shape[0] == 0:
        raise ValueError(f"Empty waveform for {path}")
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1, keepdims=True)
    raw = inference({"waveform": torch.from_numpy(waveform.T.copy()), "sample_rate": int(sample_rate)})
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding")
    return vector / norm


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='Diarization segments.json')
    p.add_argument('--output-manifest', type=Path, help='Scored output manifest (default: dynamic per family under .data/speaker/score/<family>/segments.json)')
    p.add_argument('--profile', required=True, help='Enrolled speaker profile name or directory')
    p.add_argument('--profiles-dir', type=Path, default=ROOT / '.data' / 'speaker_profiles', help='Profiles root directory')
    p.add_argument('--model-id', default=DEFAULT_EMBEDDING_MODEL_ID, help='Pyannote embedding model ID')
    p.add_argument('--device', default='auto')
    p.add_argument('--input-file', type=Path, help='Optional source audio file override')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    import numpy as np

    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    turns = normalize_turns(manifest.get('turns', []), source)

    profile_name, clip_paths = load_profile_clips(args.profile, args.profiles_dir)
    progress('CENTROID', f'Embedding {len(clip_paths)} clips for profile {profile_name!r}')
    inference = get_embedder(args.model_id, args.device)

    # Compute centroid
    clip_vectors = []
    for c in clip_paths:
        try:
            clip_vectors.append(embed_audio_interval(inference, c))
        except Exception as exc:
            progress('CLIP_FAIL', f'Failed to embed profile clip {c.name}: {exc}')
    if not clip_vectors:
        p.error(f"Failed to embed any reference clips for profile {profile_name!r}")
    centroid = np.mean(np.stack(clip_vectors), axis=0)
    c_norm = float(np.linalg.norm(centroid))
    if c_norm == 0:
        p.error("Profile centroid is zero vector")
    centroid = centroid / c_norm

    total_turns = len(turns)
    progress('SCORE', f'Scoring {total_turns} turns against profile centroid')
    scored_turns = []
    step = max(1, total_turns // 10)
    for i, turn in enumerate(turns, 1):
        dur = turn['end_s'] - turn['start_s']
        sim = -1.0
        if dur >= MIN_EMBEDDING_DURATION_S:
            try:
                vec = embed_audio_interval(inference, source, turn['start_s'], turn['end_s'])
                sim = float(np.clip(np.dot(centroid, vec), -1.0, 1.0))
            except Exception:
                sim = -1.0
        overlaps_other = any(
            other['speaker_id'] != turn['speaker_id'] and other['start_s'] < turn['end_s'] and other['end_s'] > turn['start_s']
            for other in turns
        )
        scored_turns.append({
            **turn,
            'similarity': round(sim, 4),
            'overlaps_other_speaker': overlaps_other,
        })
        if i == 1 or i == total_turns or i % step == 0:
            progress('SCORE_TURN', f'{dur:.2f}s (sim={sim:.3f})', current=i, total=total_turns)

    dest = args.output_manifest.resolve() if args.output_manifest is not None else (ROOT / '.data/speaker/score' / infer_audio_family(args.input_manifest) / 'segments.json').resolve()
    metadata = {
        'schema_version': 1,
        'source': identity(source),
        'timestamp_origin': 'diarized_input',
        'operation': 'speaker_score',
        'model': args.model_id,
        'parameters': {
            'profile': profile_name,
            'model_id': args.model_id,
            'input_manifest': identity(args.input_manifest),
        },
        'turns': scored_turns,
        'speaker_ids': sorted({t['speaker_id'] for t in scored_turns}),
        'source_sample_rate': probe(source)['sample_rate'],
        'clips_valid': False,
        'complete': True,
    }
    if dest.exists() and not args.overwrite:
        old = read_json(dest)
        if all(old.get(k) == v for k, v in metadata.items() if k != 'turns'):
            print(dest)
            return 0
        p.error(f'Conflicting output: {dest}; use --overwrite')
    write_json(dest, metadata)
    progress('SCORE_DONE', f'Saved {len(scored_turns)} scored turns to {dest.name}')
    print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
