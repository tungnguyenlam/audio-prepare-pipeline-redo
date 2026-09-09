"""Candidate-level speaker purity check with sliding identity windows and overlap vetoes."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, identity, probe, read_json, safe_name, write_json
from _common.segments import normalize_turns, source_path
from speaker.score import embed_audio_interval, get_embedder, load_profile_clips, DEFAULT_EMBEDDING_MODEL_ID


def candidate_windows(start_s: float, end_s: float, window_duration_s: float, window_hop_s: float) -> list[tuple[float, float]]:
    dur = end_s - start_s
    if dur <= window_duration_s:
        return [(start_s, end_s)]
    starts = []
    cursor = start_s
    last_full = end_s - window_duration_s
    while cursor <= last_full:
        starts.append(cursor)
        cursor += window_hop_s
    if not math.isclose(starts[-1], last_full, abs_tol=1e-9):
        starts.append(last_full)
    return [(s, s + window_duration_s) for s in starts]


def other_speaker_overlap_duration(all_turns: list[dict], spk_id: str, start_s: float, end_s: float) -> float:
    intervals = sorted(
        (max(start_s, o['start_s']), min(end_s, o['end_s']))
        for o in all_turns
        if o['speaker_id'] != spk_id and o['start_s'] < end_s and o['end_s'] > start_s
    )
    if not intervals:
        return 0.0
    merged_s = 0.0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            merged_s += cur_end - cur_start
            cur_start, cur_end = s, e
    return merged_s + (cur_end - cur_start)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='Manifest containing candidate turns')
    p.add_argument('--output-manifest', type=Path, required=True, help='Purity verification output manifest')
    p.add_argument('--profile', required=True, help='Enrolled speaker profile name or dir')
    p.add_argument('--profiles-dir', type=Path, default=ROOT / '.data' / 'speaker_profiles')
    p.add_argument('--similarity-threshold', type=float, default=0.6)
    p.add_argument('--min-candidate-duration-s', type=float, default=1.5)
    p.add_argument('--max-overlap-duration-s', type=float, default=0.05)
    p.add_argument('--window-duration-s', type=float, default=2.0)
    p.add_argument('--window-hop-s', type=float, default=0.75)
    p.add_argument('--model-id', default=DEFAULT_EMBEDDING_MODEL_ID)
    p.add_argument('--device', default='auto')
    p.add_argument('--input-file', type=Path)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    import numpy as np

    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    turns = normalize_turns(manifest.get('turns', []), source)

    profile_name, clip_paths = load_profile_clips(args.profile, args.profiles_dir)
    inference = get_embedder(args.model_id, args.device)

    clip_vectors = []
    for c in clip_paths:
        try:
            clip_vectors.append(embed_audio_interval(inference, c))
        except Exception as exc:
            print(f"Warning: Failed to embed clip {c.name}: {exc}", file=sys.stderr)
    if not clip_vectors:
        p.error(f"Failed to embed reference clips for profile {profile_name!r}")
    centroid = np.mean(np.stack(clip_vectors), axis=0)
    c_norm = float(np.linalg.norm(centroid))
    if c_norm == 0:
        p.error("Profile centroid is zero vector")
    centroid = centroid / c_norm

    verified_turns = []
    pass_count, reject_count = 0, 0
    for turn in turns:
        dur = turn['end_s'] - turn['start_s']
        spk = turn['speaker_id']
        overlap_dur = other_speaker_overlap_duration(turns, spk, turn['start_s'], turn['end_s'])

        decision = "pass"
        reason = "single_speaker"

        if overlap_dur > args.max_overlap_duration_s:
            decision = "reject"
            reason = f"other_speaker_overlap_{overlap_dur:.3f}s"
        elif dur < args.min_candidate_duration_s:
            decision = "reject"
            reason = f"duration_{dur:.3f}s_below_minimum"
        else:
            windows = candidate_windows(turn['start_s'], turn['end_s'], args.window_duration_s, args.window_hop_s)
            min_sim = 1.0
            for w_start, w_end in windows:
                try:
                    vec = embed_audio_interval(inference, source, w_start, w_end)
                    sim = float(np.clip(np.dot(centroid, vec), -1.0, 1.0))
                except Exception:
                    sim = -1.0
                min_sim = min(min_sim, sim)
                if sim < args.similarity_threshold:
                    decision = "reject"
                    reason = f"window_similarity_{sim:.3f}_below_threshold"
                    break

        if decision == "pass":
            pass_count += 1
        else:
            reject_count += 1

        verified_turns.append({
            **turn,
            'purity_decision': decision,
            'purity_reason': reason,
            'other_speaker_overlap_s': round(overlap_dur, 3),
        })

    dest = args.output_manifest.resolve()
    metadata = {
        'schema_version': 1,
        'source': identity(source),
        'timestamp_origin': 'diarized_input',
        'operation': 'speaker_purity',
        'model': args.model_id,
        'parameters': {
            'profile': profile_name,
            'similarity_threshold': args.similarity_threshold,
            'min_candidate_duration_s': args.min_candidate_duration_s,
            'max_overlap_duration_s': args.max_overlap_duration_s,
            'window_duration_s': args.window_duration_s,
            'window_hop_s': args.window_hop_s,
            'input_manifest': identity(args.input_manifest),
        },
        'turns': verified_turns,
        'speaker_ids': sorted({t['speaker_id'] for t in verified_turns}),
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
    print(dest)
    print(f'Purity check: {pass_count} passed, {reject_count} rejected', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
