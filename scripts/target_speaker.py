#!/usr/bin/env python3
"""Enroll a target speaker and filter diarization segments from the terminal.

Typical flow (run on the model server):

    # 1. Enroll from 2-3 manually cut single-speaker clips
    uv run python scripts/target_speaker.py enroll --name khanh_vy \
        --clips clip1.wav clip2.wav clip3.wav

    # 2. Score a video's diarization turns against the profile
    uv run python scripts/target_speaker.py score --audio video.wav \
        --profile khanh_vy --out scored.json

    # 3. Inspect scores, pick a threshold, filter (and optionally export cuts)
    uv run python scripts/target_speaker.py filter --scored scored.json \
        --threshold 0.6 --audio video.wav --export-dir .data/target_speaker/out
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _load_audio(path: str):
    from src.utils.AudioClass import Audio

    return Audio.from_file(path)


def _result_from_dict(payload: dict):
    from src.diarization.schemas import (
        DiarizationModelInfo,
        ScoredSegment,
        TargetSpeakerResult,
    )

    model = payload.get("model")
    return TargetSpeakerResult(
        schema_version=payload["schema_version"],
        audio_id=payload["audio_id"],
        profile_name=payload["profile_name"],
        segments=[ScoredSegment(**segment) for segment in payload["segments"]],
        model=DiarizationModelInfo(**model) if model else None,
    )


def _turns_from_json(path: str):
    """Read turns from a diarization JSON (pipeline format or raw list)."""
    from src.diarization.schemas import (
        DiarizationResult,
        Speaker,
        SpeakerTurn,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_turns = payload["turns"] if isinstance(payload, dict) else payload
    turns = [
        SpeakerTurn(
            speaker_id=str(turn["speaker_id"]),
            start_s=float(turn["start_s"]),
            end_s=float(turn["end_s"]),
        )
        for turn in raw_turns
    ]
    speakers = [
        Speaker(speaker_id=speaker_id)
        for speaker_id in sorted({turn.speaker_id for turn in turns})
    ]
    return DiarizationResult(
        schema_version="1.0",
        audio_id=str(payload.get("audio_id", "unknown")) if isinstance(payload, dict) else "unknown",
        speakers=speakers,
        turns=turns,
    )


def cmd_enroll(args: argparse.Namespace) -> None:
    from src.diarization.SpeakerVerifier import SpeakerVerifier

    verifier = SpeakerVerifier(profiles_dir=args.profiles_dir)
    clips = [_load_audio(clip) for clip in args.clips]
    profile = verifier.enroll(args.name, clips, overwrite=args.overwrite)
    print(f"Enrolled profile {profile.name!r} with {len(profile.clip_paths)} clips")
    print(f"Profile dir: {profile.profile_dir}")


def cmd_profiles(args: argparse.Namespace) -> None:
    from src.diarization.SpeakerVerifier import SpeakerVerifier

    verifier = SpeakerVerifier(profiles_dir=args.profiles_dir)
    names = verifier.list_profiles()
    if not names:
        print("No profiles found.")
        return
    for name in names:
        profile = verifier.load_profile(name)
        print(f"{name}  ({len(profile.clip_paths)} clips, created {profile.created_at})")


def cmd_score(args: argparse.Namespace) -> None:
    from src.diarization.SpeakerVerifier import SpeakerVerifier

    audio = _load_audio(args.audio)

    if args.turns_json:
        result = _turns_from_json(args.turns_json)
    else:
        from src.diarization.PyannoteDiarizer import PyannoteDiarizer

        print("No --turns-json given; running Pyannote diarization first...")
        with PyannoteDiarizer(device=args.device) as diarizer:
            result = diarizer.diarize(audio, num_speakers=args.num_speakers)
        print(f"Diarization: {len(result.speakers)} speakers, {len(result.turns)} turns")

    verifier = SpeakerVerifier(device=args.device, profiles_dir=args.profiles_dir)
    profile = verifier.load_profile(args.profile)
    with verifier:
        scored = verifier.score(audio, result, profile)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(asdict(scored), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    similarities = sorted(
        (segment.similarity for segment in scored.segments), reverse=True
    )
    print(f"Scored {len(scored.segments)} segments -> {out_path}")
    if similarities:
        print(f"Similarity: max={similarities[0]:.3f} min={similarities[-1]:.3f}")
        for threshold in (0.7, 0.6, 0.5, 0.4):
            kept = sum(1 for value in similarities if value >= threshold)
            print(f"  >= {threshold:.1f}: {kept} segments")


def cmd_filter(args: argparse.Namespace) -> None:
    from src.diarization.SpeakerVerifier import SpeakerVerifier

    scored = _result_from_dict(
        json.loads(Path(args.scored).read_text(encoding="utf-8"))
    )
    kept = SpeakerVerifier.filter(
        scored,
        threshold=args.threshold,
        min_duration_s=args.min_duration,
        exclude_overlap=not args.include_overlap,
    )
    total_s = sum(segment.duration_s for segment in kept.segments)
    print(
        f"Kept {len(kept.segments)}/{len(scored.segments)} segments "
        f"({total_s:.1f}s total speech)"
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(asdict(kept), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Filtered result -> {out_path}")

    if args.export_dir:
        if not args.audio:
            raise SystemExit("--export-dir requires --audio to cut segments from")
        from src.utils.AudioCutter import AudioCutter

        audio = _load_audio(args.audio)
        cutter = AudioCutter(output_dir=args.export_dir)
        for segment in kept.segments:
            clip = cutter.cut(audio, segment.start_s, segment.end_s)
            print(f"  exported {clip.path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Target speaker enrollment and segment filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--profiles-dir",
        default=None,
        help="Speaker profiles directory (default: .data/speaker_profiles)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll = subparsers.add_parser("enroll", help="Create a speaker profile")
    enroll.add_argument("--name", required=True, help="Profile name")
    enroll.add_argument(
        "--clips", nargs="+", required=True, help="Single-speaker reference clips"
    )
    enroll.add_argument("--overwrite", action="store_true")
    enroll.set_defaults(func=cmd_enroll)

    profiles = subparsers.add_parser("profiles", help="List stored profiles")
    profiles.set_defaults(func=cmd_profiles)

    score = subparsers.add_parser(
        "score", help="Score diarization turns against a profile"
    )
    score.add_argument("--audio", required=True, help="Audio file to score")
    score.add_argument("--profile", required=True, help="Profile name")
    score.add_argument(
        "--turns-json",
        default=None,
        help="Diarization JSON with a turns list; omit to run Pyannote diarization",
    )
    score.add_argument("--num-speakers", type=int, default=None)
    score.add_argument("--device", default="auto")
    score.add_argument("--out", default="scored.json", help="Output JSON path")
    score.set_defaults(func=cmd_score)

    filter_cmd = subparsers.add_parser(
        "filter", help="Threshold a scored result and optionally export cuts"
    )
    filter_cmd.add_argument("--scored", required=True, help="JSON from 'score'")
    filter_cmd.add_argument("--threshold", type=float, required=True)
    filter_cmd.add_argument("--min-duration", type=float, default=1.5)
    filter_cmd.add_argument(
        "--include-overlap",
        action="store_true",
        help="Keep segments overlapping other speakers (dropped by default)",
    )
    filter_cmd.add_argument("--out", default=None, help="Filtered JSON output path")
    filter_cmd.add_argument(
        "--export-dir", default=None, help="Cut kept segments into this directory"
    )
    filter_cmd.add_argument(
        "--audio", default=None, help="Source audio for --export-dir"
    )
    filter_cmd.set_defaults(func=cmd_filter)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
