"""Prepare blind audio batches for agent subagents that label clips by listening.

Target clips and hidden ViCut-Bench controls are converted to one format, renamed to
opaque IDs, shuffled with a fixed seed and split into small batches so each fresh
subagent labels only a few clips. The answer key is written outside the labelling
directory. No model is called here; `collect_subagent_labels.py` reads the responses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.files import ROOT, convert, digest, persist_path, positive_int, progress

DEFECTS_MANIFEST = ROOT / ".data/vicut_bench/defects/vicut_bench_defects.jsonl"
ANCHOR_DIR = ROOT / ".data/vicut_bench/anchors"
# Pre-registered control mix (WORKLOG 2026-10-16): (defect_type, severity) -> count.
CONTROL_SPEC = (
    ("clipped_word_end", "80ms", 8),
    ("clipped_word_start", "80ms", 6),
    ("secondary_speaker", "SIR_6dB", 7),
    ("background_music", "SNR_12dB", 7),
)
CLEAN_ANCHORS = 12
SILENCE_CONTROLS = 2
EXPECTED_CODE = {
    "clipped_word_end": "clipped_word_end",
    "clipped_word_start": "clipped_word_start",
    "secondary_speaker": "secondary_speaker",
    "background_music": "music_bleed",
}


def select_controls(rng: random.Random) -> list[dict]:
    """Pick defect controls and clean anchors from distinct train-split anchors."""
    rows = [json.loads(line) for line in DEFECTS_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["split"] == "train"]
    anchors = sorted({row["anchor_id"] for row in rows})
    rng.shuffle(anchors)
    used: set[str] = set()
    controls = []
    for defect_type, severity, count in CONTROL_SPEC:
        pool = {row["anchor_id"]: row for row in rows if row["defect_type"] == defect_type and row["severity"] == severity}
        picked = [anchor for anchor in anchors if anchor in pool and anchor not in used][:count]
        if len(picked) < count:
            raise ValueError(f"Not enough distinct anchors for {defect_type} {severity}")
        used.update(picked)
        for anchor in picked:
            row = pool[anchor]
            controls.append({
                "role": "control", "control_type": defect_type, "severity": severity,
                "expected_decision": "reject", "expected_code": EXPECTED_CODE[defect_type],
                "source_path": ROOT / row["audio_path"], "anchor_id": anchor,
            })
    clean = [anchor for anchor in anchors if anchor not in used][:CLEAN_ANCHORS]
    twin = {row["anchor_id"]: row["twin_audio_path"] for row in rows}
    for anchor in clean:
        controls.append({
            "role": "control", "control_type": "clean_anchor", "severity": None,
            "expected_decision": "pass", "expected_code": None,
            "source_path": ANCHOR_DIR / twin[anchor], "anchor_id": anchor,
        })
    for index in range(SILENCE_CONTROLS):
        controls.append({
            "role": "control", "control_type": "digital_silence", "severity": None,
            "expected_decision": "reject", "expected_code": None,
            "source_path": None, "silence_s": 3.0 + index,
        })
    return controls


def write_silence(dest: Path, seconds: float, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dest, np.zeros(int(seconds * sample_rate), dtype="int16"), sample_rate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-id", "--input-dir", type=Path, action="append", required=True,
                        help="Directory of target clips (*.wav, non-recursive); repeatable")
    parser.add_argument("-od", "--output-dir", type=Path, required=True, help="Labelling directory given to subagents")
    parser.add_argument("--key-file", type=Path, required=True,
                        help="Hidden answer key JSONL; must be outside --output-dir")
    parser.add_argument("-bs", "--batch-size", type=positive_int, default=6, help="Clips per subagent (default: 6)")
    parser.add_argument("-sr", "--sample-rate", type=positive_int, default=16000, help="Output sample rate (default: 16000)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle and control-selection seed (default: 42)")
    parser.add_argument("-w", "-ow", "--overwrite", action="store_true", help="Replace an existing labelling directory")
    args = parser.parse_args()

    output_dir, key_file = args.output_dir.resolve(), args.key_file.resolve()
    if key_file.is_relative_to(output_dir):
        raise SystemExit("--key-file must be outside --output-dir so subagents cannot see it")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} is not empty; pass --overwrite to replace it")

    rng = random.Random(args.seed)
    items = []
    for directory in args.input_dir:
        clips = sorted(directory.glob("*.wav"))
        if not clips:
            raise SystemExit(f"No *.wav clips in {directory}")
        items += [{"role": "target", "source_path": clip.resolve(), "source_dir": directory.resolve().name} for clip in clips]
    items += select_controls(rng)
    rng.shuffle(items)

    for name in ("audio", "batches", "responses"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    key_file.parent.mkdir(parents=True, exist_ok=True)
    width = len(str(len(items)))
    batches: dict[int, list[str]] = {}
    with key_file.open("w", encoding="utf-8") as key:
        for index, item in enumerate(items):
            item_id = f"c{index + 1:0{width}d}"
            dest = output_dir / "audio" / f"{item_id}.wav"
            if item["source_path"] is None:
                write_silence(dest, item.pop("silence_s"), args.sample_rate)
            else:
                convert(item["source_path"], dest, args.sample_rate, 1)
            batch = index // args.batch_size + 1
            batches.setdefault(batch, []).append(str(dest))
            source = item.pop("source_path")
            record = {
                "id": item_id, "batch": batch, **item,
                "source_path": persist_path(source) if source else None,
                "source_sha256": digest(source) if source else None,
                "audio_path": persist_path(dest), "audio_sha256": digest(dest),
            }
            key.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress("PREPARE", item_id, current=index + 1, total=len(items))
    for batch, paths in batches.items():
        (output_dir / "batches" / f"batch_{batch:03d}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    progress("DONE", f"{len(items)} clips in {len(batches)} batches (seed {args.seed})")
    print(output_dir)
    print(key_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
