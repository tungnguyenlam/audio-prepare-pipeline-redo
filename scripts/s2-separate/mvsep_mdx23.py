"""Separate stems using MVSEP-MDX23 without legacy class framework."""
from __future__ import annotations

import argparse
from collections import deque
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import ROOT, batch, completed, convert, destinations, identity, parser, positive_int, probe, publish, request

logger = logging.getLogger(__name__)

DEFAULT_REPO_URL = "https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model.git"

_UPSTREAM_DEVICE_OVERRIDE = """if __name__ == '__main__':
    import os

    gpu_use = "0"
    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
"""
_UPSTREAM_DEVICE_PATCH = """if __name__ == '__main__':
    import os

    # Device visibility is supplied by caller environment
    print('GPU visibility: {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES', 'default')))
"""
_UPSTREAM_PROGRESS_ORIGINAL = "    options = m.parse_args().__dict__\n"
_UPSTREAM_PROGRESS_PATCH = """    options = m.parse_args().__dict__
    def _print_mvsep_progress(percent):
        print('PROGRESS: {}%'.format(int(percent)), flush=True)
    options['update_percent_func'] = _print_mvsep_progress
"""
_UPSTREAM_PROGRESS_MARKER = "options['update_percent_func'] = _print_mvsep_progress"

_STEM_OUTPUT_IDS = {
    "vocals": "vocals",
    "instrumental": "instrum",
    "instrum": "instrum",
    "bass": "bass",
    "drums": "drums",
    "other": "other",
}


def _patch_upstream_device_override(inference_py: Path) -> None:
    source = inference_py.read_text(encoding="utf-8")
    original = source
    if _UPSTREAM_DEVICE_OVERRIDE in source:
        source = source.replace(_UPSTREAM_DEVICE_OVERRIDE, _UPSTREAM_DEVICE_PATCH, 1)
    elif _UPSTREAM_DEVICE_PATCH not in source:
        if 'gpu_use = "0"' in source or "gpu_use = '0'" in source:
            raise RuntimeError(f"unsupported upstream GPU-selection block in {inference_py}")
    if _UPSTREAM_PROGRESS_MARKER not in source and _UPSTREAM_PROGRESS_ORIGINAL in source:
        source = source.replace(_UPSTREAM_PROGRESS_ORIGINAL, _UPSTREAM_PROGRESS_PATCH, 1)
    if source != original:
        tmp = inference_py.with_name(f".{inference_py.name}.tmp")
        tmp.write_text(source, encoding="utf-8")
        tmp.replace(inference_py)


def _ensure_repo(repo_dir: Path, repo_url: str = DEFAULT_REPO_URL) -> Path:
    inference_py = repo_dir / "inference.py"
    if inference_py.is_file():
        (repo_dir / "models").mkdir(parents=True, exist_ok=True)
        _patch_upstream_device_override(inference_py)
        return inference_py
    if repo_dir.exists() and any(repo_dir.iterdir()):
        raise RuntimeError(f"MVSEP repo dir exists but inference.py is missing: {repo_dir}")
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning MVSEP-MDX23 repo into {repo_dir}...", file=sys.stderr)
    res = subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                         capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to clone MVSEP-MDX23: {res.stderr or res.stdout}")
    if not inference_py.is_file():
        raise RuntimeError(f"Clone finished but inference.py not found at {inference_py}")
    (repo_dir / "models").mkdir(parents=True, exist_ok=True)
    _patch_upstream_device_override(inference_py)
    return inference_py


def _prepare_inputs(src: Path, work_dir: Path, max_segment_seconds: float | None) -> list[Path]:
    input_dir = work_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    if max_segment_seconds is None:
        out_path = input_dir / f"{src.stem}.wav"
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le", str(out_path)]
    else:
        out_path = input_dir / f"{src.stem}_part_%04d.wav"
        cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "44100", "-ac", "2", "-c:a", "pcm_f32le",
               "-f", "segment", "-segment_time", str(max_segment_seconds), "-reset_timestamps", "1", str(out_path)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg input conversion failed: {res.stderr or res.stdout}")
    if max_segment_seconds is None:
        return [out_path]
    wavs = sorted(input_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError("ffmpeg input conversion produced no segments")
    return wavs


def _concat_segments(inputs: list[Path], destination: Path) -> None:
    if len(inputs) == 1:
        shutil.copy2(inputs[0], destination)
        return
    cmd = ["ffmpeg", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    labels = "".join(f"[{i}:a:0]" for i in range(len(inputs)))
    cmd.extend(["-filter_complex", f"{labels}concat=n={len(inputs)}:v=0:a=1[out]",
                "-map", "[out]", "-c:a", "pcm_f32le", str(destination)])
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {res.stderr or res.stdout}")


def _build_env(device: str) -> dict[str, str]:
    env = os.environ.copy()
    dev = device.lower()
    if dev.startswith("cuda:"):
        idx = dev.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = idx
        env["HIP_VISIBLE_DEVICES"] = idx
        env["ROCR_VISIBLE_DEVICES"] = idx
    elif dev.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = dev
        env["HIP_VISIBLE_DEVICES"] = dev
        env["ROCR_VISIBLE_DEVICES"] = dev
    elif dev == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["HIP_VISIBLE_DEVICES"] = ""
        env["ROCR_VISIBLE_DEVICES"] = ""
    return env


def main() -> int:
    p = parser(__doc__, 's2-separate', 'mvsep_mdx23')
    p.add_argument('--device', default='auto',
                   help='Compute device (e.g. auto, cpu, cuda, cuda:0) (default: auto)')
    p.add_argument('--stem', default='vocals', choices=sorted(_STEM_OUTPUT_IDS.keys()),
                   help='Target stem to separate and export (default: vocals)')
    p.add_argument('--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo) (default: 1)')
    p.add_argument('--overlap-large', type=float, default=0.25,
                   help='Overlap fraction for large sub-band models (default: 0.25)')
    p.add_argument('--overlap-small', type=float, default=0.25,
                   help='Overlap fraction for small sub-band models (default: 0.25)')
    p.add_argument('--single-onnx', action=argparse.BooleanOptionalAction, default=True,
                   help='Use single ONNX runtime session for speed (default: True)')
    p.add_argument('--large-gpu', action=argparse.BooleanOptionalAction, default=False,
                   help='Enable large GPU VRAM optimizations (default: False)')
    p.add_argument('--use-kim-model-1', action=argparse.BooleanOptionalAction, default=False,
                   help='Use Kim model 1 checkpoint variant (default: False)')
    p.add_argument('--chunk-size', type=int, default=None,
                   help='Audio processing chunk size in samples (optional; default: None)')
    p.add_argument('--max-segment-seconds', type=float, default=600.0,
                   help='Maximum segment duration in seconds before splitting (default: 600.0)')
    p.add_argument('--repo-dir', type=Path, default=None,
                   help='Path to MVSEP upstream repository directory (default: .data/mvsep_mdx23/repo)')
    args = p.parse_args()

    repo_dir = args.repo_dir or (ROOT / '.data' / 'mvsep_mdx23' / 'repo')
    inference_py = _ensure_repo(repo_dir)

    stem_id = _STEM_OUTPUT_IDS[args.stem]
    only_vocals = stem_id in ("vocals", "instrum")
    pairs = destinations(args, '_mvsep_mdx23')
    args.work_dir.mkdir(parents=True, exist_ok=True)

    use_cpu = args.device.lower() in ("cpu", "mps")

    def process(src, dest):
        rate = args.sample_rate or probe(src)['sample_rate']
        parameters = {
            'sample_rate': rate, 'channels': args.channels, 'stem': args.stem,
            'device': args.device, 'overlap_large': args.overlap_large,
            'overlap_small': args.overlap_small, 'single_onnx': args.single_onnx,
            'large_gpu': args.large_gpu, 'use_kim_model_1': args.use_kim_model_1,
            'chunk_size': args.chunk_size, 'max_segment_seconds': args.max_segment_seconds,
        }
        metadata = request(identity(src), 'separate', parameters, 'mvsep_mdx23')
        if completed(dest, metadata, args.overwrite):
            return

        with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
            work = Path(directory)
            working_wavs = _prepare_inputs(src, work, args.max_segment_seconds)
            out_folder = work / 'stems'
            out_folder.mkdir()

            cmd = [sys.executable, "-u", str(inference_py),
                   "--input_audio", *(str(w) for w in working_wavs),
                   "--output_folder", str(out_folder),
                   "--overlap_large", str(args.overlap_large),
                   "--overlap_small", str(args.overlap_small)]
            if only_vocals:
                cmd.append("--only_vocals")
            if use_cpu:
                cmd.append("--cpu")
            if args.single_onnx:
                cmd.append("--single_onnx")
            if args.large_gpu:
                cmd.append("--large_gpu")
            if args.use_kim_model_1:
                cmd.append("--use_kim_model_1")
            if args.chunk_size is not None:
                cmd.extend(["--chunk_size", str(args.chunk_size)])

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, cwd=str(repo_dir), env=_build_env(args.device))
            tail = deque(maxlen=40)
            if proc.stdout is not None:
                for line in proc.stdout:
                    msg = line.rstrip()
                    if msg:
                        tail.append(msg)
            code = proc.wait()
            if code != 0:
                detail = "\n".join(tail).strip()
                raise RuntimeError(f"MVSEP-MDX23 failed (exit {code}): {detail}")

            separated_parts = [out_folder / f"{w.stem}_{stem_id}.wav" for w in working_wavs]
            missing = [p.name for p in separated_parts if not p.is_file()]
            if missing:
                avail = sorted(p.name for p in out_folder.glob("*.wav"))
                raise RuntimeError(f"Stem {stem_id} missing for {missing}. Available: {avail}")

            merged = work / 'merged.wav'
            _concat_segments(separated_parts, merged)
            final_out = work / 'output.wav'
            convert(merged, final_out, rate, args.channels)
            publish(final_out, dest, metadata)

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


if __name__ == '__main__':
    raise SystemExit(main())
