#!/usr/bin/env python3
"""
scripts/evaluate/profile_tts_speaker_candidate.py

Comprehensive empirical acoustic and speaker purity audit script for G4 candidate datasets.
Measures:
  1. Duration distribution and total clean duration.
  2. Signal metrics: mean RMS (dBFS), dynamic range, estimated SNR, clipping rate, edge silence/padding.
  3. Text transcript pairing and verification.
  4. Speaker disjointness (TitaNet-L cosine similarity against dev-288, G1 test pool, and ViCut-Bench).

Usage:
  .venvs/main/bin/python scripts/evaluate/profile_tts_speaker_candidate.py --dataset qhuy242
  .venvs/main/bin/python scripts/evaluate/profile_tts_speaker_candidate.py --dataset phoaudiobook --speaker Diễm_Hân
"""

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]


def estimate_snr_db(signal: np.ndarray, frame_len: int = 512, hop_len: int = 256) -> float:
    """Estimate SNR using minimum-statistics noise floor approximation."""
    if len(signal) < frame_len:
        return 30.0
    frames = []
    for i in range(0, len(signal) - frame_len, hop_len):
        frame = signal[i : i + frame_len]
        energy = np.mean(frame ** 2)
        if energy > 1e-12:
            frames.append(energy)
    if not frames:
        return 30.0
    frames = np.array(frames)
    noise_floor = np.percentile(frames, 5)
    speech_power = np.percentile(frames, 95)
    if noise_floor <= 1e-12:
        return 45.0
    snr = 10 * np.log10(speech_power / noise_floor)
    return float(np.clip(snr, 5.0, 60.0))


def profile_audio_clips(
    audio_records: List[Dict[str, Any]],
    output_dir: Path,
    max_clips: int = None,
) -> Dict[str, Any]:
    """Profile acoustic characteristics of audio records."""
    durations = []
    rms_list = []
    snr_list = []
    dynamic_ranges = []
    clipped_clips = 0
    zero_pad_starts = []
    zero_pad_ends = []
    valid_transcripts = 0

    records_to_process = audio_records[:max_clips] if max_clips else audio_records

    manifest_entries = []

    for idx, rec in enumerate(records_to_process):
        raw_bytes = rec["audio_bytes"]
        text = rec.get("text", "").strip()
        clip_id = rec.get("id", f"clip_{idx:05d}")

        with io.BytesIO(raw_bytes) as bio:
            y, sr = sf.read(bio)

        if y.ndim > 1:
            y = np.mean(y, axis=1)

        dur = len(y) / sr
        durations.append(dur)

        # RMS
        rms = np.sqrt(np.mean(y ** 2) + 1e-12)
        rms_dbfs = 20 * np.log10(rms)
        rms_list.append(rms_dbfs)

        # Peak and dynamic range
        peak = np.max(np.abs(y))
        if peak >= 0.999:
            clipped_clips += 1
        peak_dbfs = 20 * np.log10(max(peak, 1e-6))
        dynamic_range = peak_dbfs - 20 * np.log10(np.percentile(np.abs(y) + 1e-12, 10))
        dynamic_ranges.append(dynamic_range)

        # SNR
        snr = estimate_snr_db(y)
        snr_list.append(snr)

        # Edge zero padding
        start_zeros = np.sum(np.cumsum(np.abs(y) > 1e-4) == 0) / sr
        end_zeros = np.sum(np.cumsum(np.abs(y[::-1]) > 1e-4) == 0) / sr
        zero_pad_starts.append(start_zeros)
        zero_pad_ends.append(end_zeros)

        if text:
            valid_transcripts += 1

        manifest_entries.append({
            "id": clip_id,
            "speaker": rec.get("speaker", "unknown"),
            "duration_s": round(dur, 3),
            "rms_dbfs": round(rms_dbfs, 2),
            "peak_dbfs": round(peak_dbfs, 2),
            "estimated_snr_db": round(snr, 1),
            "start_zero_pad_s": round(start_zeros, 4),
            "end_zero_pad_s": round(end_zeros, 4),
            "sample_rate": sr,
            "text": text,
        })

    total_s = sum(durations)
    return {
        "total_clips": len(durations),
        "total_duration_s": round(total_s, 2),
        "total_duration_min": round(total_s / 60.0, 2),
        "total_duration_hours": round(total_s / 3600.0, 3),
        "sample_rate": manifest_entries[0]["sample_rate"] if manifest_entries else 0,
        "duration_min_s": round(min(durations), 2) if durations else 0,
        "duration_max_s": round(max(durations), 2) if durations else 0,
        "duration_mean_s": round(float(np.mean(durations)), 2) if durations else 0,
        "duration_median_s": round(float(np.median(durations)), 2) if durations else 0,
        "mean_rms_dbfs": round(float(np.mean(rms_list)), 2) if rms_list else 0,
        "mean_snr_db": round(float(np.mean(snr_list)), 2) if snr_list else 0,
        "mean_dynamic_range_db": round(float(np.mean(dynamic_ranges)), 2) if dynamic_ranges else 0,
        "clipped_clips_count": clipped_clips,
        "clipping_rate_pct": round(100.0 * clipped_clips / len(durations), 3) if durations else 0,
        "mean_start_pad_ms": round(float(np.mean(zero_pad_starts)) * 1000.0, 2) if zero_pad_starts else 0,
        "mean_end_pad_ms": round(float(np.mean(zero_pad_ends)) * 1000.0, 2) if zero_pad_ends else 0,
        "transcript_coverage_pct": round(100.0 * valid_transcripts / len(durations), 2) if durations else 0,
        "manifest_entries": manifest_entries,
    }


def run_titanet_disjointness_check(
    sample_wavs: List[str],
    sortformer_python: str = ".venvs/sortformer/bin/python",
) -> Dict[str, Any]:
    """Execute TitaNet-L speaker disjointness verification via sortformer python venv."""
    cmd = [
        sortformer_python,
        "-c",
        f"""
import json, soundfile as sf, numpy as np, torch
import nemo.collections.asr as nemo_asr
from pathlib import Path

REPO_ROOT = Path('{REPO_ROOT}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('titanet_large').to(device)
model.eval()

def get_norm_emb(path: str) -> np.ndarray:
    with torch.no_grad():
        emb = model.get_embedding(path).detach().cpu().numpy().squeeze()
        return emb / (np.linalg.norm(emb) + 1e-12)

sample_wavs = {sample_wavs}
embs = [get_norm_emb(w) for w in sample_wavs]
centroid = np.mean(embs, axis=0)
centroid /= (np.linalg.norm(centroid) + 1e-12)

# 1. vs dev-288
dev_wavs = sorted(list((REPO_ROOT / '.data/tts_strategy/gold_benchmark_20260908/audio').glob('*.wav')))
dev_sims = [float(np.dot(centroid, get_norm_emb(str(w)))) for w in dev_wavs]
max_dev = max(dev_sims) if dev_sims else 0.0

# 2. vs G1 blind audio
g1_wavs = sorted(list((REPO_ROOT / '.data/goal/g1_blind_audio').glob('*.wav')))
g1_sims = [float(np.dot(centroid, get_norm_emb(str(w)))) for w in g1_wavs]
max_g1 = max(g1_sims) if g1_sims else 0.0

# 3. vs ViCut-Bench frozen 300 anchors
anchors_file = REPO_ROOT / '.data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl'
anchor_wavs = []
if anchors_file.exists():
    with open(anchors_file) as f:
        for line in f:
            d = json.loads(line)
            rel = d.get('rel_path')
            if rel:
                p = REPO_ROOT / '.data/vicut_bench/anchors' / rel
                if p.exists(): anchor_wavs.append(p)

anchor_sims = [float(np.dot(centroid, get_norm_emb(str(w)))) for w in anchor_wavs]
max_anchor = max(anchor_sims) if anchor_sims else 0.0

res = {{
    'max_sim_dev288': round(max_dev, 4),
    'max_sim_g1_test': round(max_g1, 4),
    'max_sim_vicut_anchors': round(max_anchor, 4),
    'calibrated_eer_threshold': 0.6944,
    'dev288_clips': len(dev_wavs),
    'g1_test_clips': len(g1_wavs),
    'vicut_anchor_clips': len(anchor_wavs),
    'verdict': 'PASS' if max(max_dev, max_g1, max_anchor) < 0.6944 else 'FAIL'
}}
print(json.dumps(res))
""",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "max_sim_dev288" in line:
            return json.loads(line)
    raise RuntimeError(f"Could not parse disjointness output: {proc.stdout}\n{proc.stderr}")


def load_candidate_records(dataset: str, speaker: str = None) -> List[Dict[str, Any]]:
    """Load candidate dataset records."""
    records = []
    if dataset == "qhuy242":
        url = "hf://datasets/qhuy242/vietnam-tts-southern-male-dataset/data/train-00000-of-00001.parquet"
        pf = pq.ParquetFile(url)
        t = pf.read(columns=["audio", "transcript"])
        df = t.to_pandas()
        for idx, row in df.iterrows():
            records.append({
                "id": f"qhuy_sm_{idx:04d}",
                "speaker": "qhuy242_southern_male",
                "audio_bytes": row["audio"]["bytes"],
                "text": row["transcript"],
            })
    elif dataset == "phoaudiobook":
        url = "hf://datasets/thivux/phoaudiobook/data/train-00000-of-00803.parquet"
        pf = pq.ParquetFile(url)
        t = pf.read(columns=["audio", "speaker", "text"])
        df = t.to_pandas()
        if speaker:
            df = df[df["speaker"] == speaker]
        for idx, row in df.iterrows():
            records.append({
                "id": f"pho_{speaker}_{idx:04d}",
                "speaker": row["speaker"],
                "audio_bytes": row["audio"]["bytes"],
                "text": row["text"],
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return records


def main():
    parser = argparse.ArgumentParser(description="Profile TTS Speaker Candidate for G4")
    parser.add_argument("--dataset", choices=["qhuy242", "phoaudiobook"], default="qhuy242")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("-om", "--out-manifest", type=str, default=str(REPO_ROOT / ".data/downstream_experiments/g4_speaker_manifest.jsonl"))
    parser.add_argument("-of", "--out-file", type=str, default=str(REPO_ROOT / ".data/downstream_experiments/g4_speaker_profile.json"))
    parser.add_argument("--skip-disjointness", action="store_true")
    args = parser.parse_args()

    speaker = args.speaker or ("Diễm_Hân" if args.dataset == "phoaudiobook" else "Southern Male")
    sys.stderr.write(f"Loading candidate records for dataset={args.dataset}, speaker={speaker}...\n")
    records = load_candidate_records(args.dataset, speaker)
    sys.stderr.write(f"Loaded {len(records)} records.\n")

    sys.stderr.write("Profiling acoustic and textual properties...\n")
    profile = profile_audio_clips(records, REPO_ROOT / ".data/downstream_experiments")

    print("\n" + "=" * 60)
    print(f"CANDIDATE ACOUSTIC PROFILE: {args.dataset.upper()} ({speaker})")
    print("=" * 60)
    print(f"Total Utterances:       {profile['total_clips']}")
    print(f"Total Clean Duration:   {profile['total_duration_hours']:.2f} hours ({profile['total_duration_min']:.2f} minutes, {profile['total_duration_s']:.1f} s)")
    print(f"Sample Rate:            {profile['sample_rate']} Hz")
    print(f"Mean Utterance Length:  {profile['duration_mean_s']} s (min={profile['duration_min_s']} s, max={profile['duration_max_s']} s)")
    print(f"Median Utterance Length:{profile['duration_median_s']} s")
    print(f"Mean RMS:               {profile['mean_rms_dbfs']} dBFS")
    print(f"Mean Dynamic Range:     {profile['mean_dynamic_range_db']} dB")
    print(f"Estimated SNR:          {profile['mean_snr_db']} dB")
    print(f"Clipping Percentage:    {profile['clipping_rate_pct']}% ({profile['clipped_clips_count']} clipped clips)")
    print(f"Edge Silence Padding:   Start={profile['mean_start_pad_ms']} ms, End={profile['mean_end_pad_ms']} ms")
    print(f"Transcript Coverage:    {profile['transcript_coverage_pct']}%")

    if not args.skip_disjointness:
        sys.stderr.write("\nRunning TitaNet-L Speaker Disjointness Verification...\n")
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_paths = []
            for i, rec in enumerate(records[:15]):
                p = os.path.join(tmpdir, f"sample_{i}.wav")
                with io.BytesIO(rec["audio_bytes"]) as bio:
                    y, sr = sf.read(bio)
                    sf.write(p, y, sr)
                sample_paths.append(p)
            disjointness = run_titanet_disjointness_check(sample_paths)

        print("-" * 60)
        print("TITANET-L SPEAKER DISJOINTNESS RESULTS:")
        print(f"  vs dev-288 ({disjointness['dev288_clips']} clips):              Max Cosine Sim = {disjointness['max_sim_dev288']:.4f}")
        print(f"  vs G1 Blind Audio Pool ({disjointness['g1_test_clips']} clips):    Max Cosine Sim = {disjointness['max_sim_g1_test']:.4f}")
        print(f"  vs ViCut-Bench Frozen Anchors ({disjointness['vicut_anchor_clips']} clips): Max Cosine Sim = {disjointness['max_sim_vicut_anchors']:.4f}")
        print(f"  Calibrated EER Threshold:               {disjointness['calibrated_eer_threshold']}")
        print(f"  VERDICT:                                {disjointness['verdict']}")
        print("=" * 60)
        profile["disjointness"] = disjointness

    # Save manifest if requested
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        for entry in profile["manifest_entries"]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    sys.stderr.write(f"\nWrote candidate manifest to: {out_manifest} ({len(profile['manifest_entries'])} entries)\n")

    # Save profile JSON summary
    if args.out_file:
        out_profile = Path(args.out_file)
        out_profile.parent.mkdir(parents=True, exist_ok=True)
        summary = {k: v for k, v in profile.items() if k != "manifest_entries"}
        with open(out_profile, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        sys.stderr.write(f"Wrote candidate profile summary to: {out_profile}\n")


if __name__ == "__main__":
    main()
