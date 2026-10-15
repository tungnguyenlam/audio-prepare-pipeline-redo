#!/usr/bin/env python3
"""Probe frozen speech and audio-LLM encoder layers on ViCut-Bench and dev gold set.

Implements GRAND_PLAN Phase 3 (Probes and probe-head gatekeeper, G2/G3):
  1. Extracts per-layer representations from frozen encoders:
     - `wavlm`        : microsoft/wavlm-base (L0..L12, dim 768, 20ms hop)
     - `phowhisper`   : vinai/PhoWhisper-small encoder (L0..L12, dim 768, 20ms hop, unpadded)
     - `gemma4_audio` : google/gemma-4-E2B-it USM audio tower (L0..L12, dim 1024/1536, ~40ms hop)
  2. Computes four pooling summaries over valid (unpadded) frames at every layer:
     - `mean`       : global temporal mean
     - `std`        : global temporal standard deviation
     - `start_edge` : mean of first ~100 ms of frames
     - `end_edge`   : mean of last ~100 ms of valid frames
  3. Trains L2-regularized logistic regression probes on the ViCut-Bench `train` split
     (210 anchors, 210 tight-cut 0ms clean twins, 3,360 defects; 43 speakers) and evaluates
     on the speaker-disjoint `test` split (90 anchors, 90 tight-cut twins, 1,440 defects; 19 speakers):
     - Per-layer AUROC curves across all 4 defect types
     - Pooling component ablation (`mean` vs `mean_std` vs `edge` vs `combined`)
     - Clean-twin boundary control (Regime A raw-anchor negatives vs Regime B twin-calibrated
       negatives tested against 0ms tight-cut clean twins, GOAL.md §2 Rule 4)
     - Label-shuffle sanity check and speaker-cluster bootstrap 95% CIs vs signal baselines
  4. Evaluates zero-shot transfer of the probe-head gatekeeper on the 288-clip real dev set
     (`.data/tts_strategy/gold_benchmark_20260908/eval_input.materialized.jsonl`, diagnostic only).
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from safetensors import safe_open
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

if "HF_HOME" not in os.environ and (Path.home() / ".cache/huggingface/hub").is_dir():
    os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (
    LoggingArgumentParser,
    persist_path,
    progress,
    resolve_stored_path,
    write_json,
)
from vicut_bench.score_signal_baselines import compute_auroc, extract_signal_features

logger = logging.getLogger("vicut_bench.probe_encoder_layers")

DEFECT_TYPES = (
    "clipped_word_end",
    "clipped_word_start",
    "secondary_speaker",
    "background_music",
)

SEVERITY_ORDER = {
    "clipped_word_end": ["10ms", "20ms", "40ms", "80ms"],
    "clipped_word_start": ["10ms", "20ms", "40ms", "80ms"],
    "secondary_speaker": ["SIR_24dB", "SIR_18dB", "SIR_12dB", "SIR_6dB"],
    "background_music": ["SNR_30dB", "SNR_24dB", "SNR_18dB", "SNR_12dB"],
}

BASELINE_SPEC = {
    "clipped_word_end": ("end_edge_ratio_10ms", True),
    "clipped_word_start": ("start_edge_ratio_10ms", True),
    "secondary_speaker": ("window_rms_var", True),
    "background_music": ("min_window_rms_db", True),
}


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pool_layer_frames(
    hidden_seq: torch.Tensor,
    valid_len: int,
    edge_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool valid frames [0:valid_len] into (mean, std, start_edge, end_edge)."""
    vlen = max(1, min(int(valid_len), int(hidden_seq.shape[0])))
    valid = hidden_seq[:vlen].float()
    mean_vec = valid.mean(dim=0)
    if vlen > 1:
        std_vec = valid.std(dim=0, unbiased=False)
    else:
        std_vec = torch.zeros_like(mean_vec)
    k = max(1, min(vlen, edge_frames))
    start_vec = valid[:k].mean(dim=0)
    end_vec = valid[-k:].mean(dim=0)
    return (
        mean_vec.cpu().numpy().astype(np.float32),
        std_vec.cpu().numpy().astype(np.float32),
        start_vec.cpu().numpy().astype(np.float32),
        end_vec.cpu().numpy().astype(np.float32),
    )


def load_clip_records(
    anchors_manifest: Path,
    defects_manifest: Path,
    dev_gold_manifest: Path | None,
) -> list[dict[str, Any]]:
    """Build ordered list of all clip records to extract (anchors, tight twins, defects, dev_gold)."""
    records: list[dict[str, Any]] = []

    anchors = [
        json.loads(line)
        for line in anchors_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    anchor_base = anchors_manifest.parent
    for a in anchors:
        raw_path = a.get("audio_path") or a.get("rel_path")
        apath = Path(raw_path)
        if not apath.is_file():
            apath = anchor_base / raw_path
        records.append({
            "uid": f"anchor__{a['id']}",
            "clip_id": a["id"],
            "group": "clean_anchor",
            "defect_type": "clean",
            "severity": "clean",
            "split": a["split"],
            "speaker": a["speaker"],
            "audio_path": str(apath),
            "slice_start_s": None,
            "slice_end_s": None,
            "duration_s": float(a.get("duration_s", 0.0)),
        })
        # Tight-cut clean twin (delta_t = 0 ms at [speech_onset_s, speech_offset_s])
        onset_s = float(a.get("speech_onset_s", 0.0))
        offset_s = float(a.get("speech_offset_s", a.get("duration_s", 0.0)))
        records.append({
            "uid": f"tight_twin__{a['id']}",
            "clip_id": f"{a['id']}__tight_0ms",
            "anchor_id": a["id"],
            "group": "clean_tight_twin",
            "defect_type": "clean_tight",
            "severity": "0ms",
            "split": a["split"],
            "speaker": a["speaker"],
            "audio_path": str(apath),
            "slice_start_s": onset_s,
            "slice_end_s": offset_s,
            "duration_s": max(0.1, offset_s - onset_s),
        })

    defects = [
        json.loads(line)
        for line in defects_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    defect_base = defects_manifest.parent
    for d in defects:
        raw_path = d["audio_path"]
        dpath = Path(raw_path)
        if not dpath.is_file():
            dpath = defect_base / raw_path
        records.append({
            "uid": f"defect__{d['id']}",
            "clip_id": d["id"],
            "anchor_id": d["anchor_id"],
            "group": "defect",
            "defect_type": d["defect_type"],
            "severity": str(d["severity"]),
            "split": d["split"],
            "speaker": d["speaker"],
            "audio_path": str(dpath),
            "slice_start_s": None,
            "slice_end_s": None,
            "duration_s": float(d.get("duration_s", 0.0)),
        })

    if dev_gold_manifest and dev_gold_manifest.is_file():
        gold_rows = [
            json.loads(line)
            for line in dev_gold_manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for g in gold_rows:
            gpath = Path(g["audio_path"])
            dims = g.get("gemini", {}).get("dimensions", {})
            records.append({
                "uid": f"dev_gold__{g['id']}",
                "clip_id": g["id"],
                "group": "dev_gold",
                "defect_type": g.get("decision", "unknown"),
                "severity": "real",
                "split": "dev_gold",
                "speaker": g.get("recording_id", "unknown"),
                "audio_path": str(gpath),
                "slice_start_s": None,
                "slice_end_s": None,
                "duration_s": float(g.get("duration_s", 0.0)),
                "decision": g.get("decision", "unknown"),
                "dimensions": dims,
            })

    return records


def read_audio_for_record(rec: dict[str, Any]) -> tuple[np.ndarray, int]:
    data, sr = sf.read(rec["audio_path"], dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if rec.get("slice_start_s") is not None and rec.get("slice_end_s") is not None:
        s0 = max(0, int(round(float(rec["slice_start_s"]) * sr)))
        s1 = min(len(audio), max(s0 + 160, int(round(float(rec["slice_end_s"]) * sr))))
        audio = audio[s0:s1].copy()
    return audio, sr


def extract_or_load_signal_features(
    records: list[dict[str, Any]],
    cache_dir: Path,
    overwrite: bool = False,
) -> dict[str, dict[str, float]]:
    cache_path = cache_dir / "signal_features.json"
    if cache_path.is_file() and not overwrite:
        logger.info("Loading cached signal baseline features from %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    logger.info("Computing deterministic signal features for %d records...", len(records))
    sig_map: dict[str, dict[str, float]] = {}
    for idx, rec in enumerate(records, 1):
        if idx % 500 == 0 or idx == len(records):
            progress("SIGNAL_FEATS", f"{idx}/{len(records)}", current=idx, total=len(records))
        audio, sr = read_audio_for_record(rec)
        sig_map[rec["uid"]] = extract_signal_features(audio, sr)

    write_json(cache_path, sig_map)
    return sig_map


def extract_encoder_features(
    encoder_name: str,
    records: list[dict[str, Any]],
    cache_dir: Path,
    device: str,
    batch_size: int = 16,
    overwrite: bool = False,
) -> dict[str, np.ndarray]:
    """Extract or load cached [N, num_layers, D] arrays for mean, std, start_edge, end_edge."""
    cache_file = cache_dir / f"{encoder_name}_pooled.npz"
    if cache_file.is_file() and not overwrite:
        logger.info("Loading cached pooled features for %s from %s", encoder_name, cache_file)
        loaded = np.load(cache_file, allow_pickle=False)
        saved_uids = list(loaded["uids"])
        target_uids = [r["uid"] for r in records]
        if saved_uids == target_uids:
            return {
                "mean": loaded["mean"],
                "std": loaded["std"],
                "start_edge": loaded["start_edge"],
                "end_edge": loaded["end_edge"],
                "layer_dims": loaded["layer_dims"],
            }
        logger.info("UID list changed for %s; re-extracting...", encoder_name)

    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoProcessor,
        WavLMModel,
        WhisperModel,
    )
    import transformers.models.gemma4.modeling_gemma4 as mg4

    n_recs = len(records)
    # Sort indices by duration to minimize batch padding and ROCm MIOpen kernel recompilations
    sorted_indices = sorted(range(n_recs), key=lambda i: records[i]["duration_s"])

    t0 = time.time()
    if encoder_name == "wavlm":
        model_id = "microsoft/wavlm-base"
        fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        model = (
            WavLMModel.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.bfloat16)
            .to(device)
            .eval()
        )
        num_layers = 13
        layer_dims = np.array([768] * num_layers, dtype=np.int32)
        edge_frames = 5  # 5 * 20ms = 100ms
        max_dim = 768
    elif encoder_name == "phowhisper":
        model_id = "vinai/PhoWhisper-small"
        fe = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        model = (
            WhisperModel.from_pretrained(model_id, local_files_only=True, torch_dtype=torch.bfloat16)
            .encoder.to(device)
            .eval()
        )
        num_layers = 13
        layer_dims = np.array([768] * num_layers, dtype=np.int32)
        edge_frames = 5  # 5 * 20ms = 100ms
        max_dim = 768
    elif encoder_name == "gemma4_audio":
        model_id = "google/gemma-4-E2B-it"
        proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        fe = proc.feature_extractor
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
        model = mg4.Gemma4AudioModel(cfg.audio_config)
        hub = Path.home() / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
        snap = next(hub.iterdir())
        sd = {}
        with safe_open(snap / "model.safetensors", framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.audio_tower."):
                    sd[k[len("model.audio_tower."):]] = f.get_tensor(k)
        model.load_state_dict(sd, strict=True)
        model = model.to(device=device, dtype=torch.bfloat16).eval()
        num_layers = 13
        layer_dims = np.array([1024] * 12 + [1536], dtype=np.int32)
        edge_frames = 3  # 3 * 40ms = 120ms
        max_dim = 1536
    else:
        raise ValueError(f"Unsupported encoder: {encoder_name}")

    mean_arr = np.zeros((n_recs, num_layers, max_dim), dtype=np.float32)
    std_arr = np.zeros((n_recs, num_layers, max_dim), dtype=np.float32)
    start_arr = np.zeros((n_recs, num_layers, max_dim), dtype=np.float32)
    end_arr = np.zeros((n_recs, num_layers, max_dim), dtype=np.float32)

    eff_bs = batch_size * 2 if encoder_name == "phowhisper" else batch_size
    logger.info("Extracting %s across %d clips (batch_size=%d)...", encoder_name, n_recs, eff_bs)

    processed = 0
    with torch.no_grad():
        for b_start in range(0, n_recs, eff_bs):
            batch_idxs = sorted_indices[b_start : b_start + eff_bs]
            batch_audios = []
            batch_durs = []
            for idx in batch_idxs:
                audio, sr = read_audio_for_record(records[idx])
                batch_audios.append(audio)
                batch_durs.append(len(audio) / float(sr))

            if encoder_name == "wavlm":
                inp = fe(
                    batch_audios,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    pad_to_multiple_of=32000,
                    return_attention_mask=True,
                )
                out = model(
                    inp["input_values"].to(device, dtype=torch.bfloat16),
                    attention_mask=inp["attention_mask"].to(device),
                    output_hidden_states=True,
                )
                valid_lens = model._get_feat_extract_output_lengths(
                    inp["attention_mask"].sum(dim=-1)
                ).tolist()
                hidden_states = out.hidden_states
            elif encoder_name == "phowhisper":
                inp = fe(batch_audios, sampling_rate=16000, return_tensors="pt")
                out = model(
                    inp["input_features"].to(device, dtype=torch.bfloat16),
                    output_hidden_states=True,
                )
                valid_lens = [
                    min(1500, max(1, int(round(d * 50.0)))) for d in batch_durs
                ]
                hidden_states = out.hidden_states
            elif encoder_name == "gemma4_audio":
                inp = fe(batch_audios, sampling_rate=16000, return_tensors="pt")
                out = model(
                    input_features=inp["input_features"].to(device, dtype=torch.bfloat16),
                    attention_mask=inp["input_features_mask"].to(device),
                    output_hidden_states=True,
                )
                valid_lens = out.attention_mask.sum(dim=-1).cpu().tolist()
                hidden_states = out.hidden_states

            for b_pos, orig_idx in enumerate(batch_idxs):
                vlen = int(valid_lens[b_pos])
                for l_idx in range(num_layers):
                    ldim = int(layer_dims[l_idx])
                    h_seq = hidden_states[l_idx][b_pos, :, :ldim]
                    m_v, s_v, st_v, en_v = pool_layer_frames(h_seq, vlen, edge_frames)
                    mean_arr[orig_idx, l_idx, :ldim] = m_v
                    std_arr[orig_idx, l_idx, :ldim] = s_v
                    start_arr[orig_idx, l_idx, :ldim] = st_v
                    end_arr[orig_idx, l_idx, :ldim] = en_v

            processed += len(batch_idxs)
            if processed % 600 < eff_bs or processed == n_recs:
                elapsed = time.time() - t0
                progress(
                    f"EXTRACT_{encoder_name.upper()}",
                    f"{processed}/{n_recs} ({elapsed:.1f}s)",
                    current=processed,
                    total=n_recs,
                )

    del model
    cleanup_gpu()

    uids_arr = np.array([r["uid"] for r in records])
    np.savez_compressed(
        cache_file,
        uids=uids_arr,
        mean=mean_arr,
        std=std_arr,
        start_edge=start_arr,
        end_edge=end_arr,
        layer_dims=layer_dims,
    )
    logger.info("Saved %s pooled representations to %s (%.1fs)", encoder_name, cache_file, time.time() - t0)
    return {
        "mean": mean_arr,
        "std": std_arr,
        "start_edge": start_arr,
        "end_edge": end_arr,
        "layer_dims": layer_dims,
    }


def build_feature_matrix(
    feats: dict[str, np.ndarray],
    indices: list[int],
    layer_idx: int,
    pooling_mode: str,
    defect_type: str | None = None,
) -> np.ndarray:
    ldim = int(feats["layer_dims"][layer_idx])
    m = feats["mean"][indices, layer_idx, :ldim]
    s = feats["std"][indices, layer_idx, :ldim]
    st = feats["start_edge"][indices, layer_idx, :ldim]
    en = feats["end_edge"][indices, layer_idx, :ldim]

    if pooling_mode == "mean":
        return m
    if pooling_mode == "mean_std":
        return np.concatenate([m, s], axis=1)
    if pooling_mode == "edge":
        if defect_type == "clipped_word_end":
            return en
        if defect_type == "clipped_word_start":
            return st
        return np.concatenate([st, en], axis=1)
    if pooling_mode == "combined":
        return np.concatenate([m, s, st, en], axis=1)
    raise ValueError(f"Unknown pooling_mode: {pooling_mode}")


def bootstrap_auroc_ci(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    pos_speakers: list[str],
    neg_speakers: list[str],
    base_pos_scores: np.ndarray | None = None,
    base_neg_scores: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Speaker-cluster bootstrap 95% CI for AUROC and (AUROC - baseline_AUROC)."""
    rng = np.random.default_rng(seed)
    unique_speakers = sorted(set(pos_speakers) | set(neg_speakers))
    spk_to_pos = {
        spk: np.where(np.array(pos_speakers) == spk)[0] for spk in unique_speakers
    }
    spk_to_neg = {
        spk: np.where(np.array(neg_speakers) == spk)[0] for spk in unique_speakers
    }

    aurocs = []
    diffs = []
    n_spk = len(unique_speakers)
    for _ in range(n_boot):
        sampled_spks = rng.choice(unique_speakers, size=n_spk, replace=True)
        p_idx = np.concatenate([spk_to_pos[s] for s in sampled_spks if len(spk_to_pos[s]) > 0])
        n_idx = np.concatenate([spk_to_neg[s] for s in sampled_spks if len(spk_to_neg[s]) > 0])
        if len(p_idx) == 0 or len(n_idx) == 0:
            continue
        auc = compute_auroc(pos_scores[p_idx].tolist(), neg_scores[n_idx].tolist(), higher_is_pos=True)
        aurocs.append(auc)
        if base_pos_scores is not None and base_neg_scores is not None:
            b_auc = compute_auroc(
                base_pos_scores[p_idx].tolist(),
                base_neg_scores[n_idx].tolist(),
                higher_is_pos=True,
            )
            diffs.append(auc - b_auc)

    res = {
        "ci_low": round(float(np.percentile(aurocs, 2.5)), 4),
        "ci_high": round(float(np.percentile(aurocs, 97.5)), 4),
    }
    if diffs:
        res["diff_mean"] = round(float(np.mean(diffs)), 4)
        res["diff_ci_low"] = round(float(np.percentile(diffs, 2.5)), 4)
        res["diff_ci_high"] = round(float(np.percentile(diffs, 97.5)), 4)
    return res


def fit_and_score_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval_dict: dict[str, np.ndarray],
    c_reg: float = 0.01,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(
        C=c_reg,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(X_tr_s, y_train)
    out = {}
    for k, X_ev in X_eval_dict.items():
        X_ev_s = scaler.transform(X_ev)
        out[k] = clf.predict_proba(X_ev_s)[:, 1]
    return out


def main() -> int:
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchors-manifest",
        type=Path,
        default=Path(".data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl"),
        help="Path to clean anchors JSONL.",
    )
    parser.add_argument(
        "--defects-manifest",
        type=Path,
        default=Path(".data/vicut_bench/defects/vicut_bench_defects.jsonl"),
        help="Path to synthesized defects JSONL.",
    )
    parser.add_argument(
        "--dev-gold-manifest",
        type=Path,
        default=Path(".data/tts_strategy/gold_benchmark_20260908/eval_input.materialized.jsonl"),
        help="Path to 288-clip dev gold benchmark JSONL (diagnostic only).",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        default="wavlm,phowhisper,gemma4_audio",
        help="Comma-separated encoder names to probe.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for GPU encoder feature extraction.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for feature extraction.",
    )
    parser.add_argument(
        "-od",
        "--output-dir",
        type=Path,
        default=Path(".data/vicut_bench/reports/phase3_probes"),
        help="Directory for cached features and probe evaluation reports.",
    )
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help="Re-extract encoder features even if cache exists.",
    )
    args = parser.parse_args()

    if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1":
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"

    out_dir = resolve_stored_path(args.output_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = load_clip_records(
        resolve_stored_path(args.anchors_manifest),
        resolve_stored_path(args.defects_manifest),
        resolve_stored_path(args.dev_gold_manifest) if args.dev_gold_manifest else None,
    )
    logger.info("Loaded %d total clip records across splits.", len(records))

    # 1. Extract or load deterministic signal baselines for exact test-split comparison
    sig_map = extract_or_load_signal_features(records, cache_dir, overwrite=args.overwrite)

    # Index partitions
    train_anchor_idxs = [
        i for i, r in enumerate(records) if r["group"] == "clean_anchor" and r["split"] == "train"
    ]
    test_anchor_idxs = [
        i for i, r in enumerate(records) if r["group"] == "clean_anchor" and r["split"] == "test"
    ]
    train_tight_idxs = [
        i for i, r in enumerate(records) if r["group"] == "clean_tight_twin" and r["split"] == "train"
    ]
    test_tight_idxs = [
        i for i, r in enumerate(records) if r["group"] == "clean_tight_twin" and r["split"] == "test"
    ]
    dev_gold_idxs = [i for i, r in enumerate(records) if r["group"] == "dev_gold"]

    test_anchor_speakers = [records[i]["speaker"] for i in test_anchor_idxs]
    test_tight_speakers = [records[i]["speaker"] for i in test_tight_idxs]

    # Evaluate deterministic signal baselines on the held-out test split (90 anchors + 90 tight twins)
    baseline_test_results: dict[str, Any] = {}
    baseline_test_scores: dict[str, dict[str, np.ndarray]] = {}

    for dtype in DEFECT_TYPES:
        feat_name, higher_is_pos = BASELINE_SPEC[dtype]
        sign = 1.0 if higher_is_pos else -1.0
        test_def_idxs = [
            i
            for i, r in enumerate(records)
            if r["group"] == "defect" and r["defect_type"] == dtype and r["split"] == "test"
        ]
        test_def_speakers = [records[i]["speaker"] for i in test_def_idxs]

        neg_raw = np.array([ sign * sig_map[records[i]["uid"]][feat_name] for i in test_anchor_idxs ])
        neg_tight = np.array([ sign * sig_map[records[i]["uid"]][feat_name] for i in test_tight_idxs ])
        pos_def = np.array([ sign * sig_map[records[i]["uid"]][feat_name] for i in test_def_idxs ])

        baseline_test_scores[dtype] = {
            "pos": pos_def,
            "neg_raw": neg_raw,
            "neg_tight": neg_tight,
        }

        auroc_vs_raw = compute_auroc(pos_def.tolist(), neg_raw.tolist(), higher_is_pos=True)
        auroc_vs_tight = compute_auroc(pos_def.tolist(), neg_tight.tolist(), higher_is_pos=True)
        thresh_5pct_raw = float(np.percentile(neg_raw, 95))
        frr_tight_at_5pct_raw = float(np.mean(neg_tight >= thresh_5pct_raw))
        thresh_5pct_tight = float(np.percentile(neg_tight, 95))

        sev_breakdown = {}
        for sev in SEVERITY_ORDER[dtype]:
            s_idxs = [i for i in test_def_idxs if records[i]["severity"] == sev]
            s_scores = np.array([sign * sig_map[records[i]["uid"]][feat_name] for i in s_idxs])
            sev_breakdown[sev] = {
                "auroc_vs_raw": round(compute_auroc(s_scores.tolist(), neg_raw.tolist(), True), 4),
                "auroc_vs_tight": round(compute_auroc(s_scores.tolist(), neg_tight.tolist(), True), 4),
                "det_at_5pct_raw_frr": round(float(np.mean(s_scores >= thresh_5pct_raw)), 4),
                "det_at_5pct_tight_frr": round(float(np.mean(s_scores >= thresh_5pct_tight)), 4),
            }

        ci_raw = bootstrap_auroc_ci(pos_def, neg_raw, test_def_speakers, test_anchor_speakers)
        baseline_test_results[dtype] = {
            "feature": feat_name,
            "test_auroc_vs_raw": round(auroc_vs_raw, 4),
            "test_auroc_vs_raw_ci": [ci_raw["ci_low"], ci_raw["ci_high"]],
            "test_auroc_vs_tight_0ms": round(auroc_vs_tight, 4),
            "tight_twin_frr_at_5pct_raw_thresh": round(frr_tight_at_5pct_raw, 4),
            "severities": sev_breakdown,
        }

    # Trivial baselines (duration, RMS, peak) on the held-out test split (GOAL.md §3 control)
    trivial_baselines: dict[str, dict[str, float]] = {}
    for dtype in DEFECT_TYPES:
        test_def_idxs = [
            i
            for i, r in enumerate(records)
            if r["group"] == "defect" and r["defect_type"] == dtype and r["split"] == "test"
        ]
        dur_neg = [records[i]["duration_s"] for i in test_anchor_idxs]
        dur_pos = [records[i]["duration_s"] for i in test_def_idxs]
        auc_dur = compute_auroc(dur_pos, dur_neg, True)
        trivial_baselines[dtype] = {
            "duration_auroc": round(max(auc_dur, 1.0 - auc_dur), 4),
        }

    # Map defect_type -> strict defect-matched pooling mode (eliminates opposite-boundary shortcut)
    matched_pooling_map = {
        "clipped_word_end": "edge",      # strictly end_edge (last ~100ms)
        "clipped_word_start": "edge",    # strictly start_edge (first ~100ms)
        "secondary_speaker": "mean_std", # global mean + temporal std
        "background_music": "mean_std",  # global mean + temporal std
    }

    # Load defect metadata to partition secondary_speaker test clips into unseen-test-donor vs seen-train-donor
    raw_defects_meta = {
        d["id"]: d
        for d in (
            json.loads(line)
            for line in resolve_stored_path(args.defects_manifest)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
    }
    train_speaker_set = {records[i]["speaker"] for i in train_anchor_idxs}
    test_speaker_set = {records[i]["speaker"] for i in test_anchor_idxs}

    # 2. Extract and probe each encoder
    encoder_list = [m.strip() for m in args.models.split(",") if m.strip()]
    all_encoder_results: dict[str, Any] = {}
    dev_gold_probe_probs: dict[str, dict[str, np.ndarray]] = {}

    for enc_name in encoder_list:
        feats = extract_encoder_features(
            enc_name,
            records,
            cache_dir=cache_dir,
            device=args.device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
        num_layers = len(feats["layer_dims"])
        enc_res: dict[str, Any] = {
            "num_layers": num_layers,
            "layer_dims": feats["layer_dims"].tolist(),
            "defects": {},
        }
        dev_gold_probe_probs[enc_name] = {}

        for dtype in DEFECT_TYPES:
            primary_pmode = matched_pooling_map[dtype]
            train_def_idxs = [
                i
                for i, r in enumerate(records)
                if r["group"] == "defect" and r["defect_type"] == dtype and r["split"] == "train"
            ]
            test_def_idxs = [
                i
                for i, r in enumerate(records)
                if r["group"] == "defect" and r["defect_type"] == dtype and r["split"] == "test"
            ]
            test_def_speakers = [records[i]["speaker"] for i in test_def_idxs]

            # Twin-calibrated negative training pool (210 raw anchors + 210 tight 0ms twins)
            train_neg_calibrated = train_anchor_idxs + train_tight_idxs
            X_tr_indices_cal = train_neg_calibrated + train_def_idxs
            y_tr_cal = np.array(
                [0] * len(train_neg_calibrated) + [1] * len(train_def_idxs), dtype=np.int32
            )

            # Naive negative training pool (210 raw anchors only) for clean-twin control audit
            X_tr_indices_naive = train_anchor_idxs + train_def_idxs
            y_tr_naive = np.array(
                [0] * len(train_anchor_idxs) + [1] * len(train_def_idxs), dtype=np.int32
            )

            # Speaker-disjoint inner validation split on the 43 train speakers to select best_layer
            train_speakers_sorted = sorted(
                {records[i]["speaker"] for i in train_anchor_idxs}
            )
            val_spk_set = set(train_speakers_sorted[::5])  # 9 validation speakers out of 43
            inner_tr_idxs = [
                i for i in X_tr_indices_cal if records[i]["speaker"] not in val_spk_set
            ]
            y_inner_tr = np.array(
                [0 if records[i]["group"] != "defect" else 1 for i in inner_tr_idxs],
                dtype=np.int32,
            )
            inner_val_raw = [
                i for i in train_anchor_idxs if records[i]["speaker"] in val_spk_set
            ]
            inner_val_tight = [
                i for i in train_tight_idxs if records[i]["speaker"] in val_spk_set
            ]
            inner_val_def = [
                i for i in train_def_idxs if records[i]["speaker"] in val_spk_set
            ]

            layer_curves_matched = []
            best_layer = 0
            best_val_score = -1.0

            for l_idx in range(num_layers):
                # 1) Inner-train -> inner-val for hyperparameter (best_layer) selection using primary_pmode
                X_itr = build_feature_matrix(feats, inner_tr_idxs, l_idx, primary_pmode, dtype)
                X_ival_raw = build_feature_matrix(feats, inner_val_raw, l_idx, primary_pmode, dtype)
                X_ival_tight = build_feature_matrix(feats, inner_val_tight, l_idx, primary_pmode, dtype)
                X_ival_def = build_feature_matrix(feats, inner_val_def, l_idx, primary_pmode, dtype)
                val_scores = fit_and_score_probe(
                    X_itr,
                    y_inner_tr,
                    {"raw": X_ival_raw, "tight": X_ival_tight, "def": X_ival_def},
                )
                val_auc_raw = compute_auroc(
                    val_scores["def"].tolist(), val_scores["raw"].tolist(), True
                )
                val_auc_tight = compute_auroc(
                    val_scores["def"].tolist(), val_scores["tight"].tolist(), True
                )
                val_composite = (
                    val_auc_tight
                    if dtype in ("clipped_word_end", "clipped_word_start")
                    else val_auc_raw
                )
                if val_composite > best_val_score:
                    best_val_score = val_composite
                    best_layer = l_idx

                # 2) Full train -> held-out test for layer trajectory curve
                X_tr = build_feature_matrix(feats, X_tr_indices_cal, l_idx, primary_pmode, dtype)
                X_te_raw = build_feature_matrix(feats, test_anchor_idxs, l_idx, primary_pmode, dtype)
                X_te_tight = build_feature_matrix(feats, test_tight_idxs, l_idx, primary_pmode, dtype)
                X_te_def = build_feature_matrix(feats, test_def_idxs, l_idx, primary_pmode, dtype)

                scores = fit_and_score_probe(
                    X_tr,
                    y_tr_cal,
                    {"raw": X_te_raw, "tight": X_te_tight, "def": X_te_def},
                )
                auc_raw = compute_auroc(scores["def"].tolist(), scores["raw"].tolist(), True)
                auc_tight = compute_auroc(scores["def"].tolist(), scores["tight"].tolist(), True)
                layer_curves_matched.append({
                    "layer": l_idx,
                    "val_auroc": round(val_composite, 4),
                    "auroc_vs_raw": round(auc_raw, 4),
                    "auroc_vs_tight_0ms": round(auc_tight, 4),
                })

            # Detailed evaluation at best_layer
            # 1) Pooling ablation at best_layer
            pooling_ablation = {}
            for pmode in ("mean", "mean_std", "edge", "combined"):
                X_tr_p = build_feature_matrix(feats, X_tr_indices_cal, best_layer, pmode, dtype)
                X_te_raw_p = build_feature_matrix(feats, test_anchor_idxs, best_layer, pmode, dtype)
                X_te_tight_p = build_feature_matrix(feats, test_tight_idxs, best_layer, pmode, dtype)
                X_te_def_p = build_feature_matrix(feats, test_def_idxs, best_layer, pmode, dtype)
                sc_p = fit_and_score_probe(
                    X_tr_p,
                    y_tr_cal,
                    {"raw": X_te_raw_p, "tight": X_te_tight_p, "def": X_te_def_p},
                )
                pooling_ablation[pmode] = {
                    "auroc_vs_raw": round(
                        compute_auroc(sc_p["def"].tolist(), sc_p["raw"].tolist(), True), 4
                    ),
                    "auroc_vs_tight_0ms": round(
                        compute_auroc(sc_p["def"].tolist(), sc_p["tight"].tolist(), True), 4
                    ),
                }

            # 2) Clean-twin Regime A (naive raw negatives) vs Regime B (twin-calibrated) at best_layer using primary_pmode
            X_tr_naive_mat = build_feature_matrix(feats, X_tr_indices_naive, best_layer, primary_pmode, dtype)
            X_tr_cal_mat = build_feature_matrix(feats, X_tr_indices_cal, best_layer, primary_pmode, dtype)
            X_te_raw_mat = build_feature_matrix(feats, test_anchor_idxs, best_layer, primary_pmode, dtype)
            X_te_tight_mat = build_feature_matrix(feats, test_tight_idxs, best_layer, primary_pmode, dtype)
            X_te_def_mat = build_feature_matrix(feats, test_def_idxs, best_layer, primary_pmode, dtype)
            eval_mats = {"raw": X_te_raw_mat, "tight": X_te_tight_mat, "def": X_te_def_mat}
            if dev_gold_idxs:
                eval_mats["dev_gold"] = build_feature_matrix(
                    feats, dev_gold_idxs, best_layer, primary_pmode, dtype
                )

            sc_naive = fit_and_score_probe(X_tr_naive_mat, y_tr_naive, eval_mats)
            sc_cal = fit_and_score_probe(X_tr_cal_mat, y_tr_cal, eval_mats)

            if "dev_gold" in sc_cal:
                dev_gold_probe_probs[enc_name][dtype] = sc_cal["dev_gold"]

            # Unseen-donor vs seen-donor check for secondary_speaker
            unseen_donor_auroc = None
            seen_donor_auroc = None
            if dtype == "secondary_speaker":
                unseen_mask = np.array([
                    raw_defects_meta.get(records[i]["clip_id"], {}).get("donor_speaker") in test_speaker_set
                    for i in test_def_idxs
                ], dtype=bool)
                seen_mask = np.array([
                    raw_defects_meta.get(records[i]["clip_id"], {}).get("donor_speaker") in train_speaker_set
                    for i in test_def_idxs
                ], dtype=bool)
                if unseen_mask.any():
                    unseen_donor_auroc = round(
                        compute_auroc(sc_cal["def"][unseen_mask].tolist(), sc_cal["raw"].tolist(), True), 4
                    )
                if seen_mask.any():
                    seen_donor_auroc = round(
                        compute_auroc(sc_cal["def"][seen_mask].tolist(), sc_cal["raw"].tolist(), True), 4
                    )

            # 3) Label-shuffle sanity control at best_layer
            rng_shuf = np.random.default_rng(42)
            y_shuf = rng_shuf.permutation(y_tr_cal)
            sc_shuf = fit_and_score_probe(
                X_tr_cal_mat, y_shuf, {"raw": X_te_raw_mat, "def": X_te_def_mat}
            )
            shuf_auroc = compute_auroc(sc_shuf["def"].tolist(), sc_shuf["raw"].tolist(), True)

            # 4) Bootstrap CIs vs deterministic signal baseline
            b_pos = baseline_test_scores[dtype]["pos"]
            b_neg_raw = baseline_test_scores[dtype]["neg_raw"]
            b_neg_tight = baseline_test_scores[dtype]["neg_tight"]

            ci_vs_raw = bootstrap_auroc_ci(
                sc_cal["def"],
                sc_cal["raw"],
                test_def_speakers,
                test_anchor_speakers,
                base_pos_scores=b_pos,
                base_neg_scores=b_neg_raw,
            )
            ci_vs_tight = bootstrap_auroc_ci(
                sc_cal["def"],
                sc_cal["tight"],
                test_def_speakers,
                test_tight_speakers,
                base_pos_scores=b_pos,
                base_neg_scores=b_neg_tight,
            )

            # 5) Per-severity breakdown at 5% and 10% FRR
            thresh_5pct_raw_naive = float(np.percentile(sc_naive["raw"], 95))
            naive_tight_frr = float(np.mean(sc_naive["tight"] >= thresh_5pct_raw_naive))

            thresh_5pct_raw_cal = float(np.percentile(sc_cal["raw"], 95))
            thresh_10pct_raw_cal = float(np.percentile(sc_cal["raw"], 90))
            cal_tight_frr = float(np.mean(sc_cal["tight"] >= thresh_5pct_raw_cal))
            thresh_5pct_tight_cal = float(np.percentile(sc_cal["tight"], 95))

            sev_results = {}
            for sev in SEVERITY_ORDER[dtype]:
                s_mask = np.array([records[i]["severity"] == sev for i in test_def_idxs], dtype=bool)
                s_sc = sc_cal["def"][s_mask]
                sev_results[sev] = {
                    "n": int(s_mask.sum()),
                    "auroc_vs_raw": round(
                        compute_auroc(s_sc.tolist(), sc_cal["raw"].tolist(), True), 4
                    ),
                    "auroc_vs_tight_0ms": round(
                        compute_auroc(s_sc.tolist(), sc_cal["tight"].tolist(), True), 4
                    ),
                    "det_at_5pct_raw_frr": round(float(np.mean(s_sc >= thresh_5pct_raw_cal)), 4),
                    "det_at_10pct_raw_frr": round(float(np.mean(s_sc >= thresh_10pct_raw_cal)), 4),
                    "det_at_5pct_tight_frr": round(float(np.mean(s_sc >= thresh_5pct_tight_cal)), 4),
                }

            enc_res["defects"][dtype] = {
                "primary_pooling": primary_pmode,
                "best_layer": best_layer,
                "best_auroc_vs_raw": round(
                    compute_auroc(sc_cal["def"].tolist(), sc_cal["raw"].tolist(), True), 4
                ),
                "best_auroc_vs_raw_ci": [ci_vs_raw["ci_low"], ci_vs_raw["ci_high"]],
                "delta_auroc_vs_baseline_raw": ci_vs_raw.get("diff_mean"),
                "delta_auroc_vs_baseline_raw_ci": [
                    ci_vs_raw.get("diff_ci_low"),
                    ci_vs_raw.get("diff_ci_high"),
                ],
                "best_auroc_vs_tight_0ms": round(
                    compute_auroc(sc_cal["def"].tolist(), sc_cal["tight"].tolist(), True), 4
                ),
                "best_auroc_vs_tight_0ms_ci": [ci_vs_tight["ci_low"], ci_vs_tight["ci_high"]],
                "delta_auroc_vs_baseline_tight": ci_vs_tight.get("diff_mean"),
                "delta_auroc_vs_baseline_tight_ci": [
                    ci_vs_tight.get("diff_ci_low"),
                    ci_vs_tight.get("diff_ci_high"),
                ],
                "unseen_test_donor_auroc": unseen_donor_auroc,
                "seen_train_donor_auroc": seen_donor_auroc,
                "naive_probe_auroc_vs_raw": round(
                    compute_auroc(sc_naive["def"].tolist(), sc_naive["raw"].tolist(), True), 4
                ),
                "naive_probe_auroc_vs_tight_0ms": round(
                    compute_auroc(sc_naive["def"].tolist(), sc_naive["tight"].tolist(), True), 4
                ),
                "naive_probe_tight_twin_frr_at_5pct_raw": round(naive_tight_frr, 4),
                "calibrated_probe_tight_twin_frr_at_5pct_raw": round(cal_tight_frr, 4),
                "label_shuffle_auroc": round(shuf_auroc, 4),
                "layer_curve": layer_curves_matched,
                "pooling_ablation": pooling_ablation,
                "severities": sev_results,
            }

        all_encoder_results[enc_name] = enc_res

    # 3. Diagnostic zero-shot evaluation on 288-clip dev gold set (gold_benchmark_20260908)
    dev_gold_eval: dict[str, Any] = {}
    if dev_gold_idxs:
        gold_decisions = [
            1 if records[i].get("decision") == "reject" else 0 for i in dev_gold_idxs
        ]
        gold_durs = np.array([records[i]["duration_s"] for i in dev_gold_idxs], dtype=np.float64)
        total_min = float(np.sum(gold_durs) / 60.0)
        base_r = float(np.mean(gold_decisions))

        # Dimension labels
        dim_labels = {
            "clipped_word_end": [
                1 if records[i].get("dimensions", {}).get("word_end") == "defective" else 0
                for i in dev_gold_idxs
            ],
            "clipped_word_start": [
                1 if records[i].get("dimensions", {}).get("word_start") == "defective" else 0
                for i in dev_gold_idxs
            ],
            "secondary_speaker": [
                1
                if (
                    records[i].get("dimensions", {}).get("speaker_purity") == "defective"
                    or records[i].get("dimensions", {}).get("overlap") == "defective"
                )
                else 0
                for i in dev_gold_idxs
            ],
            "background_music": [
                1 if records[i].get("dimensions", {}).get("music") == "defective" else 0
                for i in dev_gold_idxs
            ],
        }

        dev_gold_eval["summary"] = {
            "n_clips": len(dev_gold_idxs),
            "n_reject": int(sum(gold_decisions)),
            "total_minutes": round(total_min, 2),
            "recipe_nolock_R": round(base_r, 4),
            "defect_counts": {k: int(sum(v)) for k, v in dim_labels.items()},
        }
        dev_gold_eval["models"] = {}

        for enc_name, probs_by_dtype in dev_gold_probe_probs.items():
            dim_aurocs = {}
            for dtype, y_dim in dim_labels.items():
                pos_s = [
                    float(probs_by_dtype[dtype][j])
                    for j in range(len(dev_gold_idxs))
                    if y_dim[j] == 1
                ]
                neg_s = [
                    float(probs_by_dtype[dtype][j])
                    for j in range(len(dev_gold_idxs))
                    if y_dim[j] == 0
                ]
                dim_aurocs[dtype] = round(compute_auroc(pos_s, neg_s, True), 4)

            # Max defect probability across the 4 probes as gatekeeper risk score
            max_prob = np.max(
                np.stack([probs_by_dtype[dt] for dt in DEFECT_TYPES], axis=1), axis=1
            )
            pos_all = [
                float(max_prob[j]) for j in range(len(dev_gold_idxs)) if gold_decisions[j] == 1
            ]
            neg_all = [
                float(max_prob[j]) for j in range(len(dev_gold_idxs)) if gold_decisions[j] == 0
            ]
            overall_auc = compute_auroc(pos_all, neg_all, True)

            # R at 80%, 70%, 60% accepted-minute yield
            r_at_yield = {}
            for target_yield in (0.80, 0.70, 0.60):
                q = np.quantile(max_prob, target_yield)
                keep = max_prob <= q
                if np.sum(keep) > 0:
                    bad_acc = float(np.mean(np.array(gold_decisions)[keep]))
                    acc_min = float(np.sum(gold_durs[keep]) / 60.0)
                    r_at_yield[f"yield_{int(target_yield*100)}pct"] = {
                        "R": round(bad_acc, 4),
                        "accepted_minutes": round(acc_min, 2),
                        "yield_ratio": round(acc_min / total_min, 4),
                    }

            dev_gold_eval["models"][enc_name] = {
                "overall_reject_auroc": round(overall_auc, 4),
                "dimension_aurocs": dim_aurocs,
                "r_vs_yield": r_at_yield,
            }

    full_report = {
        "split_counts": {
            "train_anchors": len(train_anchor_idxs),
            "test_anchors": len(test_anchor_idxs),
            "train_tight_twins_0ms": len(train_tight_idxs),
            "test_tight_twins_0ms": len(test_tight_idxs),
            "train_defects_per_type": 840,
            "test_defects_per_type": 360,
            "dev_gold_clips": len(dev_gold_idxs),
        },
        "trivial_baselines_test": trivial_baselines,
        "signal_baselines_test": baseline_test_results,
        "encoders": all_encoder_results,
        "dev_gold_diagnostic": dev_gold_eval,
    }

    report_json = out_dir / "phase3_probe_report.json"
    report_md = out_dir / "PHASE3_PROBE_REPORT.md"
    write_json(report_json, full_report)

    # Build Markdown report
    lines = [
        "# Phase 3: Layer-Wise Acoustic Representation Probing on ViCut-Bench",
        "",
        f"- **Data split:** ViCut-Bench `train` ({len(train_anchor_idxs)} clean anchors + {len(train_tight_idxs)} tight-cut 0ms clean twins + 3,360 defects across 43 speakers) → `test` ({len(test_anchor_idxs)} clean anchors + {len(test_tight_idxs)} tight-cut 0ms clean twins + 1,440 defects across 19 disjoint speakers).",
        "- **Primary pooling (prevents opposite-boundary silence shortcuts):** `edge` (`end_edge` only for `clipped_word_end`, `start_edge` only for `clipped_word_start`) and `mean_std` (`[mean, std]` for `secondary_speaker` and `background_music`).",
        f"- **Full JSON artifact:** `{persist_path(report_json)}`",
        "",
        "## 1. Headline Test-Split AUROC: Frozen Encoder Probes vs. Deterministic Signal Baselines",
        "",
        "| Defect Type | Signal Baseline (vs Raw / vs Tight 0ms) | Encoder (Pooling, Best Layer) | Probe AUROC vs Raw [95% CI] | ΔAUROC vs Baseline Raw [95% CI] | Probe AUROC vs Tight 0ms [95% CI] | ΔAUROC vs Baseline Tight [95% CI] | Label-Shuffle Control |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for dtype in DEFECT_TYPES:
        b_res = baseline_test_results[dtype]
        b_str = f"`{b_res['feature']}`: **{b_res['test_auroc_vs_raw']:.4f}** / **{b_res['test_auroc_vs_tight_0ms']:.4f}**"
        for enc_name, enc_res in all_encoder_results.items():
            d_res = enc_res["defects"][dtype]
            ci_r = d_res["best_auroc_vs_raw_ci"]
            d_ci = d_res["delta_auroc_vs_baseline_raw_ci"]
            ci_t = d_res["best_auroc_vs_tight_0ms_ci"]
            dt_ci = d_res["delta_auroc_vs_baseline_tight_ci"]
            lines.append(
                f"| `{dtype}` | {b_str} | `{enc_name}` (`{d_res['primary_pooling']}`, L{d_res['best_layer']}) | "
                f"**{d_res['best_auroc_vs_raw']:.4f}** [{ci_r[0]:.4f}, {ci_r[1]:.4f}] | "
                f"{d_res['delta_auroc_vs_baseline_raw']:+.4f} [{d_ci[0]:+.4f}, {d_ci[1]:+.4f}] | "
                f"**{d_res['best_auroc_vs_tight_0ms']:.4f}** [{ci_t[0]:.4f}, {ci_t[1]:.4f}] | "
                f"{d_res['delta_auroc_vs_baseline_tight']:+.4f} [{dt_ci[0]:+.4f}, {dt_ci[1]:+.4f}] | "
                f"{d_res['label_shuffle_auroc']:.4f} |"
            )

    lines.extend([
        "",
        "## 2. Alternative-Explanation & Sanity Controls (GOAL.md §2 Rule 4 & §3)",
        "",
        "### 2.1 Unseen-Donor Generalization on `secondary_speaker` (96 Unseen-Test-Donor vs. 264 Seen-Train-Donor Test Clips)",
        "",
        "| Encoder | Best Layer | Overall Test AUROC | Unseen-Test-Donor AUROC ($n=96$) | Seen-Train-Donor AUROC ($n=264$) | Trivial Duration Baseline AUROC |",
        "|---|---|---|---|---|---|",
    ])
    for enc_name, enc_res in all_encoder_results.items():
        d_res = enc_res["defects"]["secondary_speaker"]
        lines.append(
            f"| `{enc_name}` | L{d_res['best_layer']} | **{d_res['best_auroc_vs_raw']:.4f}** | "
            f"**{d_res['unseen_test_donor_auroc']:.4f}** | {d_res['seen_train_donor_auroc']:.4f} | "
            f"{trivial_baselines['secondary_speaker']['duration_auroc']:.4f} |"
        )

    lines.extend([
        "",
        "### 2.2 Clean-Twin Boundary Control Audit (`0ms` Tight-Cut Twins at `[speech_onset_s, speech_offset_s]`)",
        "",
        "Because `synthesize_defects.py` slices `clipped_word_end` at `speech_offset - delta_t` (removing ~300 ms of trailing room silence while keeping ~500 ms of leading silence) and `clipped_word_start` at `speech_onset + delta_t` (removing ~500 ms of leading room silence while keeping ~300 ms of trailing silence):",
        "1. A **naive** detector trained only against raw un-sliced VIVOS anchors learns to flag any clip without trailing/leading silence, causing high false-reject rates on clean `0ms` tight-cut twins.",
        "2. Using **`combined` (`[mean, std, start_edge, end_edge]`)** against two-sided `0ms` tight twins creates a synthetic XOR shortcut (`start_edge` has silence while `end_edge` does not). Using **strict single-edge pooling (`end_edge` only for end clipping, `start_edge` only for start clipping)** eliminates the opposite-boundary shortcut and measures true physical boundary discrimination:",
        "",
        "| Defect Type | Detector / Encoder | Naive Training FRR on 0ms Tight Twins | Naive AUROC vs 0ms Tight Twins | Twin-Calibrated FRR on 0ms Tight Twins | Twin-Calibrated AUROC vs 0ms Tight Twins |",
        "|---|---|---|---|---|---|",
    ])

    for dtype in ("clipped_word_end", "clipped_word_start"):
        b_res = baseline_test_results[dtype]
        lines.append(
            f"| `{dtype}` | Signal `{b_res['feature']}` | **{b_res['tight_twin_frr_at_5pct_raw_thresh']*100:.1f}%** | {b_res['test_auroc_vs_tight_0ms']:.4f} | 5.0% (by threshold calibration) | {b_res['test_auroc_vs_tight_0ms']:.4f} |"
        )
        for enc_name, enc_res in all_encoder_results.items():
            d_res = enc_res["defects"][dtype]
            lines.append(
                f"| `{dtype}` | `{enc_name}` (`{d_res['primary_pooling']}`, L{d_res['best_layer']}) | "
                f"**{d_res['naive_probe_tight_twin_frr_at_5pct_raw']*100:.1f}%** | "
                f"{d_res['naive_probe_auroc_vs_tight_0ms']:.4f} | "
                f"**{d_res['calibrated_probe_tight_twin_frr_at_5pct_raw']*100:.1f}%** | "
                f"**{d_res['best_auroc_vs_tight_0ms']:.4f}** |"
            )

    lines.extend([
        "",
        "## 3. Pooling Component Ablation (Criterion G3)",
        "",
        "| Encoder | Defect Type | Best Layer | `mean` Only (vs Raw / vs Tight) | `mean+std` (vs Raw / vs Tight) | `edge` Only (vs Raw / vs Tight) | `combined` (vs Raw / vs Tight) |",
        "|---|---|---|---|---|---|---|",
    ])

    for enc_name, enc_res in all_encoder_results.items():
        for dtype in DEFECT_TYPES:
            d_res = enc_res["defects"][dtype]
            pa = d_res["pooling_ablation"]
            lines.append(
                f"| `{enc_name}` | `{dtype}` | L{d_res['best_layer']} | "
                f"{pa['mean']['auroc_vs_raw']:.4f} / {pa['mean']['auroc_vs_tight_0ms']:.4f} | "
                f"{pa['mean_std']['auroc_vs_raw']:.4f} / {pa['mean_std']['auroc_vs_tight_0ms']:.4f} | "
                f"{pa['edge']['auroc_vs_raw']:.4f} / {pa['edge']['auroc_vs_tight_0ms']:.4f} | "
                f"{pa['combined']['auroc_vs_raw']:.4f} / {pa['combined']['auroc_vs_tight_0ms']:.4f} |"
            )

    lines.extend([
        "",
        "## 4. Per-Layer Test AUROC Trajectory (Primary Defect-Matched Pooling, Twin-Calibrated)",
        "",
        "| Encoder | Defect Type | L0 | L2 | L4 | L6 | L8 | L10 | L12 |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for enc_name, enc_res in all_encoder_results.items():
        for dtype in DEFECT_TYPES:
            lc = {item["layer"]: item for item in enc_res["defects"][dtype]["layer_curve"]}
            cols = [
                f"{lc[l]['auroc_vs_raw']:.3f} ({lc[l]['auroc_vs_tight_0ms']:.3f})"
                for l in (0, 2, 4, 6, 8, 10, 12)
            ]
            lines.append(f"| `{enc_name}` | `{dtype}` | " + " | ".join(cols) + " |")

    lines.extend([
        "",
        "## 5. Detection Rate by Severity at 5% Clean False-Reject Rate (Held-Out Test Split)",
        "",
        "| Defect Type | Severity | Signal Baseline (Det @ 5% Raw / Tight FRR) | "
        + " | ".join(f"`{e}` (Det @ 5% Raw / Tight FRR)" for e in all_encoder_results)
        + " |",
        "|---|---|---|" + "|".join(["---"] * len(all_encoder_results)) + "|",
    ])

    for dtype in DEFECT_TYPES:
        for sev in SEVERITY_ORDER[dtype]:
            bs_s = baseline_test_results[dtype]["severities"][sev]
            b_cell = f"{bs_s['det_at_5pct_raw_frr']*100:.1f}% / {bs_s['det_at_5pct_tight_frr']*100:.1f}%"
            e_cells = []
            for enc_name, enc_res in all_encoder_results.items():
                es = enc_res["defects"][dtype]["severities"][sev]
                e_cells.append(
                    f"**{es['det_at_5pct_raw_frr']*100:.1f}%** / **{es['det_at_5pct_tight_frr']*100:.1f}%**"
                )
            lines.append(f"| `{dtype}` | `{sev}` | {b_cell} | " + " | ".join(e_cells) + " |")

    if dev_gold_eval:
        s_info = dev_gold_eval["summary"]
        lines.extend([
            "",
            "## 6. Zero-Shot Diagnostic Transfer to 288-Clip Real Dev Set (`gold_benchmark_20260908`)",
            "",
            f"- **Note (GOAL.md §2 Rule 2):** `gold_benchmark_20260908` is Gemini-labelled dev data ({s_info['n_clips']} clips, {s_info['total_minutes']} min, un-gated $R = {s_info['recipe_nolock_R']:.4f}$). Shown strictly as an out-of-domain diagnostic check of synthetic-to-real transfer.",
            "",
            "| Encoder Gatekeeper | Overall Reject AUROC | `clipped_word_end` AUROC | `clipped_word_start` AUROC | `secondary_speaker` AUROC | `background_music` AUROC | $R$ @ 80% Yield | $R$ @ 70% Yield |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for enc_name, m_ev in dev_gold_eval["models"].items():
            da = m_ev["dimension_aurocs"]
            r80 = m_ev["r_vs_yield"].get("yield_80pct", {}).get("R", float("nan"))
            r70 = m_ev["r_vs_yield"].get("yield_70pct", {}).get("R", float("nan"))
            lines.append(
                f"| `{enc_name}` | **{m_ev['overall_reject_auroc']:.4f}** | "
                f"{da['clipped_word_end']:.4f} | {da['clipped_word_start']:.4f} | "
                f"{da['secondary_speaker']:.4f} | {da['background_music']:.4f} | "
                f"{r80:.4f} | {r70:.4f} |"
            )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote Phase 3 probe report to %s and %s", report_json, report_md)
    print(persist_path(report_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
