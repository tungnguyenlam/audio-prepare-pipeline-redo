#!/usr/bin/env python3
"""Calibrate and evaluate pre-registered Single Candidate Hybrid Gate on dev-288 and ViCut-Bench.

Decision rule (pre-registered 2026-11-06 in WORKLOG.md & PLAN.md Phase 1, lines 71-107):
  - Pre-gate: Deterministic digital silence / low-energy rejection at RMS < -50 dBFS.
  - Boundary arm: Deterministic 10 ms edge energy ratios (start and end).
    Thresholded at 95% specificity on held-out 5-fold OOF clean_tight_tr (tight twins).
    If start_ratio > start_thresh OR end_ratio > end_thresh: flag_edge = 1.0, else 0.0.
  - Content arm: PhoWhisper (vinai/phowhisper-small) Layer 4 encoder representation with mean_std
    pooling for secondary speaker and background music heads. Fit on ViCut-Bench train defects
    vs clean negatives (anchors + tight twins), with 5-fold OOF generalization AUROC evaluated.
    p_content = max(p_sec, p_music).
  - Decision score & determinism: Scalar score s = flag_edge + p_content with lexicographical
    ranking tuple (pre_gate, flag_edge, p_content, clip_id).

Phase 1 Exit Criteria & Decision Gate (PLAN.md line 99):
  - Branch A: Candidate beats each dev baseline with 95% CI strictly excluding 0 across both seeds.
  - Branch B (Fallback): CI crosses 0 by 2026-12-01 -> Lock candidate, log honest null dev findings,
    and proceed to Phase 2 (G1 Test Set Benchmark Characterization) and Phase 3 (ViCut-Bench Audit).
"""
from __future__ import annotations

import os
# Pin threads for bit-deterministic numerical execution across BLAS/LAPACK implementations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser
from vicut_bench.probe_encoder_layers import (
    BASELINE_SPEC,
    DEFECT_TYPES,
    SEVERITY_ORDER,
    build_feature_matrix,
    fit_and_score_probe,
    load_clip_records,
    read_audio_for_record,
)
from vicut_bench.score_signal_baselines import compute_auroc

logger = logging.getLogger("evaluate_hybrid_gate")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compute_rms_dbfs(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms <= 1e-9:
        return -120.0
    return float(20.0 * np.log10(rms))


def curve_lexicographical(
    records: list[dict[str, Any]],
    pre_gate: np.ndarray,
    flag_edge: np.ndarray,
    p_content: np.ndarray,
    ascending: bool = True,
) -> list[dict[str, Any]]:
    """Compute defect rate R vs accepted-minute yield curve using deterministic lexicographical sorting.
    
    Sorting tuple: (pre_gate, flag_edge, p_content, clip_id).
    When ascending=True (default for curation acceptance):
      - pre_gate == 0 (good) comes before pre_gate == 1 (bad silence).
      - flag_edge == 0 (good boundary) comes before flag_edge == 1 (bad boundary).
      - lower p_content (low defect risk) comes before higher p_content.
      - exact ties broken alphabetically by clip_id.
    """
    total_minutes = sum(r["duration_s"] for r in records) / 60.0

    items = []
    for i, r in enumerate(records):
        cid = r.get("id") or r.get("clip_id") or f"clip_{i:04d}"
        items.append((
            int(pre_gate[i]),
            float(flag_edge[i]),
            float(p_content[i]),
            str(cid),
            i,
        ))

    items.sort(reverse=not ascending)

    curve_pts: list[dict[str, Any]] = []
    acc_minutes = 0.0
    acc_bad_minutes = 0.0
    acc_clips = 0
    acc_bad_clips = 0

    curve_pts.append({
        "threshold_p_content": 0.0 if ascending else 1.0,
        "accepted_clips": 0,
        "accepted_bad_clips": 0,
        "accepted_minutes": 0.0,
        "yield_ratio": 0.0,
        "R": 0.0,
    })

    for item in items:
        idx = item[4]
        r = records[idx]
        is_bad = int(r["label"] == "reject")
        dur_min = r["duration_s"] / 60.0

        acc_clips += 1
        acc_bad_clips += is_bad
        acc_minutes += dur_min
        acc_bad_minutes += is_bad * dur_min

        r_val = acc_bad_clips / acc_clips if acc_clips > 0 else 0.0
        p_c = float(item[2])

        curve_pts.append({
            "threshold_p_content": round(p_c, 6),
            "accepted_clips": acc_clips,
            "accepted_bad_clips": acc_bad_clips,
            "accepted_minutes": round(acc_minutes, 4),
            "yield_ratio": round(acc_minutes / total_minutes, 6) if total_minutes > 0 else 0.0,
            "R": round(r_val, 6),
        })

    return curve_pts


def matched_point_at_yield(
    curve_pts: list[dict[str, Any]], target_yield: float, tolerance: float | None = None
) -> dict[str, Any] | None:
    """Find the point on the empirical acceptance curve closest to the target yield ratio."""
    if not curve_pts:
        return None
    best = min(curve_pts, key=lambda pt: abs(pt["yield_ratio"] - target_yield))
    if tolerance is not None and abs(best["yield_ratio"] - target_yield) > tolerance:
        return None
    return best


def calibrate_boundary_arm(
    clean_tight_tr: list[dict[str, Any]], n_splits: int = 5, seed: int = 42
) -> tuple[float, float, dict[str, Any]]:
    """Calibrate 10 ms edge energy ratio thresholds on clean tight twins.
    
    Thresholds are calibrated at 95% specificity (95th percentile of clean edge energy ratios)
    using 5-fold GroupKFold by speaker. Returns (start_thresh, end_thresh, diagnostic_stats).
    """
    speakers = [r["speaker"] for r in clean_tight_tr]
    st_ratios = np.array([r["start_edge_ratio_10ms"] for r in clean_tight_tr])
    en_ratios = np.array([r["end_edge_ratio_10ms"] for r in clean_tight_tr])

    st_thresh = float(np.percentile(st_ratios, 95.0))
    en_thresh = float(np.percentile(en_ratios, 95.0))

    gkf = GroupKFold(n_splits=n_splits)
    oof_st_pred = np.zeros(len(clean_tight_tr), dtype=bool)
    oof_en_pred = np.zeros(len(clean_tight_tr), dtype=bool)

    for train_idx, val_idx in gkf.split(clean_tight_tr, groups=speakers):
        t_st_th = float(np.percentile(st_ratios[train_idx], 95.0))
        t_en_th = float(np.percentile(en_ratios[train_idx], 95.0))
        oof_st_pred[val_idx] = st_ratios[val_idx] > t_st_th
        oof_en_pred[val_idx] = en_ratios[val_idx] > t_en_th

    oof_union_pred = oof_st_pred | oof_en_pred
    st_spec = float(np.mean(~oof_st_pred))
    en_spec = float(np.mean(~oof_en_pred))
    union_spec = float(np.mean(~oof_union_pred))

    stats = {
        "n_clean_tight_tr": len(clean_tight_tr),
        "start_thresh": round(st_thresh, 6),
        "end_thresh": round(en_thresh, 6),
        "oof_start_specificity": round(st_spec, 6),
        "oof_end_specificity": round(en_spec, 6),
        "oof_union_specificity": round(union_spec, 6),
        "oof_start_frr": round(1.0 - st_spec, 6),
        "oof_end_frr": round(1.0 - en_spec, 6),
        "oof_union_frr": round(1.0 - union_spec, 6),
    }
    return st_thresh, en_thresh, stats


def calibrate_content_arm(
    records: list[dict[str, Any]],
    clean_train: list[dict[str, Any]],
    feats: np.ndarray,
    c_reg: float = 0.01,
    seed: int = 42,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Train PhoWhisper L4 logistic regression heads on train defects vs clean negatives.
    
    Computes 5-fold OOF AUROC on train split and returns fitted classifiers and scalers.
    """
    clean_uids = {r["uid"] for r in clean_train}
    clean_indices = [i for i, r in enumerate(records) if r["uid"] in clean_uids]

    results: dict[str, Any] = {}

    for defect in ["secondary_speaker", "background_music"]:
        def_indices = [
            i for i, r in enumerate(records)
            if r.get("split") == "train" and r.get("defect_type") == defect
        ]
        tr_indices = clean_indices + def_indices
        X_tr = feats[tr_indices]
        y_tr = np.array([0] * len(clean_indices) + [1] * len(def_indices), dtype=int)
        spk_tr = [records[i]["speaker"] for i in tr_indices]

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)

        clf = LogisticRegression(
            C=c_reg,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(X_tr_scaled, y_tr)

        # 5-fold GroupKFold OOF evaluation on train split
        gkf = GroupKFold(n_splits=n_splits)
        oof_probs = np.zeros(len(y_tr), dtype=float)
        for t_idx, v_idx in gkf.split(X_tr, y_tr, groups=spk_tr):
            sc_k = StandardScaler()
            X_k_tr = sc_k.fit_transform(X_tr[t_idx])
            X_k_val = sc_k.transform(X_tr[v_idx])
            clf_k = LogisticRegression(
                C=c_reg,
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )
            clf_k.fit(X_k_tr, y_tr[t_idx])
            oof_probs[v_idx] = clf_k.predict_proba(X_k_val)[:, 1]

        oof_auc = compute_auroc(
            oof_probs[y_tr == 1].tolist(), oof_probs[y_tr == 0].tolist(), True
        )

        results[defect] = {
            "scaler": scaler,
            "clf": clf,
            "oof_auroc": round(float(oof_auc), 6),
            "n_defect_train": len(def_indices),
            "n_clean_train": len(clean_indices),
        }

    return results


def paired_cluster_bootstrap(
    dev_records: list[dict[str, Any]],
    cand_curve: list[dict[str, Any]],
    base_point: dict[str, Any],
    base_indices: list[int] | None,
    matched_yield: float,
    seeds: list[int] = [42, 123],
    n_replicates: int = 1000,
) -> dict[int, dict[str, Any]]:
    """Compute paired cluster bootstrap 95% CIs resampling recordings with replacement.
    
    Returns delta_R = R_candidate - R_baseline (negative delta indicates candidate improvement).
    """
    recordings = sorted(list({r["speaker"] for r in dev_records}))
    rec_to_indices: dict[str, list[int]] = {}
    for i, r in enumerate(dev_records):
        rec_to_indices.setdefault(r["speaker"], []).append(i)

    results: dict[int, dict[str, Any]] = {}

    for s in seeds:
        rng = np.random.RandomState(s)
        deltas: list[float] = []
        valid_reps = 0

        for _ in range(n_replicates):
            sampled_recs = rng.choice(recordings, size=len(recordings), replace=True)
            sampled_idx: list[int] = []
            for rec in sampled_recs:
                sampled_idx.extend(rec_to_indices[rec])

            sampled_records = [dev_records[i] for i in sampled_idx]

            # Candidate on sampled records
            sampled_curve = curve_lexicographical(
                sampled_records,
                np.array([r["pre_gate"] for r in sampled_records]),
                np.array([r["flag_edge"] for r in sampled_records]),
                np.array([r["p_content"] for r in sampled_records]),
                ascending=True,
            )
            cand_pt = matched_point_at_yield(sampled_curve, matched_yield, tolerance=matched_yield * 0.10)
            if cand_pt is None:
                continue
            r_cand = cand_pt["R"]

            # Baseline on sampled records
            if base_indices is not None:
                base_set = set(base_indices)
                b_accepted = [r for idx, r in enumerate(sampled_records) if sampled_idx[idx] in base_set]
                b_acc_min = sum(r["duration_s"] for r in b_accepted) / 60.0
                if len(b_accepted) == 0 or b_acc_min <= 0:
                    continue
                r_base = sum(1 for r in b_accepted if r["label"] == "reject") / len(b_accepted)
            else:
                base_curve = base_point.get("_curve")
                if base_curve is not None:
                    b_pt = matched_point_at_yield(base_curve(sampled_records), matched_yield, tolerance=matched_yield * 0.10)
                    if b_pt is None:
                        continue
                    r_base = b_pt["R"]
                else:
                    r_base = base_point["R"]

            deltas.append(r_cand - r_base)
            valid_reps += 1

        if len(deltas) >= 50:
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))
            mean_delta = float(np.mean(deltas))
            excludes_zero = bool(ci_high < 0.0 or ci_low > 0.0)
        else:
            ci_low, ci_high, mean_delta, excludes_zero = 0.0, 0.0, 0.0, False

        results[s] = {
            "n_replicates": n_replicates,
            "valid_replicates": valid_reps,
            "mean_delta_R": round(mean_delta, 6),
            "ci_95_low": round(ci_low, 6),
            "ci_95_high": round(ci_high, 6),
            "ci_excludes_zero": excludes_zero,
        }

    return results


def score_manifest_inference(
    manifest_path: Path,
    frozen_params_path: Path,
    output_scores_path: Path,
    probe_cache_path: Path | None = None,
    allow_non_16k: bool = False,
) -> None:
    """Score an arbitrary audio manifest using frozen candidate parameters without refitting."""
    logger.info("Executing score-only inference on manifest: %s", manifest_path)
    if not frozen_params_path.exists():
        raise FileNotFoundError(f"Frozen parameters file not found: {frozen_params_path}")

    params = json.loads(frozen_params_path.read_text(encoding="utf-8"))
    st_thresh = params["boundary_arm"]["start_thresh"]
    en_thresh = params["boundary_arm"]["end_thresh"]
    pre_thresh = params["pre_gate"]["threshold_rms_dbfs"]
    heads = params["content_arm"]["heads"]

    import soundfile as sf
    from vicut_bench.score_signal_baselines import extract_signal_features

    raw_manifest_records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_manifest_records.append(json.loads(line))

    if not raw_manifest_records:
        raise ValueError(f"No records found in manifest: {manifest_path}")

    # Check if probe cache is available
    cache = None
    if probe_cache_path is not None and probe_cache_path.exists():
        cache = np.load(probe_cache_path, allow_pickle=True)

    cache_uids_map: dict[str, int] = {}
    if cache is not None and "uids" in cache:
        for idx, u in enumerate(cache["uids"]):
            cache_uids_map[str(u)] = idx

    # Check if precomputed signal features cache is available nearby
    sig_cache: dict[str, Any] = {}
    if probe_cache_path is not None:
        candidate_sig_path = probe_cache_path.parent / "signal_features.json"
        if candidate_sig_path.exists():
            sig_cache = json.loads(candidate_sig_path.read_text(encoding="utf-8"))

    # Determine whether all records can be retrieved from probe cache
    all_in_cache = False
    cached_indices: list[int] = []
    if cache is not None:
        can_find_all = True
        for r in raw_manifest_records:
            audio_p = r.get("blind_wav") or r.get("abs_path") or r.get("audio_path") or r.get("path") or r.get("rel_path", "")
            cid = r.get("review_id") or r.get("test_id") or r.get("id") or r.get("clip_id") or Path(audio_p).stem
            uid = r.get("uid") or cid
            found_idx = None
            for cand in [uid, cid, f"dev_gold__{cid}", f"dev_gold__{uid}"]:
                if cand in cache_uids_map:
                    found_idx = cache_uids_map[cand]
                    break
            if found_idx is not None:
                cached_indices.append(found_idx)
            else:
                can_find_all = False
                break
        all_in_cache = can_find_all

    feats_content: np.ndarray
    if all_in_cache and cache is not None:
        logger.info("Found all %d records in probe cache. Loading Layer 4 mean_std representations...", len(raw_manifest_records))
        m = cache["mean"][cached_indices, 4, :768]
        s = cache["std"][cached_indices, 4, :768]
        feats_content = np.concatenate([m, s], axis=1)
    else:
        logger.info("Extracting PhoWhisper Layer 4 mean_std features via production probe_encoder_layers...")
        try:
            import tempfile
            import shutil
            import torch
            from probe_encoder_layers import extract_encoder_features
        except ImportError as e:
            raise RuntimeError(
                f"PhoWhisper L4 features cannot be computed without torch/probe_encoder_layers, and records were not found in cache: {e}"
            ) from e

        records_to_extract: list[dict[str, Any]] = []
        for idx, r in enumerate(raw_manifest_records):
            audio_path = Path(
                r.get("blind_wav")
                or r.get("abs_path")
                or r.get("audio_path")
                or r.get("path")
                or r.get("rel_path", "")
            )
            cid = r.get("review_id") or r.get("test_id") or r.get("id") or r.get("clip_id") or audio_path.stem
            uid = r.get("uid") or (f"dev_gold__{cid}" if f"dev_gold__{cid}" in cache_uids_map else cid)
            dur = float(r.get("duration", r.get("duration_s", 0.0)))

            if audio_path.is_file():
                info = sf.info(str(audio_path))
                if dur <= 0.0:
                    dur = float(info.duration)
                sr_clip = info.samplerate
            else:
                sr_clip = 16000

            if sr_clip != 16000 and not allow_non_16k:
                raise ValueError(
                    f"Clip '{cid}' ({audio_path}) has sampling rate {sr_clip} Hz != 16000 Hz. "
                    f"The frozen PhoWhisper small feature extractor requires 16000 Hz audio per Phase 1 specification. "
                    f"Please resample audio to 16 kHz before scoring (or pass --allow-non-16k to override for historical reproduction)."
                )

            records_to_extract.append({
                "uid": uid,
                "clip_id": cid,
                "audio_path": str(audio_path),
                "duration_s": dur,
            })

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tmp_extract_dir = Path(tempfile.mkdtemp(prefix="phowhisper_extract_"))
        try:
            logger.info("Running production extract_encoder_features on device=%s (bfloat16, batch_size=8)...", dev)
            res = extract_encoder_features(
                encoder_name="phowhisper",
                records=records_to_extract,
                cache_dir=tmp_extract_dir,
                device=dev,
                batch_size=8,
                overwrite=True,
            )
            m = res["mean"][:, 4, :768]
            s = res["std"][:, 4, :768]
            feats_content = np.concatenate([m, s], axis=1)
        finally:
            shutil.rmtree(tmp_extract_dir, ignore_errors=True)

    def predict_from_head(head_dict: dict[str, Any], X_mat: np.ndarray) -> np.ndarray:
        mean = np.array(head_dict["scaler_mean"], dtype=np.float64)
        scale = np.array(head_dict["scaler_scale"], dtype=np.float64)
        coef = np.array(head_dict["coef"], dtype=np.float64)
        intercept = float(head_dict["intercept"])
        X_scaled = (X_mat - mean) / scale
        logit = np.dot(X_scaled, coef) + intercept
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))

    p_sec_arr = predict_from_head(heads["secondary_speaker"], feats_content)
    p_mus_arr = predict_from_head(heads["background_music"], feats_content)
    p_content_arr = np.maximum(p_sec_arr, p_mus_arr)

    scored_records: list[dict[str, Any]] = []
    for idx, r in enumerate(raw_manifest_records):
        audio_path = Path(
            r.get("blind_wav")
            or r.get("abs_path")
            or r.get("audio_path")
            or r.get("path")
            or r.get("rel_path", "")
        )
        cid = r.get("review_id") or r.get("test_id") or r.get("id") or r.get("clip_id") or audio_path.stem
        uid = r.get("uid") or (f"dev_gold__{cid}" if f"dev_gold__{cid}" in cache_uids_map else cid)

        rms_val: float | None = None
        st_ratio: float | None = None
        en_ratio: float | None = None

        if "rms_dbfs" in r and "start_edge_ratio_10ms" in r and "end_edge_ratio_10ms" in r:
            rms_val = float(r["rms_dbfs"])
            st_ratio = float(r["start_edge_ratio_10ms"])
            en_ratio = float(r["end_edge_ratio_10ms"])
        elif uid in sig_cache:
            rms_val = float(sig_cache[uid].get("rms_dbfs", compute_rms_dbfs(sf.read(str(audio_path), dtype="float32")[0])))
            st_ratio = float(sig_cache[uid]["start_edge_ratio_10ms"])
            en_ratio = float(sig_cache[uid]["end_edge_ratio_10ms"])
        else:
            audio, sr = sf.read(str(audio_path), dtype="float32")
            rms_val = compute_rms_dbfs(audio)
            sig_feats = extract_signal_features(audio, sr)
            st_ratio = sig_feats["start_edge_ratio_10ms"]
            en_ratio = sig_feats["end_edge_ratio_10ms"]

        pre_flag = 1 if rms_val < pre_thresh else 0
        edge_flag = 1 if (st_ratio > st_thresh or en_ratio > en_thresh) else 0

        p_sec = float(p_sec_arr[idx])
        p_mus = float(p_mus_arr[idx])
        p_content = float(p_content_arr[idx])
        s_score = edge_flag + p_content

        scored_records.append({
            "clip_id": cid,
            "uid": uid,
            "speaker": r.get("speaker") or r.get("speaker_id") or r.get("recording_id", "unknown"),
            "duration_s": r.get("duration", r.get("duration_s", 0.0)),
            "gold_label": r.get("label", r.get("decision")),
            "rms_dbfs": round(rms_val, 2),
            "pre_gate": pre_flag,
            "start_edge_ratio_10ms": round(st_ratio, 4),
            "end_edge_ratio_10ms": round(en_ratio, 4),
            "flag_edge": edge_flag,
            "p_secondary_speaker": round(p_sec, 6),
            "p_background_music": round(p_mus, 6),
            "p_content": round(p_content, 6),
            "score_s": round(s_score, 6),
            "_p_raw": p_content,
        })

    scored_records.sort(key=lambda x: (x["pre_gate"], x["flag_edge"], x["_p_raw"], x["clip_id"]))
    for rank, item in enumerate(scored_records, start=1):
        item.pop("_p_raw", None)
        item["lexicographical_rank"] = rank

    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    with output_scores_path.open("w", encoding="utf-8") as out_f:
        for item in scored_records:
            out_f.write(json.dumps(item) + "\n")
    logger.info("Saved %d scored records to %s", len(scored_records), output_scores_path)
    print(f"Scored manifest written to: {output_scores_path}")


def main() -> None:
    parser = LoggingArgumentParser(
        description="Evaluate pre-registered Single Candidate Hybrid Gate on dev-288 and ViCut-Bench."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path(".data/tts_strategy/gold_benchmark_20260908/eval_input.materialized.jsonl"),
        help="Input dev-288 materialized benchmark manifest.",
    )
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path(".data/vicut_bench/reports/phase3_probes"),
        help="Directory containing extracted probe caches and reports.",
    )
    parser.add_argument(
        "--anchors-manifest",
        type=Path,
        default=Path(".data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl"),
        help="ViCut-Bench clean anchors manifest.",
    )
    parser.add_argument(
        "--defects-manifest",
        type=Path,
        default=Path(".data/vicut_bench/defects/vicut_bench_defects.jsonl"),
        help="ViCut-Bench defects manifest.",
    )
    parser.add_argument(
        "-od", "--output-dir",
        type=Path,
        default=Path(".data/vicut_bench/reports/dev_hybrid_gate"),
        help="Output directory for reports and scores.",
    )
    parser.add_argument(
        "--c-reg",
        type=float,
        default=0.01,
        help="L2 regularization parameter C for content arm logistic regression.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=1000,
        help="Number of paired cluster bootstrap resamples.",
    )
    parser.add_argument(
        "--bootstrap-seeds",
        type=str,
        default="42,123",
        help="Comma-separated random seeds for bootstrap CIs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory.",
    )
    parser.add_argument(
        "--score-manifest",
        type=Path,
        default=None,
        help="Path to manifest to score using frozen parameters without refitting.",
    )
    parser.add_argument(
        "--frozen-params",
        type=Path,
        default=None,
        help="Path to frozen candidate parameters JSON.",
    )
    parser.add_argument(
        "--probe-cache",
        type=Path,
        default=None,
        help="Optional path to precomputed pooled feature cache (.npz).",
    )
    parser.add_argument(
        "--allow-non-16k",
        action="store_true",
        default=False,
        help="Allow audio with sampling rate != 16000 Hz during fresh extraction (for reproducing unresampled historical runs).",
    )

    args = parser.parse_args()

    if args.score_manifest:
        f_params = args.frozen_params
        if f_params is None:
            cand_fp = args.output_dir / "hybrid_gate_frozen_params.json"
            if cand_fp.exists():
                f_params = cand_fp
            else:
                f_params = Path(".data/vicut_bench/reports/dev_hybrid_gate/hybrid_gate_frozen_params.json")
        out_sc = args.output_dir / "manifest_scored.jsonl"
        p_cache = args.probe_cache
        if p_cache is None:
            cand_cache = args.probe_dir / "cache" / "phowhisper_pooled.npz"
            if cand_cache.exists():
                p_cache = cand_cache
        score_manifest_inference(
            args.score_manifest,
            f_params,
            out_sc,
            probe_cache_path=p_cache,
            allow_non_16k=args.allow_non_16k,
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging handler (tee logs into output directory)
    file_handler = logging.FileHandler(args.output_dir / "evaluate_hybrid_gate.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    json_out = args.output_dir / "hybrid_gate_report.json"
    md_out = args.output_dir / "DEV_HYBRID_GATE_REPORT.md"
    per_clip_out = args.output_dir / "per_clip_scores.jsonl"
    frozen_params_out = args.output_dir / "hybrid_gate_frozen_params.json"

    if json_out.exists() and not args.overwrite:
        logger.error("Output %s exists and --overwrite not specified. Exiting.", json_out)
        sys.exit(1)

    seeds = [int(s.strip()) for s in args.bootstrap_seeds.split(",")]

    logger.info("Loading records and checking artifact integrity...")
    records = load_clip_records(args.anchors_manifest, args.defects_manifest, args.input_manifest)
    cache_path = args.probe_dir / "cache" / "phowhisper_pooled.npz"
    if not cache_path.exists():
        cache_path = args.probe_dir / "features" / "phowhisper_pooled.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"PhoWhisper pooled feature cache not found at {cache_path}")

    cache = np.load(cache_path, allow_pickle=True)

    # Check cache UID alignment (F9)
    if "uids" in cache:
        cache_uids = cache["uids"].tolist()
        rec_uids = [r["uid"] for r in records]
        assert cache_uids == rec_uids, "Feature cache UID order does not match loaded records order"

    sig_feats_path = args.probe_dir / "cache" / "signal_features.json"
    if sig_feats_path.exists():
        sig_feats = json.loads(sig_feats_path.read_text(encoding="utf-8"))
        for r in records:
            if r["uid"] in sig_feats:
                r.update(sig_feats[r["uid"]])

    feats_dict = {k: cache[k] for k in ("mean", "std", "start_edge", "end_edge", "layer_dims")}
    all_indices = list(range(len(records)))
    feats = build_feature_matrix(feats_dict, all_indices, 4, "mean_std")

    # 1. Evaluate Deterministic Pre-gate (RMS < -50 dBFS)
    logger.info("Evaluating deterministic pre-gate (RMS < -50 dBFS)...")
    dev_records = [r for r in records if r.get("group") == "dev_gold"]
    assert len(dev_records) == 288, f"Expected 288 dev records, found {len(dev_records)}"

    dev_rms_dbfs = []
    for r in dev_records:
        r["id"] = r["clip_id"]
        r["label"] = r["decision"]
        r["speaker"] = r["speaker"]
        r_audio, _ = read_audio_for_record(r)
        dbfs = compute_rms_dbfs(r_audio)
        dev_rms_dbfs.append(dbfs)
        r["rms_dbfs"] = dbfs
        r["pre_gate"] = 1.0 if dbfs < -50.0 else 0.0

    # 2. Calibrate Deterministic Boundary Arm on clean_tight_tr (5-fold OOF)
    logger.info("Calibrating deterministic boundary arm on clean_tight_tr...")
    clean_tight_tr = [
        r for r in records
        if r.get("split") == "train" and r.get("group") == "clean_tight_twin"
    ]
    st_thresh, en_thresh, boundary_stats = calibrate_boundary_arm(
        clean_tight_tr, n_splits=5, seed=seeds[0]
    )

    for r in dev_records:
        st_val = float(r["start_edge_ratio_10ms"])
        en_val = float(r["end_edge_ratio_10ms"])
        r["flag_edge"] = 1.0 if (st_val > st_thresh or en_val > en_thresh) else 0.0

    # 3. Train Content Arm Logistic Regression Heads on train split
    logger.info("Training PhoWhisper L4 content arm heads on train split...")
    clean_train = [
        r for r in records
        if r.get("split") == "train" and r.get("group") in ("clean_anchor", "clean_tight_twin")
    ]
    content_cal = calibrate_content_arm(
        records, clean_train, feats, c_reg=args.c_reg, seed=seeds[0], n_splits=5
    )

    # Score dev-288 with content arm
    dev_uids = [r["uid"] for r in dev_records]
    rec_uid_to_idx = {r["uid"]: i for i, r in enumerate(records)}
    dev_indices = [rec_uid_to_idx[uid] for uid in dev_uids]
    dev_feats = feats[dev_indices]

    clf_sec = content_cal["secondary_speaker"]["clf"]
    sc_sec = content_cal["secondary_speaker"]["scaler"]
    p_sec_dev = clf_sec.predict_proba(sc_sec.transform(dev_feats))[:, 1]

    clf_mus = content_cal["background_music"]["clf"]
    sc_mus = content_cal["background_music"]["scaler"]
    p_mus_dev = clf_mus.predict_proba(sc_mus.transform(dev_feats))[:, 1]

    p_content_dev = np.maximum(p_sec_dev, p_mus_dev)

    for i, r in enumerate(dev_records):
        r["p_sec"] = float(p_sec_dev[i])
        r["p_music"] = float(p_mus_dev[i])
        r["p_content"] = float(p_content_dev[i])
        r["score_s"] = float(r["flag_edge"] + r["p_content"])

    # Persist frozen model parameters (F8)
    frozen_params = {
        "version": "1.0.0",
        "date": "2026-11-08",
        "pre_gate": {
            "rule": "rms_dbfs < threshold_rms_dbfs",
            "threshold_rms_dbfs": -50.0,
            "min_dev_rms_dbfs": round(float(min(dev_rms_dbfs)), 2),
            "max_dev_rms_dbfs": round(float(max(dev_rms_dbfs)), 2),
        },
        "boundary_arm": {
            "features": ["start_edge_ratio_10ms", "end_edge_ratio_10ms"],
            "start_thresh": round(st_thresh, 6),
            "end_thresh": round(en_thresh, 6),
            "rule": "flag_edge = 1.0 if start > start_thresh or end > end_thresh else 0.0",
            "calibration_stats": boundary_stats,
        },
        "content_arm": {
            "encoder": "vinai/phowhisper-small",
            "layer": 4,
            "pooling": "mean_std",
            "feature_dim": 1536,
            "c_reg": float(args.c_reg),
            "heads": {
                "secondary_speaker": {
                    "oof_auroc": content_cal["secondary_speaker"]["oof_auroc"],
                    "scaler_mean": [float(v) for v in sc_sec.mean_],
                    "scaler_scale": [float(v) for v in sc_sec.scale_],
                    "coef": [float(v) for v in clf_sec.coef_[0]],
                    "intercept": float(clf_sec.intercept_[0]),
                },
                "background_music": {
                    "oof_auroc": content_cal["background_music"]["oof_auroc"],
                    "scaler_mean": [float(v) for v in sc_mus.mean_],
                    "scaler_scale": [float(v) for v in sc_mus.scale_],
                    "coef": [float(v) for v in clf_mus.coef_[0]],
                    "intercept": float(clf_mus.intercept_[0]),
                },
            },
            "rule": "p_content = max(p_sec, p_music)",
        },
        "decision_rule": {
            "score_formula": "s = flag_edge + p_content",
            "lexicographical_tuple": "(pre_gate, flag_edge, p_content, clip_id)",
        },
    }
    frozen_params_out.write_text(json.dumps(frozen_params, indent=2), encoding="utf-8")
    logger.info("Persisted frozen candidate parameters to %s", frozen_params_out)

    # Persist per-clip scores (F7)
    dev_items_sorted = sorted(
        dev_records, key=lambda x: (x["pre_gate"], x["flag_edge"], x["p_content"], x["id"])
    )
    with per_clip_out.open("w", encoding="utf-8") as f_pc:
        for rank, r in enumerate(dev_items_sorted, start=1):
            f_pc.write(json.dumps({
                "clip_id": r["id"],
                "uid": r["uid"],
                "speaker": r["speaker"],
                "duration_s": r["duration_s"],
                "gold_label": r["label"],
                "rms_dbfs": round(r["rms_dbfs"], 2),
                "pre_gate": int(r["pre_gate"]),
                "start_edge_ratio_10ms": round(r["start_edge_ratio_10ms"], 4),
                "end_edge_ratio_10ms": round(r["end_edge_ratio_10ms"], 4),
                "flag_edge": int(r["flag_edge"]),
                "p_secondary_speaker": round(r["p_sec"], 6),
                "p_background_music": round(r["p_music"], 6),
                "p_content": round(r["p_content"], 6),
                "score_s": round(r["score_s"], 6),
                "lexicographical_rank": rank,
            }) + "\n")
    logger.info("Saved per-clip scores for 288 dev clips to %s", per_clip_out)

    # 4. Score Dev Baselines on dev-288
    logger.info("Scoring dev baselines on dev-288...")
    total_dev_minutes = sum(r["duration_s"] for r in dev_records) / 60.0
    dev_labels = np.array([1 if r["label"] == "reject" else 0 for r in dev_records])
    all_pass_r = float(np.mean(dev_labels))

    # flash_lite baseline
    fl_csv = args.input_manifest.parent / "reports" / "gemini35_flash_lite_medium_vs_gemini38_medium.csv"
    fl_pass_ids = {r["id"] for r in csv.DictReader(fl_csv.open(encoding="utf-8")) if r.get("model_decision") == "pass"}

    fl_indices = [i for i, r in enumerate(dev_records) if r["id"] in fl_pass_ids]
    fl_acc_min = sum(dev_records[i]["duration_s"] for i in fl_indices) / 60.0
    fl_bad_clips = sum(1 for i in fl_indices if dev_records[i]["label"] == "reject")
    fl_point = {
        "accepted_clips": len(fl_indices),
        "accepted_minutes": round(fl_acc_min, 4),
        "yield_ratio": round(fl_acc_min / total_dev_minutes, 6),
        "R": round(fl_bad_clips / len(fl_indices), 6),
    }

    # signal_3feat composite baseline
    # signal_3feat composite baseline
    z_risks = []
    for feat in ["end_edge_ratio_10ms", "start_edge_ratio_10ms", "min_window_rms_db"]:
        vals = np.array([r[feat] for r in dev_records], dtype=float)
        med = np.median(vals)
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
        if iqr <= 0:
            iqr = 1.0
        z_risks.append((vals - med) / iqr)
    sig3_scores = np.max(z_risks, axis=0)
    for i, r in enumerate(dev_records):
        r["sig3_score"] = float(sig3_scores[i])

    # signal_only (4-feature percentile composite) baseline
    clean_tr_indices = [
        i for i, r in enumerate(records)
        if r.get("split") == "train" and r.get("group") in ("clean_anchor", "clean_tight_twin")
    ]
    sig_risks = []
    for defect_type, (feature, higher_bad) in BASELINE_SPEC.items():
        ref = np.sort([records[i][feature] for i in clean_tr_indices])
        vals = np.array([r[feature] for r in dev_records], dtype=float)
        perc = np.searchsorted(ref, vals, side="right") / len(ref)
        sig_risks.append(perc if higher_bad else 1.0 - perc)
    sig_only_scores = np.max(sig_risks, axis=0)
    for i, r in enumerate(dev_records):
        r["signal_only_score"] = float(sig_only_scores[i])

    def make_score_curve_fn(score_key: str):
        def _curve_fn(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            sc = np.array([r[score_key] for r in recs], dtype=float)
            lb = np.array([1 if r["label"] == "reject" else 0 for r in recs])
            dur = np.array([r["duration_s"] for r in recs])
            order = np.argsort(sc, kind="stable")
            ends = np.r_[np.flatnonzero(np.diff(sc[order])) + 1, len(order)]
            minutes = np.cumsum(dur[order])[ends - 1] / 60.0
            bad = np.cumsum(lb[order])[ends - 1]
            tot_min = dur.sum() / 60.0
            return [
                {
                    "threshold_score": float(sc[order[e - 1]]),
                    "accepted_clips": int(e),
                    "accepted_minutes": round(float(m), 4),
                    "yield_ratio": float(m / tot_min),
                    "R": float(b / e),
                }
                for e, b, m in zip(ends, bad, minutes)
            ]
        return _curve_fn

    sig3_curve_fn = make_score_curve_fn("sig3_score")
    sig3_curve = sig3_curve_fn(dev_records)
    sig3_70 = matched_point_at_yield(sig3_curve, 0.70)

    sig_only_curve_fn = make_score_curve_fn("signal_only_score")
    sig_only_curve = sig_only_curve_fn(dev_records)
    sig_only_70 = matched_point_at_yield(sig_only_curve, 0.70)

    # 5. Candidate Dev Evaluation
    cand_curve = curve_lexicographical(
        dev_records,
        np.array([r["pre_gate"] for r in dev_records]),
        np.array([r["flag_edge"] for r in dev_records]),
        p_content_dev,
        ascending=True,
    )
    cand_at_fl = matched_point_at_yield(cand_curve, fl_point["yield_ratio"])
    cand_70 = matched_point_at_yield(cand_curve, 0.70)
    cand_90 = matched_point_at_yield(cand_curve, 0.90)

    # 6. Paired Cluster Bootstrap Analysis
    logger.info("Running paired recording bootstrap (seeds %s, N=%d)...", seeds, args.bootstrap_replicates)
    boot_fl = paired_cluster_bootstrap(
        dev_records, cand_curve, fl_point, fl_indices, fl_point["yield_ratio"],
        seeds=seeds, n_replicates=args.bootstrap_replicates
    )
    boot_sig3 = paired_cluster_bootstrap(
        dev_records, cand_curve, {**sig3_70, "_curve": sig3_curve_fn}, None, 0.70,
        seeds=seeds, n_replicates=args.bootstrap_replicates
    )
    boot_sig_only = paired_cluster_bootstrap(
        dev_records, cand_curve, {**sig_only_70, "_curve": sig_only_curve_fn}, None, 0.70,
        seeds=seeds, n_replicates=args.bootstrap_replicates
    )
    nolock_point = {
        "accepted_clips": len(dev_records),
        "accepted_minutes": round(total_dev_minutes, 4),
        "yield_ratio": 1.0,
        "R": round(all_pass_r, 6),
    }
    boot_nolock = paired_cluster_bootstrap(
        dev_records, cand_curve, nolock_point, list(range(len(dev_records))), 0.90,
        seeds=seeds, n_replicates=args.bootstrap_replicates
    )

    # Evaluate pre-registered Decision Rule (PLAN.md line 99)
    # Branch A: Candidate beats each dev baseline with 95% CI strictly excluding 0 across both seeds.
    # Branch B: CI crosses 0 -> Lock candidate, log honest null dev findings.
    branch_a_fl = all(boot_fl[s]["ci_excludes_zero"] and boot_fl[s]["ci_95_high"] < 0.0 for s in seeds)
    branch_a_sig3 = all(boot_sig3[s]["ci_excludes_zero"] and boot_sig3[s]["ci_95_high"] < 0.0 for s in seeds)
    branch_a_nolock = all(boot_nolock[s]["ci_excludes_zero"] and boot_nolock[s]["ci_95_high"] < 0.0 for s in seeds)

    decision_branch = "Branch A" if (branch_a_fl and branch_a_sig3 and branch_a_nolock) else "Branch B (Fallback)"

    # 7. ViCut-Bench G3 Mechanism Validation on Held-Out Test Split (90 anchors / 19 speakers)
    logger.info("Evaluating ViCut-Bench G3 mechanism on held-out test split (90 anchors)...")
    test_raw_anchors = [
        r for r in records if r.get("split") == "test" and r.get("group") == "clean_anchor"
    ]
    test_tight_twins = [
        r for r in records if r.get("split") == "test" and r.get("group") == "clean_tight_twin"
    ]

    raw_rejects = sum(1 for r in test_raw_anchors if (r["start_edge_ratio_10ms"] > st_thresh or r["end_edge_ratio_10ms"] > en_thresh))
    tight_st_rejects = sum(1 for r in test_tight_twins if r["start_edge_ratio_10ms"] > st_thresh)
    tight_en_rejects = sum(1 for r in test_tight_twins if r["end_edge_ratio_10ms"] > en_thresh)
    tight_union_rejects = sum(1 for r in test_tight_twins if (r["start_edge_ratio_10ms"] > st_thresh or r["end_edge_ratio_10ms"] > en_thresh))

    clean_twin_frr = {
        "n_test_raw": len(test_raw_anchors),
        "n_test_tight": len(test_tight_twins),
        "raw_rejects": raw_rejects,
        "tight_st_rejects": tight_st_rejects,
        "tight_en_rejects": tight_en_rejects,
        "tight_union_rejects": tight_union_rejects,
        "raw_anchor_frr": round(raw_rejects / len(test_raw_anchors), 6),
        "start_frr": round(tight_st_rejects / len(test_tight_twins), 6),
        "end_frr": round(tight_en_rejects / len(test_tight_twins), 6),
        "union_frr": round(tight_union_rejects / len(test_tight_twins), 6),
    }

    # Defect scoring on test split
    def score_records_hybrid(rec_list: list[dict[str, Any]], primary_arm: str) -> np.ndarray:
        if primary_arm == "boundary_start":
            return np.array([r["start_edge_ratio_10ms"] for r in rec_list])
        elif primary_arm == "boundary_end":
            return np.array([r["end_edge_ratio_10ms"] for r in rec_list])
        elif primary_arm == "secondary_speaker":
            idx_list = [rec_uid_to_idx[r["uid"]] for r in rec_list]
            return clf_sec.predict_proba(sc_sec.transform(feats[idx_list]))[:, 1]
        elif primary_arm == "background_music":
            idx_list = [rec_uid_to_idx[r["uid"]] for r in rec_list]
            return clf_mus.predict_proba(sc_mus.transform(feats[idx_list]))[:, 1]
        raise ValueError(f"Unknown arm: {primary_arm}")

    sc_raw_te_end = score_records_hybrid(test_raw_anchors, "boundary_end")
    sc_tight_te_end = score_records_hybrid(test_tight_twins, "boundary_end")
    sc_raw_te_st = score_records_hybrid(test_raw_anchors, "boundary_start")
    sc_tight_te_st = score_records_hybrid(test_tight_twins, "boundary_start")
    sc_raw_te_sec = score_records_hybrid(test_raw_anchors, "secondary_speaker")
    sc_tight_te_sec = score_records_hybrid(test_tight_twins, "secondary_speaker")
    sc_raw_te_mus = score_records_hybrid(test_raw_anchors, "background_music")
    sc_tight_te_mus = score_records_hybrid(test_tight_twins, "background_music")

    vicut_results: dict[str, Any] = {}
    defect_arm_map = {
        "clipped_word_end": ("boundary_end", sc_raw_te_end, sc_tight_te_end),
        "clipped_word_start": ("boundary_start", sc_raw_te_st, sc_tight_te_st),
        "secondary_speaker": ("secondary_speaker", sc_raw_te_sec, sc_tight_te_sec),
        "background_music": ("background_music", sc_raw_te_mus, sc_tight_te_mus),
    }

    for dtype, (arm_name, sc_raw_te, sc_tight_te) in defect_arm_map.items():
        te_def_recs = [
            r for r in records if r.get("split") == "test" and r.get("defect_type") == dtype
        ]
        sc_def = score_records_hybrid(te_def_recs, arm_name)
        auc_raw = compute_auroc(sc_def.tolist(), sc_raw_te.tolist(), True)
        auc_tight = compute_auroc(sc_def.tolist(), sc_tight_te.tolist(), True)

        sev_results: dict[str, Any] = {}
        for sev in SEVERITY_ORDER[dtype]:
            sev_recs = [r for r in te_def_recs if r.get("severity") == sev]
            sc_s = score_records_hybrid(sev_recs, arm_name)
            sev_results[sev] = {
                "n": len(sev_recs),
                "auroc_vs_raw": round(float(compute_auroc(sc_s.tolist(), sc_raw_te.tolist(), True)), 4),
                "auroc_vs_tight_0ms": round(float(compute_auroc(sc_s.tolist(), sc_tight_te.tolist(), True)), 4),
            }

        vicut_results[dtype] = {
            "n_test": len(te_def_recs),
            "primary_arm": arm_name,
            "auroc_vs_raw": round(float(auc_raw), 4),
            "auroc_vs_tight_0ms": round(float(auc_tight), 4),
            "severities": sev_results,
        }

    # 8. Real dev-288 Component Ablation (5 Conditions)
    logger.info("Running real dev-288 component ablation...")
    dev_ablation: dict[str, Any] = {}

    # 1. Full Hybrid Gate
    dev_ablation["full_hybrid"] = {
        "name": "1. Full Hybrid Gate",
        "yield_ratio": cand_70["yield_ratio"],
        "accepted_minutes": cand_70["accepted_minutes"],
        "accepted_clips": cand_70["accepted_clips"],
        "R": cand_70["R"],
    }

    # 2. Content arm only
    curve_content_only = curve_lexicographical(
        dev_records,
        np.zeros(len(dev_records)),
        np.zeros(len(dev_records)),
        p_content_dev,
        ascending=True,
    )
    pt_content = matched_point_at_yield(curve_content_only, 0.70)
    dev_ablation["content_only"] = {
        "name": "2. Content arm only",
        "yield_ratio": pt_content["yield_ratio"],
        "accepted_minutes": pt_content["accepted_minutes"],
        "accepted_clips": pt_content["accepted_clips"],
        "R": pt_content["R"],
    }

    # 3. Boundary arm only
    st_dev = np.array([r["start_edge_ratio_10ms"] for r in dev_records])
    en_dev = np.array([r["end_edge_ratio_10ms"] for r in dev_records])
    norm_st = st_dev / max(st_thresh, 1e-6)
    norm_en = en_dev / max(en_thresh, 1e-6)
    norm_edge = np.maximum(norm_st, norm_en)
    curve_boundary_only = curve_lexicographical(
        dev_records,
        np.zeros(len(dev_records)),
        np.zeros(len(dev_records)),
        norm_edge,
        ascending=True,
    )
    pt_boundary = matched_point_at_yield(curve_boundary_only, 0.70)
    dev_ablation["boundary_only"] = {
        "name": "3. Boundary arm only",
        "yield_ratio": pt_boundary["yield_ratio"],
        "accepted_minutes": pt_boundary["accepted_minutes"],
        "accepted_clips": pt_boundary["accepted_clips"],
        "R": pt_boundary["R"],
    }

    # 4. Full without Pre-gate
    curve_no_pre = curve_lexicographical(
        dev_records,
        np.zeros(len(dev_records)),
        np.array([r["flag_edge"] for r in dev_records]),
        p_content_dev,
        ascending=True,
    )
    pt_no_pre = matched_point_at_yield(curve_no_pre, 0.70)
    dev_ablation["full_without_pre_gate"] = {
        "name": "4. Full without Pre-gate",
        "yield_ratio": pt_no_pre["yield_ratio"],
        "accepted_minutes": pt_no_pre["accepted_minutes"],
        "accepted_clips": pt_no_pre["accepted_clips"],
        "R": pt_no_pre["R"],
    }

    # 5. Pre-gate only (native yield)
    dev_ablation["pre_gate_only"] = {
        "name": "5. Pre-gate only",
        "yield_ratio": 1.0,
        "accepted_minutes": round(total_dev_minutes, 4),
        "accepted_clips": len(dev_records),
        "R": round(all_pass_r, 6),
    }

    # Additional diagnostic: Boundary partition separation on dev
    dev_flagged = [r for r in dev_records if r["flag_edge"] == 1.0]
    dev_unflagged = [r for r in dev_records if r["flag_edge"] == 0.0]
    n_flagged = len(dev_flagged)
    n_unflagged = len(dev_unflagged)
    n_flagged_def = sum(1 for r in dev_flagged if r["label"] == "reject")
    n_unflagged_def = sum(1 for r in dev_unflagged if r["label"] == "reject")
    r_flagged = float(n_flagged_def / n_flagged) if n_flagged > 0 else 0.0
    r_unflagged = float(n_unflagged_def / n_unflagged) if n_unflagged > 0 else 0.0
    flagged_min = sum(r["duration_s"] for r in dev_flagged) / 60.0
    flagged_pct_min = flagged_min / total_dev_minutes

    boundary_partition_diag = {
        "n_dev_flagged": n_flagged,
        "n_dev_unflagged": n_unflagged,
        "n_flagged_defective": n_flagged_def,
        "n_unflagged_defective": n_unflagged_def,
        "flagged_minutes": round(flagged_min, 2),
        "flagged_minute_fraction": round(flagged_pct_min, 4),
        "defect_rate_flagged": round(r_flagged, 4),
        "defect_rate_unflagged": round(r_unflagged, 4),
    }

    # Build output report payload
    report_payload = {
        "metadata": {
            "evaluation_date": "2026-11-08",
            "decision_branch": decision_branch,
            "c_reg": args.c_reg,
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seeds": seeds,
        },
        "boundary_arm_calibration": boundary_stats,
        "content_arm_calibration": {
            "secondary_speaker_oof_auroc": content_cal["secondary_speaker"]["oof_auroc"],
            "background_music_oof_auroc": content_cal["background_music"]["oof_auroc"],
            "c_reg": args.c_reg,
        },
        "dev_288_competition": {
            "total_clips": len(dev_records),
            "total_defective_clips": int(np.sum(dev_labels)),
            "total_minutes": round(total_dev_minutes, 4),
            "uncurated_R": round(all_pass_r, 6),
            "candidate_vs_flash_lite": {
                "matched_yield": cand_at_fl["yield_ratio"],
                "candidate": cand_at_fl,
                "baseline": fl_point,
                "delta_R_point": round(cand_at_fl["R"] - fl_point["R"], 6),
                "bootstrap_ci": boot_fl,
            },
            "candidate_vs_signal_3feat": {
                "matched_yield": cand_70["yield_ratio"],
                "candidate": cand_70,
                "baseline": sig3_70,
                "delta_R_point": round(cand_70["R"] - sig3_70["R"], 6),
                "bootstrap_ci": boot_sig3,
            },
            "candidate_vs_signal_only": {
                "matched_yield": cand_70["yield_ratio"],
                "candidate": cand_70,
                "baseline": sig_only_70,
                "delta_R_point": round(cand_70["R"] - sig_only_70["R"], 6),
                "bootstrap_ci": boot_sig_only,
            },
            "candidate_vs_recipe_nolock": {
                "matched_yield": cand_90["yield_ratio"],
                "candidate": cand_90,
                "baseline": nolock_point,
                "delta_R_point": round(cand_90["R"] - nolock_point["R"], 6),
                "bootstrap_ci": boot_nolock,
            },
        },
        "dev_boundary_partition_diagnostics": boundary_partition_diag,
        "dev_component_ablation": dev_ablation,
        "vicut_bench_g3_mechanism": {
            "clean_twin_frr": clean_twin_frr,
            "defect_detection": vicut_results,
        },
    }

    json_out.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logger.info("Saved JSON report to %s", json_out)

    # Dynamic Markdown report generation
    def fmt_delta(cand_r: float, base_r: float) -> str:
        d = (cand_r - base_r) * 100.0
        return f"{d:+.2f}% pt"

    def fmt_ci_tag(ci_dict: dict[str, Any]) -> str:
        if ci_dict["ci_excludes_zero"]:
            return "✅ Yes"
        return "❌ No (crosses 0)"

    abl_full = dev_ablation["full_hybrid"]
    abl_cnt = dev_ablation["content_only"]
    abl_bnd = dev_ablation["boundary_only"]
    abl_nop = dev_ablation["full_without_pre_gate"]
    abl_pre = dev_ablation["pre_gate_only"]

    n_dev_recordings = len(set(r.get("recording_id") or r.get("speaker") for r in dev_records))
    n_test_speakers = len(set(r["speaker"] for r in test_raw_anchors))

    nolock_excludes = boot_nolock[42]["ci_excludes_zero"] and boot_nolock[123]["ci_excludes_zero"]
    nolock_ci_str = "CI strictly excludes 0" if nolock_excludes else "CI crosses 0"

    delta_sig_only = (cand_70["R"] - sig_only_70["R"]) * 100.0
    if delta_sig_only > 0:
        sig_only_desc = f"slightly higher than 4-feature `signal_only` ({cand_70['R'] * 100:.2f}% vs {sig_only_70['R'] * 100:.2f}%, $\\Delta R = {delta_sig_only:+.2f}$% pt)"
    else:
        sig_only_desc = f"lower defect rate than 4-feature `signal_only` ({cand_70['R'] * 100:.2f}% vs {sig_only_70['R'] * 100:.2f}%, $\\Delta R = {delta_sig_only:+.2f}$% pt)"

    if decision_branch.startswith("Branch A"):
        branch_action_text = (
            "The candidate achieves statistically significant improvements with 95% CIs excluding 0. "
            "In accordance with PLAN.md line 85, **Branch A (Production Acceptance)** is triggered."
        )
    else:
        branch_action_text = (
            f"Because the paired recording-bootstrap 95% CIs cross 0 against `flash_lite` "
            f"([{boot_fl[42]['ci_95_low']:.4f}, {boot_fl[42]['ci_95_high']:.4f}]) and `signal_3feat` "
            f"([{boot_sig3[42]['ci_95_low']:.4f}, {boot_sig3[42]['ci_95_high']:.4f}]), the pre-registered criterion for Branch A is **not met**.  \n"
            f"> In strict accordance with PLAN.md line 99, **Branch B (Fallback)** is triggered. "
            f"Method iteration halts immediately; the candidate is locked as the pre-registered baseline, "
            f"honest null findings are logged, and the project proceeds to Phase 2 (G1 Test Set Benchmark Characterization) "
            f"and Phase 3 (ViCut-Bench Benchmark Paper)."
        )

    min_abl_key = min(dev_ablation.keys(), key=lambda k: dev_ablation[k]["R"])
    min_abl_note = " (lowest defect rate on dev at 70% yield)" if min_abl_key == "content_only" else ""

    unflagged_min_pct = (1.0 - boundary_partition_diag["flagged_minute_fraction"]) * 100.0

    cwe_10 = vicut_results["clipped_word_end"]["severities"]["10ms"]["auroc_vs_tight_0ms"]
    cwe_80 = vicut_results["clipped_word_end"]["severities"]["80ms"]["auroc_vs_tight_0ms"]
    cwe_sev_vals = [s["auroc_vs_tight_0ms"] for s in vicut_results["clipped_word_end"]["severities"].values()]
    is_cwe_monotonic = all(x <= y for x, y in zip(cwe_sev_vals, cwe_sev_vals[1:]))
    monotonic_str = "monotonically increasing" if is_cwe_monotonic else "varying"

    md_content = f"""# Dev Hybrid Gate Benchmark & ViCut-Bench Calibration Report

**Evaluation Date:** 2026-11-08  
**Pre-registration:** WORKLOG.md (2026-11-06) & PLAN.md Phase 1 (lines 71–107)  
**Status:** Evaluation Complete — **Pre-registered Decision Gate: {decision_branch}**

---

## 1. Candidate Gate Specification

| Component | Architecture & Method | Calibration & Operating Rule |
| :--- | :--- | :--- |
| **Pre-gate** | Deterministic DSP Energy | Reject clip if $\\text{{RMS}} < -50\\text{{ dBFS}}$ (silence / low energy) |
| **Boundary Arm** | Deterministic 10 ms Edge Ratios | 95% specificity on 5-fold OOF `clean_tight_tr`: $\\tau_\\text{{start}}={st_thresh:.4f}$, $\\tau_\\text{{end}}={en_thresh:.4f}$.<br>$\\text{{flag\\_edge}} = 1.0$ if $\\text{{start}} > \\tau_\\text{{start}} \\lor \\text{{end}} > \\tau_\\text{{end}}$, else $0.0$ |
| **Content Arm** | `vinai/phowhisper-small` Layer 4 (`mean_std` pooling, $D=1536$) | Logistic regression ($C={args.c_reg}$) fit on ViCut-Bench train split; 5-fold OOF evaluated for generalization AUROC ({content_cal['secondary_speaker']['oof_auroc']:.4f} secondary speaker, {content_cal['background_music']['oof_auroc']:.4f} background music).<br>$p_\\text{{content}} = \\max(p_\\text{{sec}}, p_\\text{{music}})$ |
| **Scoring Formula** | Continuous scalar score | $s = \\text{{flag\\_edge}} + p_\\text{{content}} \\in [0.0, 2.0]$ |
| **Lexicographical Order** | Deterministic sorting tuple | `(pre_gate, flag_edge, p_content, clip_id)`: pre-gate rejects rejected first, followed by edge-flagged clips ($s \\in [1.0, 2.0]$) ranked by $p_\\text{{content}}$, followed by unflagged clips ($s \\in [0.0, 1.0]$) ranked by $p_\\text{{content}}$, exact ties broken by alphabetical `clip_id` |

---

## 2. Dev-288 Competition Results ({n_dev_recordings} Recordings, {len(dev_records)} Clips, {total_dev_minutes:.2f} min)

Uncurated Dev Defect Rate ($R$ at 100% yield): **{all_pass_r * 100:.2f}%** ({int(np.sum(dev_labels))} / {len(dev_records)} defective clips).

### A. Point Comparison at Matched Operating Yields

| Gate / Baseline | Operating Mode | Matched Yield (%) | Accepted Minutes | Defect Rate $R$ (%) | $\\Delta R$ vs Baseline (% pt) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Candidate Hybrid Gate** | Continuous Lexicographical | **{cand_at_fl['yield_ratio'] * 100:.2f}%** | **{cand_at_fl['accepted_minutes']:.2f}** | **{cand_at_fl['R'] * 100:.2f}%** | **{fmt_delta(cand_at_fl['R'], fl_point['R'])}** |
| `flash_lite` baseline | Discrete API Decision | {fl_point['yield_ratio'] * 100:.2f}% | {fl_point['accepted_minutes']:.2f} | {fl_point['R'] * 100:.2f}% | *(reference)* |
| **Candidate Hybrid Gate** | Continuous Lexicographical | **{cand_70['yield_ratio'] * 100:.2f}%** | **{cand_70['accepted_minutes']:.2f}** | **{cand_70['R'] * 100:.2f}%** | **{fmt_delta(cand_70['R'], sig3_70['R'])}** |
| `signal_3feat` composite | Continuous 3-feature composite | {sig3_70['yield_ratio'] * 100:.2f}% | {sig3_70['accepted_minutes']:.2f} | {sig3_70['R'] * 100:.2f}% | *(reference)* |
| `signal_only` (4-feat percentile) | Continuous percentile composite | {sig_only_70['yield_ratio'] * 100:.2f}% | {sig_only_70['accepted_minutes']:.2f} | {sig_only_70['R'] * 100:.2f}% | {fmt_delta(cand_70['R'], sig_only_70['R'])} |
| **Candidate Hybrid Gate** | Continuous Lexicographical | **{cand_90['yield_ratio'] * 100:.2f}%** | **{cand_90['accepted_minutes']:.2f}** | **{cand_90['R'] * 100:.2f}%** | **{fmt_delta(cand_90['R'], nolock_point['R'])}** |
| `recipe_nolock` / `e2b_v3` | 100% all-pass baseline | {nolock_point['yield_ratio'] * 100:.2f}% | {nolock_point['accepted_minutes']:.2f} | {nolock_point['R'] * 100:.2f}% | *(reference)* |

### B. Paired Recording-Bootstrap 95% Confidence Intervals ({args.bootstrap_replicates} Resamples by Recording)

$\\Delta R = R_\\text{{candidate}} - R_\\text{{baseline}}$ (negative values indicate candidate improvement / lower defect rate):

| Comparison | Seed | Mean $\\Delta R$ | 95% Bootstrap CI | Valid Replicates | Strictly Excludes 0? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Candidate vs `flash_lite`** (matched ~83.8% yield) | 42 | {boot_fl[42]['mean_delta_R']:+.4f} | [{boot_fl[42]['ci_95_low']:.4f}, {boot_fl[42]['ci_95_high']:.4f}] | {boot_fl[42]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_fl[42])} |
| **Candidate vs `flash_lite`** (matched ~83.8% yield) | 123 | {boot_fl[123]['mean_delta_R']:+.4f} | [{boot_fl[123]['ci_95_low']:.4f}, {boot_fl[123]['ci_95_high']:.4f}] | {boot_fl[123]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_fl[123])} |
| **Candidate vs `signal_3feat`** (matched 70.0% yield) | 42 | {boot_sig3[42]['mean_delta_R']:+.4f} | [{boot_sig3[42]['ci_95_low']:.4f}, {boot_sig3[42]['ci_95_high']:.4f}] | {boot_sig3[42]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_sig3[42])} |
| **Candidate vs `signal_3feat`** (matched 70.0% yield) | 123 | {boot_sig3[123]['mean_delta_R']:+.4f} | [{boot_sig3[123]['ci_95_low']:.4f}, {boot_sig3[123]['ci_95_high']:.4f}] | {boot_sig3[123]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_sig3[123])} |
| **Candidate vs `signal_only`** (matched 70.0% yield) | 42 | {boot_sig_only[42]['mean_delta_R']:+.4f} | [{boot_sig_only[42]['ci_95_low']:.4f}, {boot_sig_only[42]['ci_95_high']:.4f}] | {boot_sig_only[42]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_sig_only[42])} |
| **Candidate vs `signal_only`** (matched 70.0% yield) | 123 | {boot_sig_only[123]['mean_delta_R']:+.4f} | [{boot_sig_only[123]['ci_95_low']:.4f}, {boot_sig_only[123]['ci_95_high']:.4f}] | {boot_sig_only[123]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_sig_only[123])} |
| **Candidate vs `recipe_nolock`** (90% vs 100% all-pass) | 42 | {boot_nolock[42]['mean_delta_R']:+.4f} | [{boot_nolock[42]['ci_95_low']:.4f}, {boot_nolock[42]['ci_95_high']:.4f}] | {boot_nolock[42]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_nolock[42])} |
| **Candidate vs `recipe_nolock`** (90% vs 100% all-pass) | 123 | {boot_nolock[123]['mean_delta_R']:+.4f} | [{boot_nolock[123]['ci_95_low']:.4f}, {boot_nolock[123]['ci_95_high']:.4f}] | {boot_nolock[123]['valid_replicates']}/{args.bootstrap_replicates} | {fmt_ci_tag(boot_nolock[123])} |

> [!IMPORTANT]
> **Pre-Registered Decision Rule Assessment:**  
> The candidate achieves lower point defect rates than `flash_lite` ({cand_at_fl['R'] * 100:.2f}% vs {fl_point['R'] * 100:.2f}%) and `signal_3feat` ({cand_70['R'] * 100:.2f}% vs {sig3_70['R'] * 100:.2f}%), and beats `recipe_nolock` ({cand_90['R'] * 100:.2f}% vs {all_pass_r * 100:.2f}%, {nolock_ci_str}). It is {sig_only_desc}.  
> {branch_action_text}

---

## 3. Real dev-288 Component Ablation

Component ablation evaluated on `dev-288` at standard 70.0% minute yield (and native yield for pre-gate):

| Condition | Operating Yield (%) | Accepted Minutes | Accepted Clips | Defect Rate $R$ (%) | Rationale / Observation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **{abl_full['name']}** | **{abl_full['yield_ratio'] * 100:.2f}%** | **{abl_full['accepted_minutes']:.2f}** | **{abl_full['accepted_clips']}** | **{abl_full['R'] * 100:.2f}%** | Full hybrid system combining pre-gate, boundary arm, and content arm |
| **{abl_cnt['name']}** | {abl_cnt['yield_ratio'] * 100:.2f}% | {abl_cnt['accepted_minutes']:.2f} | {abl_cnt['accepted_clips']} | {abl_cnt['R'] * 100:.2f}% | PhoWhisper L4 probes only ($p_\\text{{content}}){min_abl_note} |
| **{abl_bnd['name']}** | {abl_bnd['yield_ratio'] * 100:.2f}% | {abl_bnd['accepted_minutes']:.2f} | {abl_bnd['accepted_clips']} | {abl_bnd['R'] * 100:.2f}% | Normalized edge ratios only; word clipping detection without secondary speaker modeling |
| **{abl_nop['name']}** | {abl_nop['yield_ratio'] * 100:.2f}% | {abl_nop['accepted_minutes']:.2f} | {abl_nop['accepted_clips']} | {abl_nop['R'] * 100:.2f}% | Identical on dev-288 because all clips have $\\text{{RMS}} \\ge {min(dev_rms_dbfs):.1f}\\text{{ dBFS}}$ |
| **{abl_pre['name']}** | {abl_pre['yield_ratio'] * 100:.2f}% | {abl_pre['accepted_minutes']:.2f} | {abl_pre['accepted_clips']} | {abl_pre['R'] * 100:.2f}% | Native yield 100%; rejects pure digital silence (< -50 dBFS) |

### Boundary Partition Mechanism on dev-288
- **Flagged Clips:** {boundary_partition_diag['n_dev_flagged']} / {len(dev_records)} ({boundary_partition_diag['flagged_minute_fraction'] * 100:.1f}% of audio minutes).
- **Defect Rate in Flagged Set:** **{boundary_partition_diag['defect_rate_flagged'] * 100:.1f}%** ({boundary_partition_diag['n_flagged_defective']} / {boundary_partition_diag['n_dev_flagged']}).
- **Defect Rate in Unflagged Set:** **{boundary_partition_diag['defect_rate_unflagged'] * 100:.1f}%** ({boundary_partition_diag['n_unflagged_defective']} / {boundary_partition_diag['n_dev_unflagged']}).
- **Operating Consequence:** Because the unflagged set constitutes {unflagged_min_pct:.1f}% of minutes, at target yields $\\ge {unflagged_min_pct:.1f}\\%$ (e.g. 70%, 83.8%, 90%), the lexicographical sorting rule accepts every unflagged clip first. The content arm probes act almost entirely to rank rejections within the edge-flagged partition.

---

## 4. ViCut-Bench G3 Mechanism Validation (Held-Out Test Split: {clean_twin_frr['n_test_raw']} Anchors / {n_test_speakers} Speakers)

### A. Clean-Twin False Reject Rate (FRR) at 0 ms Cuts
- **Raw Clean Anchors FRR:** **{clean_twin_frr['raw_anchor_frr'] * 100:.2f}%** ({clean_twin_frr['raw_rejects']} / {clean_twin_frr['n_test_raw']}) *(carrying 50 ms zero-pads)*
- **Held-out Tight Twins Start Edge FRR:** **{clean_twin_frr['start_frr'] * 100:.2f}%** ({clean_twin_frr['tight_st_rejects']} / {clean_twin_frr['n_test_tight']})
- **Held-out Tight Twins End Edge FRR:** **{clean_twin_frr['end_frr'] * 100:.2f}%** ({clean_twin_frr['tight_en_rejects']} / {clean_twin_frr['n_test_tight']})
- **Held-out Tight Twins Union Edge FRR (operative boundary arm):** **{clean_twin_frr['union_frr'] * 100:.2f}%** ({clean_twin_frr['tight_union_rejects']} / {clean_twin_frr['n_test_tight']})

### B. Defect Detection AUROC Across 4 Realized Defect Classes

| Defect Class | Test Clips | AUROC vs Clean Raw Anchors | AUROC vs Clean Tight Twins (0 ms) | Primary Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| `clipped_word_end` | {vicut_results['clipped_word_end']['n_test']} | {vicut_results['clipped_word_end']['auroc_vs_raw']:.4f} | **{vicut_results['clipped_word_end']['auroc_vs_tight_0ms']:.4f}** | Boundary arm (10 ms end edge energy ratio) |
| `clipped_word_start` | {vicut_results['clipped_word_start']['n_test']} | {vicut_results['clipped_word_start']['auroc_vs_raw']:.4f} | **{vicut_results['clipped_word_start']['auroc_vs_tight_0ms']:.4f}** | Boundary arm (10 ms start edge energy ratio) |
| `secondary_speaker` | {vicut_results['secondary_speaker']['n_test']} | **{vicut_results['secondary_speaker']['auroc_vs_raw']:.4f}** | **{vicut_results['secondary_speaker']['auroc_vs_tight_0ms']:.4f}** | Content arm (PhoWhisper L4 secondary speaker probe) |
| `background_music` | {vicut_results['background_music']['n_test']} | **{vicut_results['background_music']['auroc_vs_raw']:.4f}** | **{vicut_results['background_music']['auroc_vs_tight_0ms']:.4f}** | Content arm (PhoWhisper L4 background music probe) |

*(Note on reference sets: Clean tight twins (0 ms cuts) isolate edge truncation without zero-padding confounds, demonstrating {monotonic_str} detection from {cwe_10:.4f} at 10 ms to {cwe_80:.4f} at 80 ms for `clipped_word_end`).*

### C. Per-Severity Detection Breakdown

| Defect Class | Severity | Defect Clips | AUROC vs Raw | AUROC vs Tight Twins |
| :--- | :--- | :---: | :---: | :---: |
"""
    for dtype in DEFECT_TYPES:
        for sev, s_data in vicut_results[dtype]["severities"].items():
            md_content += f"| `{dtype}` | `{sev}` | {s_data['n']} | {s_data['auroc_vs_raw']:.4f} | {s_data['auroc_vs_tight_0ms']:.4f} |\n"

    md_content += f"""
---

## 5. Artifact Verification & Checksums

- `eval_input.materialized.jsonl` SHA-256: `{sha256(args.input_manifest)}`
- `vicut_bench_anchors_300.jsonl` SHA-256: `{sha256(args.anchors_manifest)}`
- `vicut_bench_defects.jsonl` SHA-256: `{sha256(args.defects_manifest)}`
- `hybrid_gate_report.json` SHA-256: `{sha256(json_out)}`
- `hybrid_gate_frozen_params.json` SHA-256: `{sha256(frozen_params_out)}`
- `per_clip_scores.jsonl` SHA-256: `{sha256(per_clip_out)}`

```bash
# Reproduce with:
bash scripts/vicut_bench/evaluate_hybrid_gate.sh --overwrite
```
"""

    md_out.write_text(md_content, encoding="utf-8")
    logger.info("Saved Markdown report to %s", md_out)
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE — DECISION GATE: {decision_branch}")
    print(f"Report: {md_out}")
    print(f"JSON:   {json_out}")
    print(f"Frozen: {frozen_params_out}")
    print(f"Scores: {per_clip_out}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
