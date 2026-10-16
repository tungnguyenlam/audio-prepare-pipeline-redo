#!/usr/bin/env python3
"""Evaluate cached synthetic-trained probe composites on Gemini-labelled dev audio only."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser
from vicut_bench.probe_encoder_layers import (
    BASELINE_SPEC, DEFECT_TYPES, build_feature_matrix, fit_and_score_probe, load_clip_records,
)


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def curve(scores, labels, durations):
    """Whole-score-group acceptance; never break ties using labels or manifest order."""
    order = np.argsort(scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order])) + 1, len(order)]
    minutes = np.cumsum(durations[order])[ends - 1] / 60
    bad = np.cumsum(labels[order])[ends - 1]
    return [dict(threshold=float(scores[order[e - 1]]), R=float(b / e),
                 accepted_clips=int(e), accepted_minutes=float(m),
                 yield_ratio=float(m / (durations.sum() / 60)))
            for e, b, m in zip(ends, bad, minutes)]


def matched_point(scores, labels, durations, target):
    points = curve(scores, labels, durations)
    p = min(points, key=lambda x: abs(x['yield_ratio'] - target))
    return p if abs(p['yield_ratio'] - target) <= target * 0.10 else None


def empirical_cdf(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1) / len(values)
    # Assign the upper rank to tied values so tied groups stay together.
    unique_vals, inverse = np.unique(values, return_inverse=True)
    max_ranks = np.zeros(len(unique_vals), dtype=float)
    np.maximum.at(max_ranks, inverse, ranks)
    return max_ranks[inverse]


def auroc(pos, neg):
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))


def oof_clean_scores(feats, records, clean_anchor_tr, clean_tight_tr, pos_tr, layer, pooling, defect, n_splits=5):
    from sklearn.model_selection import GroupKFold
    train_idx = clean_anchor_tr + clean_tight_tr + pos_tr
    y = np.r_[np.zeros(len(clean_anchor_tr) + len(clean_tight_tr)), np.ones(len(pos_tr))]
    train_spks = np.array([records[i]['speaker'] for i in train_idx])
    anchor_spks = np.array([records[i]['speaker'] for i in clean_tight_tr])
    X_tr = build_feature_matrix(feats, train_idx, layer, pooling, defect)
    oof_a = np.zeros(len(clean_anchor_tr), dtype=float)
    oof_t = np.zeros(len(clean_tight_tr), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for _, val_pos in gkf.split(clean_tight_tr, groups=anchor_spks):
        val_spks = set(anchor_spks[val_pos])
        tr_mask = np.array([s not in val_spks for s in train_spks])
        eval_mats = {
            'a': build_feature_matrix(feats, [clean_anchor_tr[j] for j in val_pos], layer, pooling, defect),
            't': build_feature_matrix(feats, [clean_tight_tr[j] for j in val_pos], layer, pooling, defect),
        }
        fold_scores = fit_and_score_probe(X_tr[tr_mask], y[tr_mask], eval_mats)
        oof_a[val_pos] = fold_scores['a']
        oof_t[val_pos] = fold_scores['t']
    return oof_a, oof_t


def main():
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument('--probe-dir', type=Path, default=Path('.data/vicut_bench/reports/phase3_probes'))
    parser.add_argument('--anchors-manifest', type=Path, default=Path('.data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl'))
    parser.add_argument('--defects-manifest', type=Path, default=Path('.data/vicut_bench/defects/vicut_bench_defects.jsonl'))
    parser.add_argument('-im', '--input-manifest', type=Path, default=Path('.data/tts_strategy/gold_benchmark_20260908/eval_input.materialized.jsonl'))
    parser.add_argument('-od', '--output-dir', type=Path, default=Path('.data/vicut_bench/reports/composite_probe_gate'))
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    parser.add_argument('--bootstrap-seeds', default='42,123')
    parser.add_argument('-w', '--overwrite', action='store_true')
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        parser.error('--bootstrap-replicates must be positive')
    # This command deliberately accepts only the established diagnostic manifest.
    expected = Path('.data/tts_strategy/gold_benchmark_20260908/eval_input.materialized.jsonl').resolve()
    if args.input_manifest.resolve() != expected:
        parser.error('Only the established Gemini-labelled dev manifest is allowed')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        parser.error('Output directory is not empty; use --overwrite or another directory')
    records = load_clip_records(args.anchors_manifest, args.defects_manifest, args.input_manifest)
    dev = [i for i, r in enumerate(records) if r['group'] == 'dev_gold']
    clean_anchor_tr = [i for i, r in enumerate(records) if r['split'] == 'train' and r['group'] == 'clean_anchor']
    clean_tight_tr = [i for i, r in enumerate(records) if r['split'] == 'train' and r['group'] == 'clean_tight_twin']
    clean_train = clean_anchor_tr + clean_tight_tr
    anchor_te = [i for i, r in enumerate(records) if r['split'] == 'test' and r['group'] == 'clean_anchor']
    tight = [i for i, r in enumerate(records) if r['split'] == 'test' and r['group'] == 'clean_tight_twin']
    if not dev or len({records[i]['speaker'] for i in dev}) < 2:
        raise ValueError('Need dev clips with at least two recording IDs')
    for i in dev:
        if records[i]['decision'] not in ('pass', 'reject') or records[i]['speaker'] == 'unknown' or records[i]['duration_s'] <= 0:
            raise ValueError(f'Invalid dev record: {records[i]["uid"]}')
    gold = [json.loads(line) for line in args.input_manifest.read_text().splitlines() if line.strip()]
    for row in gold:
        if sha256(Path(row['audio_path'])) != row['audio_sha256']:
            raise ValueError(f'Dev audio changed: {row["id"]}')
    report_path = args.probe_dir / 'phase3_probe_report.json'
    previous = json.loads(report_path.read_text())
    provenance = {str(p): sha256(p) for p in (args.anchors_manifest, args.defects_manifest, args.input_manifest, report_path, Path(__file__))}
    heads, twin_heads, anchor_te_heads, clean_heads = {}, {}, {}, {}
    oof_anchor_heads, oof_tight_heads = {}, {}
    head_specs = {}
    for enc in ('wavlm', 'phowhisper', 'gemma4_audio'):
        path = args.probe_dir / 'cache' / f'{enc}_pooled.npz'
        provenance[str(path)] = sha256(path)
        with np.load(path, allow_pickle=False) as cache:
            if list(cache['uids']) != [r['uid'] for r in records]:
                raise ValueError(f'Cache UID/order mismatch: {path}')
            feats = {k: cache[k] for k in ('mean', 'std', 'start_edge', 'end_edge', 'layer_dims')}
        for defect in DEFECT_TYPES:
            spec = previous['encoders'][enc]['defects'][defect]
            layer, pooling = spec['best_layer'], spec['primary_pooling']
            positive = [i for i, r in enumerate(records) if r['split'] == 'train' and r['group'] == 'defect' and r['defect_type'] == defect]
            train = clean_train + positive
            y = np.r_[np.zeros(len(clean_train)), np.ones(len(positive))]
            eval_indices = {'dev': dev, 'tight': tight, 'anchor_te': anchor_te, 'clean_train': clean_train}
            scores = fit_and_score_probe(build_feature_matrix(feats, train, layer, pooling, defect), y,
                {k: build_feature_matrix(feats, ids, layer, pooling, defect) for k, ids in eval_indices.items()})
            oof_a, oof_t = oof_clean_scores(feats, records, clean_anchor_tr, clean_tight_tr, positive, layer, pooling, defect)
            key = f'{enc}:{defect}'
            heads[key], twin_heads[key], anchor_te_heads[key], clean_heads[key] = (
                scores['dev'], scores['tight'], scores['anchor_te'], scores['clean_train']
            )
            oof_anchor_heads[key], oof_tight_heads[key] = oof_a, oof_t
            head_specs[key] = {'layer': layer, 'pooling': pooling, 'C': 0.01}
        del feats
        print(f'Fitted cached {enc} probe heads (full + 5-fold speaker OOF)', file=sys.stderr, flush=True)
    boundary = [f'wavlm:{d}' for d in DEFECT_TYPES[:2]]
    content = [f'gemma4_audio:{d}' for d in DEFECT_TYPES[2:]]
    configurations = {enc: [f'{enc}:{d}' for d in DEFECT_TYPES] for enc in ('wavlm', 'phowhisper', 'gemma4_audio')}
    configurations.update(composite=boundary + content, boundary_only=boundary, content_only=content,
                          composite_phowhisper_content=boundary + [f'phowhisper:{d}' for d in DEFECT_TYPES[2:]])
    for d in DEFECT_TYPES:
        configurations[f'composite_without_{d}'] = [k for k in boundary + content if k.split(':')[1] != d]
    risks = {name: np.max([heads[k] for k in keys], axis=0) for name, keys in configurations.items()}
    # Calibrated fusion variants (unsupervised empirical CDF on dev pool; OOF-calibrated tail boost)
    head_cdfs = {k: empirical_cdf(v) for k, v in heads.items()}
    risks['composite_quantile'] = np.max([head_cdfs[k] for k in boundary + content], axis=0)
    risks['composite_end_only_quantile'] = np.max([head_cdfs[k] for k in ['wavlm:clipped_word_end', *content]], axis=0)
    risks['phowhisper_end_only_quantile'] = np.max(
        [head_cdfs[k] for k in ['wavlm:clipped_word_end', 'phowhisper:secondary_speaker', 'phowhisper:background_music']],
        axis=0,
    )
    # Tail-gated fusion: content rank except when wavlm:clipped_word_end exceeds its speaker-OOF 99% tight-twin threshold
    oof_end_t99 = float(np.quantile(oof_tight_heads['wavlm:clipped_word_end'], 0.99))
    content_cdf_max = np.max([head_cdfs[k] for k in content], axis=0)
    risks['composite_oof_end_tail'] = np.where(heads['wavlm:clipped_word_end'] > oof_end_t99, 1.0 + heads['wavlm:clipped_word_end'], content_cdf_max)

    signal_path = args.probe_dir / 'cache' / 'signal_features.json'
    provenance[str(signal_path)] = sha256(signal_path)
    signals = json.loads(signal_path.read_text())
    signal_risks = []
    signal_z_risks = []
    for defect, (feature, higher_bad) in BASELINE_SPEC.items():
        reference = np.sort([signals[records[i]['uid']][feature] for i in clean_train])
        values = np.array([signals[records[i]['uid']][feature] for i in dev], dtype=float)
        percentile = np.searchsorted(reference, values, side='right') / len(reference)
        signal_risks.append(percentile if higher_bad else 1 - percentile)
        if defect != 'secondary_speaker':
            # Continuous standardized signal baseline (excluding chance-level window_rms_var)
            signed = values if higher_bad else -values
            med = float(np.median(signed))
            iqr = float(np.quantile(signed, 0.75) - np.quantile(signed, 0.25)) or 1.0
            signal_z_risks.append((signed - med) / iqr)
    risks['signal_only'] = np.max(signal_risks, axis=0)
    risks['signal_continuous_3feat'] = np.max(signal_z_risks, axis=0)
    durations = np.array([records[i]['duration_s'] for i in dev])
    labels = np.array([records[i]['decision'] == 'reject' for i in dev], dtype=int)
    recordings = np.array([records[i]['speaker'] for i in dev])
    risks['duration_control'] = -durations
    risks['recipe_nolock'] = np.zeros(len(dev))
    # Prior model verdicts are binary: report their actual attainable operating points.
    baseline_dir = args.input_manifest.parent / 'reports'
    for name, filename in [('flash_lite', 'gemini35_flash_lite_medium_vs_gemini38_medium.csv'), ('e2b_v3', 'e2b_v3_vs_gemini38_medium.csv')]:
        path = baseline_dir / filename
        if path.is_file():
            provenance[str(path)] = sha256(path)
            rows = {r['id']: r for r in csv.DictReader(path.open())}
            if all(records[i]['clip_id'] in rows and rows[records[i]['clip_id']]['model_decision'] in ('pass', 'reject') for i in dev):
                risks[name] = np.array([rows[records[i]['clip_id']]['model_decision'] == 'reject' for i in dev], dtype=float)
    curves = {k: curve(s, labels, durations) for k, s in risks.items()}
    operating = {k: matched_point(s, labels, durations, 0.70) for k, s in risks.items()}
    flash_lite_yield = curves['flash_lite'][0]['yield_ratio'] if 'flash_lite' in curves else None
    operating_at_flash_lite = (
        {k: matched_point(s, labels, durations, flash_lite_yield) for k, s in risks.items()}
        if flash_lite_yield is not None else {}
    )
    groups = sorted(set(recordings))
    bootstraps = {}
    for seed in map(int, args.bootstrap_seeds.split(',')):
        rng = np.random.default_rng(seed)
        rs = {k: [] for k in risks}
        delta = {k: [] for k in risks if k != 'composite'}
        delta_vs_end_quantile = {k: [] for k in risks if k != 'composite_end_only_quantile'}
        flash_lite_matched_delta = {k: [] for k in risks if k != 'flash_lite'}
        for _ in range(args.bootstrap_replicates):
            idx = np.concatenate([np.flatnonzero(recordings == g) for g in rng.choice(groups, len(groups), replace=True)])
            points = {k: matched_point(s[idx], labels[idx], durations[idx], 0.70) for k, s in risks.items()}
            for k, p in points.items():
                if p is not None:
                    rs[k].append(p['R'])
                if k != 'composite' and p is not None and points['composite'] is not None:
                    delta[k].append(p['R'] - points['composite']['R'])
                if k != 'composite_end_only_quantile' and p is not None and points['composite_end_only_quantile'] is not None:
                    delta_vs_end_quantile[k].append(p['R'] - points['composite_end_only_quantile']['R'])
            if 'flash_lite' in risks:
                fl_pts = curve(risks['flash_lite'][idx], labels[idx], durations[idx])
                fl_p = fl_pts[0]
                fl_target = fl_p['yield_ratio']
                for k, s in risks.items():
                    if k == 'flash_lite':
                        continue
                    mp = matched_point(s[idx], labels[idx], durations[idx], fl_target)
                    if mp is not None:
                        flash_lite_matched_delta[k].append(fl_p['R'] - mp['R'])
        def interval(values):
            return {'valid_replicates': len(values), 'ci95': np.quantile(values, [0.025, 0.975]).tolist() if values else None}
        bootstraps[str(seed)] = {
            'R': {k: interval(v) for k, v in rs.items()},
            'baseline_minus_composite_R': {k: interval(v) for k, v in delta.items()},
            'baseline_minus_end_only_quantile_R': {k: interval(v) for k, v in delta_vs_end_quantile.items()},
            'flash_lite_minus_gate_at_flash_lite_yield_R': {k: interval(v) for k, v in flash_lite_matched_delta.items()},
        }
    twin_controls = {}
    for name, keys in configurations.items():
        train_scores = np.max([clean_heads[k] for k in keys], axis=0)
        threshold_insample = float(np.quantile(train_scores, 0.95))
        oof_tight_scores = np.max([oof_tight_heads[k] for k in keys], axis=0)
        threshold_oof_tight = float(np.quantile(oof_tight_scores, 0.95))
        oof_pair_scores = np.maximum(
            np.max([oof_anchor_heads[k] for k in keys], axis=0),
            oof_tight_scores,
        )
        threshold_oof_pair = float(np.quantile(oof_pair_scores, 0.95))
        twin_scores = np.max([twin_heads[k] for k in keys], axis=0)
        anchor_scores = np.max([anchor_te_heads[k] for k in keys], axis=0)
        twin_controls[name] = {
            'threshold_from_synthetic_train_clean': threshold_insample,
            'n_tight_twins': len(tight),
            'tight_twin_FRR': float(np.mean(twin_scores > threshold_insample)),
            'raw_anchor_FRR_insample': float(np.mean(anchor_scores > threshold_insample)),
            'threshold_oof_tight_95pct': threshold_oof_tight,
            'tight_twin_FRR_oof_tight': float(np.mean(twin_scores > threshold_oof_tight)),
            'raw_anchor_FRR_oof_tight': float(np.mean(anchor_scores > threshold_oof_tight)),
            'threshold_oof_pair_95pct': threshold_oof_pair,
            'tight_twin_FRR_oof_pair': float(np.mean(twin_scores > threshold_oof_pair)),
            'raw_anchor_FRR_oof_pair': float(np.mean(anchor_scores > threshold_oof_pair)),
        }
    # Diagnostic audit of probability scale mismatch and marginal precision ceiling on dev-288
    comp_keys = configurations['composite']
    comp_thresh = operating['composite']['threshold'] if operating['composite'] else 1.0
    accepted_mask = risks['composite'] <= comp_thresh
    rejected_mask = ~accepted_mask
    rej_indices = np.flatnonzero(rejected_mask)
    acc_indices = np.flatnonzero(accepted_mask)
    rej_winners = [comp_keys[int(np.argmax([heads[k][j] for k in comp_keys]))] for j in rej_indices]
    winner_counts = {k: int(sum(w == k for w in rej_winners)) for k in comp_keys}

    dim_keys = ('word_end', 'word_start', 'speaker_purity', 'overlap', 'music', 'reverberation', 'sound_effect', 'noise', 'voice_damage')
    acc_bad_indices = np.flatnonzero(accepted_mask & (labels == 1))
    probed_dims = {'word_end', 'word_start', 'speaker_purity', 'overlap', 'music'}
    taxonomy = {'unprobed_only': 0, 'has_boundary_defect': 0, 'content_defect_only': 0}
    for j in acc_bad_indices:
        dims = {d for d, v in records[dev[j]]['dimensions'].items() if v == 'defective'}
        if not (dims & probed_dims):
            taxonomy['unprobed_only'] += 1
        elif 'word_end' in dims or 'word_start' in dims:
            taxonomy['has_boundary_defect'] += 1
        else:
            taxonomy['content_defect_only'] += 1

    word_end_acc_bad = [j for j in acc_bad_indices if records[dev[j]]['dimensions'].get('word_end') == 'defective']
    wavlm_end_pcts = [float(np.mean(heads['wavlm:clipped_word_end'] <= heads['wavlm:clipped_word_end'][j]) * 100.0) for j in word_end_acc_bad]

    # Marginal precision of content_only rejections vs top boundary candidates among accepted clips
    rej_sorted_by_content = sorted(rej_indices, key=lambda j: risks['content_only'][j])
    acc_sorted_by_wavlm_end = sorted(acc_indices, key=lambda j: heads['wavlm:clipped_word_end'][j], reverse=True)
    acc_sorted_by_phowhisper_end = sorted(acc_indices, key=lambda j: heads['phowhisper:clipped_word_end'][j], reverse=True)
    acc_sorted_by_gemma_start = sorted(acc_indices, key=lambda j: heads['gemma4_audio:clipped_word_start'][j], reverse=True)
    marginal_vs_tail = {}
    for k_top in (5, 10, 15, 20):
        marginal_vs_tail[f'k={k_top}'] = {
            'marginal_content_reject_bad_rate': float(np.mean(labels[rej_sorted_by_content[:k_top]])),
            'accepted_top_wavlm_end_bad_rate': float(np.mean(labels[acc_sorted_by_wavlm_end[:k_top]])),
            'accepted_top_phowhisper_end_bad_rate': float(np.mean(labels[acc_sorted_by_phowhisper_end[:k_top]])),
            'accepted_top_gemma_start_bad_rate': float(np.mean(labels[acc_sorted_by_gemma_start[:k_top]])),
        }

    head_scale_summary = {}
    for k, vals in heads.items():
        q10, q50, q70, q90 = np.quantile(vals, [0.10, 0.50, 0.70, 0.90]).tolist()
        y_we = np.array([records[i]['dimensions'].get('word_end') == 'defective' for i in dev], dtype=int)
        y_ws = np.array([records[i]['dimensions'].get('word_start') == 'defective' for i in dev], dtype=int)
        head_scale_summary[k] = {
            'q10': float(q10), 'q50': float(q50), 'q70': float(q70), 'q90': float(q90), 'max': float(np.max(vals)),
            'overall_reject_auroc': auroc(vals[labels == 1], vals[labels == 0]),
            'word_end_auroc': auroc(vals[y_we == 1], vals[y_we == 0]),
            'word_start_auroc': auroc(vals[y_ws == 1], vals[y_ws == 0]),
        }

    calibration_diagnostics = {
        'composite_70pct_threshold': float(comp_thresh),
        'n_rejected_at_70pct': int(len(rej_indices)),
        'rejected_argmax_head_counts': winner_counts,
        'accepted_reject_taxonomy_n56': taxonomy,
        'accepted_word_end_reject_n': len(word_end_acc_bad),
        'accepted_word_end_wavlm_dev_percentiles': {
            'median': float(np.median(wavlm_end_pcts)) if wavlm_end_pcts else None,
            'mean': float(np.mean(wavlm_end_pcts)) if wavlm_end_pcts else None,
            'count_ge_p80': int(sum(p >= 80.0 for p in wavlm_end_pcts)),
            'count_ge_p90': int(sum(p >= 90.0 for p in wavlm_end_pcts)),
        },
        'marginal_content_vs_accepted_boundary_tail': marginal_vs_tail,
        'head_scale_and_auroc_summary': head_scale_summary,
    }

    result = dict(command=[sys.executable, *sys.argv], n_clips=len(dev), n_recordings=len(groups), total_minutes=float(durations.sum()/60),
        label_source='Gemini dev diagnostic only; no G1 evaluation', primary_target_minute_yield=0.70,
        flash_lite_minute_yield=flash_lite_yield,
        input_sha256=provenance, head_specs=head_specs, configurations=configurations,
        operating_points=operating, operating_points_at_flash_lite_yield=operating_at_flash_lite,
        curves=curves, bootstrap=bootstraps, clean_twin_controls=twin_controls,
        calibration_diagnostics=calibration_diagnostics,
        per_recording={g: {k: matched_point(s[recordings == g], labels[recordings == g], durations[recordings == g], 0.70) for k, s in risks.items()} for g in groups},
        limitations=['Prior yield used clip quantiles; this report corrects to minute yield.',
                    'Cached feature provenance has UID matching and current hashes, but no original audio/model hash binding.',
                    'Gemini 3.8 is the dev label source, so evaluating it here as a baseline is circular and omitted.',
                    'Binary/all-pass baselines cannot generally match 70% yield; their full tied-score curves are retained and flash_lite is additionally compared at its native 83.65% accepted-minute yield.',
                    'Bootstrap reranks within each sampled recording pool; intervals describe diagnostic ranking, not a frozen deployment threshold.',
                    'Synthetic tight twins are reused mechanism diagnostics; speaker-disjoint 5-fold OOF thresholds correct in-sample threshold overfitting.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'composite_probe_gate_report.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output_dir / 'dev_probe_scores.jsonl').open('w') as handle:
        for j, i in enumerate(dev):
            handle.write(json.dumps(dict(uid=records[i]['uid'], recording_id=str(recordings[j]), duration_s=float(durations[j]), reject_label=int(labels[j]), heads={k:float(v[j]) for k,v in heads.items()}, risks={k:float(v[j]) for k,v in risks.items()})) + '\n')
    accepted = risks['composite'] <= operating['composite']['threshold'] if operating['composite'] else np.zeros(len(dev), dtype=bool)
    failure_rows = []
    for j in np.flatnonzero(accepted & (labels == 1))[:20]:
        failure_rows.append(dict(uid=records[dev[j]]['uid'], recording_id=str(recordings[j]),
            audio_path=records[dev[j]]['audio_path'], dimensions=records[dev[j]]['dimensions'],
            heads={k: float(v[j]) for k, v in heads.items()}))
    (args.output_dir / 'accepted_defect_diagnostics.json').write_text(json.dumps(dict(
        selection='First up to 20 accepted rejects in manifest order; Gemini-labelled diagnostic errors, no human verification',
        total_accepted_rejects=int(np.sum(accepted & (labels == 1))), rows=failure_rows), indent=2) + '\n')
    lines = ['# Composite probe gate & calibration diagnostic: real-dev (288 clips)', '', f'{len(dev)} clips, {len(groups)} recordings, {durations.sum()/60:.2f} minutes. Gemini labels; no human-test claim.', '',
             '## 1. Operating points at 70% accepted-minute yield', '',
             'Primary: R = bad accepted / accepted clips at 70% accepted-minute yield. Whole tied-score groups; ±10% relative yield tolerance. Unmatched baselines are shown as unavailable.', '',
             '| Gate | R | Actual minute yield | R 95% CI (seed 42) | Baseline − composite R CI | Baseline − end_only_quantile R CI |', '|---|---:|---:|---|---|---|']
    first = bootstraps[next(iter(bootstraps))]
    for k, p in operating.items():
        def format_ci(value):
            return 'unavailable' if value is None else f'[{value[0]:.4f}, {value[1]:.4f}]'
        r_value = f"{p['R']:.4f}" if p else 'unmatched'
        yield_value = f"{p['yield_ratio']:.4f}" if p else 'unmatched'
        r_ci = format_ci(first['R'][k]['ci95']) if p else 'unavailable'
        delta_ci = format_ci(first['baseline_minus_composite_R'].get(k, {}).get('ci95')) if p else 'unavailable'
        delta_eq_ci = format_ci(first['baseline_minus_end_only_quantile_R'].get(k, {}).get('ci95')) if p else 'unavailable'
        lines.append(f"| {k} | {r_value} | {yield_value} | {r_ci} | {delta_ci} | {delta_eq_ci} |")
    if flash_lite_yield is not None:
        lines += ['', f'## 2. Comparison against binary `flash_lite` at its native accepted-minute yield ({flash_lite_yield:.4f})', '',
                  '| Gate | R at ~83.7% yield | Actual minute yield | Flash-Lite − Gate R 95% CI (seed 42) | Valid replicates |', '|---|---:|---:|---|---:|']
        for k in ('flash_lite', 'phowhisper', 'wavlm', 'gemma4_audio', 'composite', 'composite_end_only_quantile', 'phowhisper_end_only_quantile', 'signal_continuous_3feat', 'duration_control'):
            p = operating_at_flash_lite.get(k)
            fl_ci_obj = first['flash_lite_minus_gate_at_flash_lite_yield_R'].get(k, {})
            fl_ci = format_ci(fl_ci_obj.get('ci95'))
            v_rep = fl_ci_obj.get('valid_replicates', 0)
            lines.append(f"| {k} | {p['R']:.4f} | {p['yield_ratio']:.4f} | {fl_ci} | {v_rep} |" if p else f"| {k} | unmatched | unmatched | unavailable | 0 |")
    lines += ['', '## 3. Clean tight-twin controls (n=90 test tight twins, n=90 test raw anchors)', '',
              '| Configuration | In-sample mix 95% thresh | Tight-twin FRR (in-sample) | Speaker-OOF tight 95% thresh | Tight-twin FRR (OOF tight) | Raw-anchor FRR (OOF tight) | Speaker-OOF pair 95% thresh | Tight-twin FRR (OOF pair) | Raw-anchor FRR (OOF pair) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k, v in twin_controls.items():
        lines.append(
            f"| {k} | {v['threshold_from_synthetic_train_clean']:.4f} | {v['tight_twin_FRR']:.1%} | "
            f"{v['threshold_oof_tight_95pct']:.4f} | {v['tight_twin_FRR_oof_tight']:.1%} | {v['raw_anchor_FRR_oof_tight']:.1%} | "
            f"{v['threshold_oof_pair_95pct']:.4f} | {v['tight_twin_FRR_oof_pair']:.1%} | {v['raw_anchor_FRR_oof_pair']:.1%} |"
        )
    lines += [
        '', '## 4. Probability-scale mismatch & fixed-yield marginal precision audit', '',
        f"- At 70% accepted-minute yield (`threshold = {comp_thresh:.6f}`), `composite` rejects {len(rej_indices)} clips: "
        f"`gemma4_audio:background_music` wins argmax on {winner_counts['gemma4_audio:background_music']}/{len(rej_indices)}, "
        f"`gemma4_audio:secondary_speaker` on {winner_counts['gemma4_audio:secondary_speaker']}/{len(rej_indices)}, "
        f"`wavlm:clipped_word_end` on {winner_counts['wavlm:clipped_word_end']}/{len(rej_indices)}, and "
        f"`wavlm:clipped_word_start` on {winner_counts['wavlm:clipped_word_start']}/{len(rej_indices)}.",
        f"- Taxonomy of the {len(acc_bad_indices)} accepted rejects: {taxonomy['content_defect_only']} content-only (`music`/`speaker_purity`/`overlap`), "
        f"{taxonomy['has_boundary_defect']} with boundary defects (`word_end`/`word_start`), and {taxonomy['unprobed_only']} with unprobed defects only (`reverberation`/`sound_effect`/`noise`).",
        f"- Among the {len(word_end_acc_bad)} accepted `word_end` rejects, `wavlm:clipped_word_end` ranks them at median dev percentile "
        f"{calibration_diagnostics['accepted_word_end_wavlm_dev_percentiles']['median']:.1f}% "
        f"({calibration_diagnostics['accepted_word_end_wavlm_dev_percentiles']['count_ge_p80']}/{len(word_end_acc_bad)} >= P80, "
        f"{calibration_diagnostics['accepted_word_end_wavlm_dev_percentiles']['count_ge_p90']}/{len(word_end_acc_bad)} >= P90), "
        f"but its maximum dev probability (`{head_scale_summary['wavlm:clipped_word_end']['max']:.4f}`) is below `{comp_thresh:.4f}`.",
        '- Fixed-70%-yield marginal precision ceiling: the lowest 5 / 10 / 15 / 20 clips rejected by `content_only` have defect rates '
        f"{marginal_vs_tail['k=5']['marginal_content_reject_bad_rate']:.1%} / {marginal_vs_tail['k=10']['marginal_content_reject_bad_rate']:.1%} / "
        f"{marginal_vs_tail['k=15']['marginal_content_reject_bad_rate']:.1%} / {marginal_vs_tail['k=20']['marginal_content_reject_bad_rate']:.1%}, "
        'while the top 5 / 10 / 15 / 20 `wavlm:clipped_word_end` clips among accepted clips have defect rates '
        f"{marginal_vs_tail['k=5']['accepted_top_wavlm_end_bad_rate']:.1%} / {marginal_vs_tail['k=10']['accepted_top_wavlm_end_bad_rate']:.1%} / "
        f"{marginal_vs_tail['k=15']['accepted_top_wavlm_end_bad_rate']:.1%} / {marginal_vs_tail['k=20']['accepted_top_wavlm_end_bad_rate']:.1%} "
        f"(and `gemma4_audio:clipped_word_start` has {marginal_vs_tail['k=5']['accepted_top_gemma_start_bad_rate']:.1%} / {marginal_vs_tail['k=10']['accepted_top_gemma_start_bad_rate']:.1%}).",
    ]
    lines += ['', '## 5. Limitations', ''] + [f'- {s}' for s in result['limitations']]
    (args.output_dir / 'COMPOSITE_PROBE_GATE_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(args.output_dir / 'COMPOSITE_PROBE_GATE_REPORT.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
