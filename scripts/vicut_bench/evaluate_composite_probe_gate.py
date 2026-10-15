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
    clean_train = [i for i, r in enumerate(records) if r['split'] == 'train' and r['group'].startswith('clean_')]
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
    heads, twin_heads, clean_heads = {}, {}, {}
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
            eval_indices = {'dev': dev, 'tight': tight, 'clean_train': clean_train}
            scores = fit_and_score_probe(build_feature_matrix(feats, train, layer, pooling, defect), y,
                {k: build_feature_matrix(feats, ids, layer, pooling, defect) for k, ids in eval_indices.items()})
            key = f'{enc}:{defect}'
            heads[key], twin_heads[key], clean_heads[key] = scores['dev'], scores['tight'], scores['clean_train']
            head_specs[key] = {'layer': layer, 'pooling': pooling, 'C': 0.01}
        del feats
        print(f'Fitted cached {enc} probe heads', file=sys.stderr, flush=True)
    boundary = [f'wavlm:{d}' for d in DEFECT_TYPES[:2]]
    content = [f'gemma4_audio:{d}' for d in DEFECT_TYPES[2:]]
    configurations = {enc: [f'{enc}:{d}' for d in DEFECT_TYPES] for enc in ('wavlm', 'phowhisper', 'gemma4_audio')}
    configurations.update(composite=boundary + content, boundary_only=boundary, content_only=content,
                          composite_phowhisper_content=boundary + [f'phowhisper:{d}' for d in DEFECT_TYPES[2:]])
    for d in DEFECT_TYPES:
        configurations[f'composite_without_{d}'] = [k for k in boundary + content if k.split(':')[1] != d]
    risks = {name: np.max([heads[k] for k in keys], axis=0) for name, keys in configurations.items()}
    signal_path = args.probe_dir / 'cache' / 'signal_features.json'
    provenance[str(signal_path)] = sha256(signal_path)
    signals = json.loads(signal_path.read_text())
    signal_risks = []
    for defect, (feature, higher_bad) in BASELINE_SPEC.items():
        reference = np.sort([signals[records[i]['uid']][feature] for i in clean_train])
        values = np.array([signals[records[i]['uid']][feature] for i in dev])
        percentile = np.searchsorted(reference, values, side='right') / len(reference)
        signal_risks.append(percentile if higher_bad else 1 - percentile)
    risks['signal_only'] = np.max(signal_risks, axis=0)
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
    groups = sorted(set(recordings))
    bootstraps = {}
    for seed in map(int, args.bootstrap_seeds.split(',')):
        rng = np.random.default_rng(seed)
        rs = {k: [] for k in risks}
        delta = {k: [] for k in risks if k != 'composite'}
        for _ in range(args.bootstrap_replicates):
            idx = np.concatenate([np.flatnonzero(recordings == g) for g in rng.choice(groups, len(groups), replace=True)])
            points = {k: matched_point(s[idx], labels[idx], durations[idx], 0.70) for k, s in risks.items()}
            for k, p in points.items():
                if p is not None:
                    rs[k].append(p['R'])
                if k != 'composite' and p is not None and points['composite'] is not None:
                    delta[k].append(p['R'] - points['composite']['R'])
        def interval(values):
            return {'valid_replicates': len(values), 'ci95': np.quantile(values, [0.025, 0.975]).tolist() if values else None}
        bootstraps[str(seed)] = {'R': {k: interval(v) for k, v in rs.items()}, 'baseline_minus_composite_R': {k: interval(v) for k, v in delta.items()}}
    twin_controls = {}
    for name, keys in configurations.items():
        train_scores = np.max([clean_heads[k] for k in keys], axis=0)
        threshold = float(np.quantile(train_scores, 0.95))
        twin_scores = np.max([twin_heads[k] for k in keys], axis=0)
        twin_controls[name] = {'threshold_from_synthetic_train_clean': threshold, 'n_tight_twins': len(tight), 'tight_twin_FRR': float(np.mean(twin_scores > threshold))}
    result = dict(command=[sys.executable, *sys.argv], n_clips=len(dev), n_recordings=len(groups), total_minutes=float(durations.sum()/60),
        label_source='Gemini dev diagnostic only; no G1 evaluation', primary_target_minute_yield=0.70,
        input_sha256=provenance, head_specs=head_specs, configurations=configurations,
        operating_points=operating, curves=curves, bootstrap=bootstraps, clean_twin_controls=twin_controls,
        per_recording={g: {k: matched_point(s[recordings == g], labels[recordings == g], durations[recordings == g], 0.70) for k, s in risks.items()} for g in groups},
        limitations=['Prior yield used clip quantiles; this report corrects to minute yield.',
                    'Cached feature provenance has UID matching and current hashes, but no original audio/model hash binding.',
                    'Gemini 3.8 is the dev label source, so evaluating it here as a baseline is circular and omitted.',
                    'Binary/all-pass baselines cannot generally match 70% yield; their full tied-score curves are retained. Bootstrap valid-replicate counts must be checked; conditional intervals for unmatched point estimates are not comparisons.',
                    'Bootstrap reranks within each sampled recording pool; intervals describe diagnostic ranking, not a frozen deployment threshold.',
                    'Synthetic tight twins are reused mechanism diagnostics; no new held-out claim.'])
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
    lines = ['# Composite probe gate: real-dev diagnostic', '', f'{len(dev)} clips, {len(groups)} recordings, {durations.sum()/60:.2f} minutes. Gemini labels; no human-test claim.', '',
             'Primary: R = bad accepted / accepted clips at 70% accepted-minute yield. Whole tied-score groups; ±10% relative yield tolerance. Unmatched baselines are shown as unavailable.', '',
             '| Gate | R | Actual minute yield | R 95% CI (first seed) | Baseline − composite R CI |', '|---|---:|---:|---|---|']
    first = bootstraps[next(iter(bootstraps))]
    for k, p in operating.items():
        def format_ci(value):
            return 'unavailable' if value is None else f'[{value[0]:.4f}, {value[1]:.4f}]'
        r_value = f"{p['R']:.4f}" if p else 'unmatched'
        yield_value = f"{p['yield_ratio']:.4f}" if p else 'unmatched'
        r_ci = format_ci(first['R'][k]['ci95']) if p else 'unavailable'
        delta_ci = format_ci(first['baseline_minus_composite_R'].get(k, {}).get('ci95')) if p else 'unavailable'
        lines.append(f"| {k} | {r_value} | {yield_value} | {r_ci} | {delta_ci} |")
    lines += ['', 'Clean tight-twin controls (thresholds set on synthetic train clean only):', '']
    lines += [f"- {k}: {v['tight_twin_FRR']:.1%} FRR, n={len(tight)}." for k,v in twin_controls.items()]
    lines += ['', 'Limitations:', ''] + [f'- {s}' for s in result['limitations']]
    (args.output_dir / 'COMPOSITE_PROBE_GATE_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(args.output_dir / 'COMPOSITE_PROBE_GATE_REPORT.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
