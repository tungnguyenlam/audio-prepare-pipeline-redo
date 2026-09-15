"""Compare saved verifier runs against an explicitly selected reference.

Each input is one run: either a flat clip directory or a tree of audio families.
No model is loaded. Metrics describe agreement with the reference, not ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
from statistics import median
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common.files import ROOT, persist_path, progress, read_json, resolve_stored_path, write_json
from _verdicts import FAILURE_CODES, _known_prompts, _validate_verdict

BACKEND_SUFFIX = re.compile(r"_(gemini|hf|vllm|unsloth|endpoint|minicpm|kimi|moss|vibevoice)$")
SKIP_DIRS = {"comparisons", "work", "plot", "plots", "experiments", "__pycache__"}
PAIR_FIELDS = (
    "candidate", "family", "key", "status", "ref_decision", "cand_decision",
    "ref_codes", "cand_codes", "missed_codes", "overcalled_codes",
    "ref_reason", "cand_reason", "ref_schema_profile", "cand_schema_profile",
    "transcript_status", "ref_transcript", "cand_transcript",
    "emotion_status", "ref_emotion", "cand_emotion",
    "reference_file", "candidate_file", "audio_path",
)


def load_run(directory: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Read one run, retaining invalid artifacts for coverage accounting."""
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    records = {}
    ignored = 0
    known_prompts = _known_prompts()

    def walk_error(error: OSError) -> None:
        raise error

    for root, dirs, files in os.walk(directory, onerror=walk_error):
        # Prune children only: .data in the selected root must never hide inputs.
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        parent = Path(root)
        family = parent.relative_to(directory).as_posix()
        for name in sorted(files):
            path = parent / name
            if path.suffix.lower() != '.json' or name.startswith('.'):
                continue
            error = None
            try:
                data = read_json(path)
            except (OSError, ValueError):
                data = {}
                error = 'invalid_json'
            if error is None and not (
                data.get('operation') == 'verify' or 'verdict' in data or 'decision' in data
                or ('status' in data and 'source' in data)
            ):
                ignored += 1
                continue
            verdict = data.get('verdict', data)
            parameters = data.get('parameters')
            parameters = parameters if isinstance(parameters, dict) else {}
            profile = 'unknown'
            if error is None:
                if data.get('status') not in (None, 'success'):
                    error = 'verifier_failed'
                else:
                    try:
                        profile, error = _validate_verdict(
                            verdict, parameters.get('prompt'), str(data.get('model', '')), known_prompts
                        )
                    except (TypeError, ValueError):
                        error = 'invalid_schema'
            verdict = verdict if isinstance(verdict, dict) else {}
            source = data.get('source')
            source = source if isinstance(source, dict) else {}
            transcript = verdict.get('transcript')
            transcript = transcript if isinstance(transcript, str) else ''
            emotion = verdict.get('emotion')
            emotion = emotion.strip() if isinstance(emotion, str) else ''
            key = BACKEND_SUFFIX.sub('', path.stem)
            identity = (family, key)
            if identity in records:
                raise ValueError(
                    f"Duplicate clip {family}/{key}: {records[identity]['file']} and {path}. "
                    'Select one backend/model/effort per input directory.'
                )
            raw_codes = verdict.get('failure_codes')
            codes = {c for c in raw_codes if isinstance(c, str) and c in FAILURE_CODES} if isinstance(raw_codes, list) else set()
            for field in ('speaker_purity', 'word_completeness', 'audio_quality'):
                value = verdict.get(field)
                if isinstance(value, str) and value in FAILURE_CODES:
                    codes.add(value)
            for boundary, code in (('boundary_start', 'clipped_word_start'), ('boundary_end', 'clipped_word_end')):
                if verdict.get(boundary) == 'clipped':
                    codes.add(code)
            records[identity] = {
                'file': str(path), 'error': error,
                'decision': verdict.get('decision') if error is None else None,
                'codes': sorted(codes), 'reason': str(verdict.get('reason') or ''),
                'schema_profile': profile, 'transcript': transcript, 'emotion': emotion,
                'audio_path': persist_path(source['path']) if source.get('path') else (verdict.get('audio_path') or ''),
                'sha256': source.get('sha256'), 'latency_s': verdict.get('_latency_s'),
            }
    if not records:
        raise ValueError(f'No verifier artifacts in {directory} (ignored {ignored} unrelated JSON files)')
    return records, {
        'directory': persist_path(directory), 'artifacts': len(records),
        'valid': sum(r['error'] is None for r in records.values()),
        'invalid': sum(r['error'] is not None for r in records.values()),
        'ignored_json': ignored,
        'errors': dict(Counter(r['error'] for r in records.values() if r['error'])),
    }


def normalize_families(records: dict, directory: Path, flat_family: str | None = None) -> dict:
    """A flat directory names its family; two flat inputs share the reference name."""
    normalized = {}
    for (family, key), value in records.items():
        family = (flat_family or directory.name) if family == '.' else family
        identity = (family, key)
        if identity in normalized:
            raise ValueError(f'Ambiguous flat/nested family key {family}/{key} in {directory}')
        normalized[identity] = value
    return normalized


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize(rows: list[dict], candidate: str, family: str) -> dict:
    matched = [r for r in rows if r['status'] in {'agree', 'bad_accept', 'false_reject', 'code_mismatch'}]
    counts = Counter(r['status'] for r in rows)
    tp = sum(r['ref_decision'] == r['cand_decision'] == 'reject' for r in matched)
    tn = sum(r['ref_decision'] == r['cand_decision'] == 'pass' for r in matched)
    fn, fp = counts['bad_accept'], counts['false_reject']
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    latencies = []
    for row in matched:
        try:
            latency = float(row['_latency_s'])
        except (TypeError, ValueError):
            continue
        if math.isfinite(latency) and latency >= 0:
            latencies.append(latency)
    latencies.sort()
    criteria = {}
    for code in FAILURE_CODES:
        ref_count = sum(code in r['ref_codes'].split(';') for r in matched)
        caught = sum(code in r['ref_codes'].split(';') and code in r['cand_codes'].split(';') and r['cand_decision'] == 'reject' for r in matched)
        overcalled = sum(code in r['cand_codes'].split(';') and code not in r['ref_codes'].split(';') for r in matched)
        criteria[code] = {
            'ref_count': ref_count, 'caught': caught, 'missed': ref_count - caught,
            'overcalled': overcalled, 'recall': ratio(caught, ref_count),
        }
    ref_valid = sum(r['ref_decision'] in ('pass', 'reject') for r in rows)
    transcript_counts = Counter(r['transcript_status'] for r in rows)
    reference_transcripts = sum(r['transcript_status'] != 'not_applicable' for r in rows)
    candidate_transcripts = sum(bool(r['cand_transcript']) for r in rows)
    covered_reference_transcripts = transcript_counts['exact_match'] + transcript_counts['different']
    comparable_transcripts = transcript_counts['exact_match'] + transcript_counts['different']

    emotion_counts = Counter(r['emotion_status'] for r in rows)
    reference_emotions = sum(r['emotion_status'] != 'not_applicable' for r in rows)
    candidate_emotions = sum(bool(r['cand_emotion']) for r in rows)
    covered_reference_emotions = emotion_counts['exact_match'] + emotion_counts['different']
    comparable_emotions = emotion_counts['exact_match'] + emotion_counts['different']

    return {
        'candidate': candidate, 'family': family, 'matched_clips': len(matched),
        'reference_valid': ref_valid, 'coverage': ratio(len(matched), ref_valid),
        'counts': dict(counts),
        'overall': {
            'accuracy': ratio(tp + tn, len(matched)), 'defect_recall': recall,
            'false_rejection_rate': ratio(fp, fp + tn), 'reject_precision': precision,
            'reject_f1': ratio(2 * tp, 2 * tp + fp + fn),
        },
        'confusion_matrix': {
            'true_rejects_caught': tp, 'bad_accepts_missed': fn,
            'true_passes_kept': tn, 'false_rejects_dropped': fp,
        },
        'latency': {
            'samples': len(latencies),
            'p50_s': round(median(latencies), 3) if latencies else None,
            'p95_s': round(latencies[math.ceil(len(latencies) * .95) - 1], 3) if latencies else None,
        },
        'transcripts': {
            'reference_available': reference_transcripts,
            'candidate_available': candidate_transcripts,
            'candidate_on_reference': covered_reference_transcripts,
            'coverage': ratio(covered_reference_transcripts, reference_transcripts),
            'comparable': comparable_transcripts,
            'exact_matches': transcript_counts['exact_match'],
            'exact_match_rate': ratio(transcript_counts['exact_match'], comparable_transcripts),
            'statuses': dict(transcript_counts),
        },
        'emotions': {
            'reference_available': reference_emotions,
            'candidate_available': candidate_emotions,
            'candidate_on_reference': covered_reference_emotions,
            'coverage': ratio(covered_reference_emotions, reference_emotions),
            'comparable': comparable_emotions,
            'exact_matches': emotion_counts['exact_match'],
            'exact_match_rate': ratio(emotion_counts['exact_match'], comparable_emotions),
            'statuses': dict(emotion_counts),
        },
        'per_criterion': criteria,
    }


def compare_records(reference: dict, candidate: dict, label: str) -> list[dict]:
    rows = []
    for (family, key), ref in sorted(reference.items()):
        cand = candidate.get((family, key))
        if ref['error']:
            status = 'invalid_reference'
        elif cand is None:
            status = 'missing_candidate'
        elif cand['error']:
            status = 'invalid_candidate'
        elif ref['sha256'] and cand['sha256'] and ref['sha256'] != cand['sha256']:
            status = 'audio_mismatch'
        elif ref['decision'] != cand['decision']:
            status = 'bad_accept' if cand['decision'] == 'pass' else 'false_reject'
        elif ref['codes'] != cand['codes']:
            status = 'code_mismatch'
        else:
            status = 'agree'
        cand = cand or {}
        ref_codes, cand_codes = set(ref['codes']), set(cand.get('codes', []))
        ref_transcript = ref.get('transcript', '')
        cand_transcript = cand.get('transcript', '')
        if ref.get('decision') != 'pass' or not ref_transcript:
            transcript_status = 'not_applicable'
        elif cand.get('decision') != 'pass' or not cand_transcript:
            transcript_status = 'missing_candidate'
        elif ref_transcript == cand_transcript:
            transcript_status = 'exact_match'
        else:
            transcript_status = 'different'

        ref_emotion = ref.get('emotion', '')
        cand_emotion = cand.get('emotion', '')
        if not ref_emotion:
            emotion_status = 'not_applicable'
        elif not cand_emotion:
            emotion_status = 'missing_candidate'
        elif ref_emotion.lower() == cand_emotion.lower():
            emotion_status = 'exact_match'
        else:
            emotion_status = 'different'

        rows.append({
            'candidate': label, 'family': family, 'key': key, 'status': status,
            'ref_decision': ref['decision'], 'cand_decision': cand.get('decision'),
            'ref_codes': ';'.join(sorted(ref_codes)), 'cand_codes': ';'.join(sorted(cand_codes)),
            'missed_codes': ';'.join(sorted(ref_codes if status == 'bad_accept' else ref_codes - cand_codes)),
            'overcalled_codes': ';'.join(sorted(cand_codes - ref_codes)),
            'ref_reason': ref['error'] or ref['reason'],
            'cand_reason': cand.get('error') or cand.get('reason', ''),
            'ref_schema_profile': ref.get('schema_profile', ''),
            'cand_schema_profile': cand.get('schema_profile', ''),
            'transcript_status': transcript_status,
            'ref_transcript': ref_transcript,
            'cand_transcript': cand_transcript,
            'emotion_status': emotion_status,
            'ref_emotion': ref_emotion,
            'cand_emotion': cand_emotion,
            'reference_file': ref['file'], 'candidate_file': cand.get('file', ''),
            'audio_path': ref['audio_path'] or cand.get('audio_path', ''),
            '_latency_s': cand.get('latency_s'),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def percent(value: float | None) -> str:
    return 'N/A' if value is None else f'{value * 100:.1f}%'


def markdown_cell(value: Any) -> str:
    return str(value or '').replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _markdown_relpath(path_str: str, base_dir: Path) -> str:
    if not path_str:
        return ""
    try:
        p = Path(path_str)
        base = base_dir.resolve()
        if p.is_absolute():
            target_p = p.resolve() if p.exists() else None
            if target_p is None:
                for anchor in (".data", base.name):
                    if anchor in p.parts:
                        idx = p.parts.index(anchor)
                        subpath = Path(*p.parts[idx:])
                        curr = base
                        while curr != curr.parent:
                            if (curr / subpath).exists() or (curr / anchor).exists():
                                target_p = (curr / subpath).resolve()
                                break
                            curr = curr.parent
                        if target_p:
                            break
            if target_p is None:
                target_p = p
            try:
                rel = os.path.relpath(target_p, base)
            except ValueError:
                rel = str(target_p)
        else:
            target_p = resolve_stored_path(p)
            try:
                rel = os.path.relpath(target_p, base)
            except ValueError:
                rel = persist_path(target_p)
        return Path(rel).as_posix()
    except Exception:
        return path_str


def markdown_file_link(path_str: str | None, base_dir: Path, label: str) -> str:
    if not path_str:
        return ""
    target = _markdown_relpath(path_str, base_dir)
    if not target:
        return markdown_cell(label)
    if any(c in target for c in (" ", "(", ")")):
        return f"[{markdown_cell(label)}](<{target}>)"
    return f"[{markdown_cell(label)}]({target})"


def render_plots(summaries: list[dict], directory: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        progress('WARN', 'matplotlib unavailable; CSV/JSON/Markdown reports are still generated')
        return []
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, summary in enumerate(summaries, 1):
        fig, ax = plt.subplots(figsize=(10, 5))
        caught = [summary['per_criterion'][c]['caught'] for c in FAILURE_CODES]
        missed = [summary['per_criterion'][c]['missed'] for c in FAILURE_CODES]
        ax.barh(FAILURE_CODES, caught, label='Caught', color='#22c55e')
        ax.barh(FAILURE_CODES, missed, left=caught, label='Missed', color='#ef4444')
        ax.set_title(f"{summary['candidate']} — {summary['matched_clips']} matched clips")
        ax.set_xlabel('Reference defects on matched valid clips (zero means no observations)')
        ax.legend()
        fig.tight_layout()
        path = directory / f'candidate_{index}_defects.png'
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
        transcripts = summary['transcripts']
        if transcripts['reference_available']:
            labels = ('exact_match', 'different', 'missing_candidate')
            values = [transcripts['statuses'].get(label, 0) for label in labels]
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(labels, values, color=('#22c55e', '#f59e0b', '#ef4444'))
            ax.bar_label(bars)
            ax.set_title(f"{summary['candidate']} — transcript comparison")
            ax.set_ylabel('Reference transcripts')
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            fig.tight_layout()
            path = directory / f'candidate_{index}_transcripts.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
        emotions = summary.get('emotions', {})
        if emotions.get('reference_available'):
            labels = ('exact_match', 'different', 'missing_candidate')
            values = [emotions['statuses'].get(label, 0) for label in labels]
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(labels, values, color=('#22c55e', '#f59e0b', '#ef4444'))
            ax.bar_label(bars)
            ax.set_title(f"{summary['candidate']} — emotion agreement")
            ax.set_ylabel('Reference emotions')
            ax.grid(True, axis='y', linestyle='--', alpha=0.4)
            fig.tight_layout()
            path = directory / f'candidate_{index}_emotions.png'
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
        '  compare.sh --reference-dir .data/verdicts/medium/video --candidates-dir .data/verdicts/low\n'
        '  compare.sh --reference-dir .data/verdicts/medium --candidates-dir .data/verdicts/low '
        '--candidates-dir .data/verdicts/hf\n\n'
        'Choose one backend/model/effort per input. Both flat folders and family trees are read recursively.\n'
        'The old implicit Gemini reference, positional folder, and --base-dir discovery have been removed.',
    )
    parser.add_argument('--reference-dir', type=Path, required=True, help='Reference run or one audio-family directory')
    parser.add_argument('--candidates-dir', '--candidate-dir', dest='candidates_dir', type=Path, action='append', required=True, help='Candidate run or family; repeat for multiple candidates')
    parser.add_argument('--output-dir', type=Path, help='Exact report destination; default: .data/s4-agent/verifier/comparisons/<timestamp>-<selection-hash>')
    parser.add_argument('--title', help='Report title')
    parser.add_argument('--no-plots', action='store_true', help='Write tables and reports without matplotlib')
    args = parser.parse_args()
    reference_dir = args.reference_dir.expanduser().resolve()
    candidate_dirs = [p.expanduser().resolve() for p in args.candidates_dir]
    if len(set(candidate_dirs)) != len(candidate_dirs):
        parser.error('A candidate directory was supplied more than once')
    for directory in candidate_dirs:
        if directory == reference_dir or directory in reference_dir.parents or reference_dir in directory.parents:
            parser.error(f'Reference and candidate directories must not overlap: {directory}')
    for index, directory in enumerate(candidate_dirs):
        if any(directory in other.parents or other in directory.parents for other in candidate_dirs[:index]):
            parser.error(f'Candidate directories must not overlap: {directory}')
    signature = hashlib.sha256('\n'.join(map(str, [reference_dir, *candidate_dirs])).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else ROOT / '.data/s4-agent/verifier/comparisons' / f'{stamp}-{signature}'
    for directory in [reference_dir, *candidate_dirs]:
        if output_dir == directory or output_dir in directory.parents or directory in output_dir.parents:
            parser.error(f'Output directory must be outside input trees: {output_dir}')
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        parser.error(f'Output destination is not an empty directory: {output_dir}; choose a new --output-dir')

    try:
        raw_reference, reference_info = load_run(reference_dir)
        reference_flat = all(family == '.' for family, _ in raw_reference)
        reference = normalize_families(raw_reference, reference_dir)
        families = sorted({family for family, _ in reference})
        if not reference_info['valid']:
            raise ValueError(f'No valid reference verdicts in {reference_dir}: {reference_info["errors"]}')
        progress('REFERENCE', f'{reference_dir}: {reference_info["valid"]} valid, {reference_info["invalid"]} invalid, {len(families)} families')
        all_rows, summaries, per_family, inputs = [], [], [], []
        common_parent = Path(os.path.commonpath([str(p.parent) for p in candidate_dirs]))
        for directory in candidate_dirs:
            label = directory.relative_to(common_parent).as_posix()
            raw_candidate, info = load_run(directory)
            flat_family = reference_dir.name if reference_flat and all(f == '.' for f, _ in raw_candidate) else None
            candidate = normalize_families(raw_candidate, directory, flat_family)
            rows = compare_records(reference, candidate, label)
            summary = summarize(rows, label, 'all')
            info['label'] = label
            info['extra_clips'] = len(candidate.keys() - reference.keys())
            info['outside_reference_families'] = sorted({f for f, _ in candidate} - set(families))
            info['unmatched_artifacts'] = [
                {'family': f, 'key': k, 'file': candidate[(f, k)]['file']}
                for f, k in sorted(candidate.keys() - reference.keys())
            ]
            progress('CANDIDATE', f'{directory}: {summary["matched_clips"]}/{summary["reference_valid"]} matched valid; '
                     f'{info["invalid"]} invalid artifacts; {info["extra_clips"]} extra clips; statuses={summary["counts"]}')
            if not summary['matched_clips']:
                raise ValueError(
                    f'No valid matching clips for {directory}. Reference families: {families[:8]}; '
                    f'candidate families: {sorted({f for f, _ in candidate})[:8]}; '
                    f'statuses: {summary["counts"]}. Select one run root or the corresponding family folder; '
                    'clip stems must match after removing the backend suffix.'
                )
            inputs.append(info)
            summaries.append(summary)
            all_rows.extend(rows)
            per_family.extend(summarize([r for r in rows if r['family'] == family], label, family) for family in families)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    # All candidates must have valid pairs before any report directory is created.
    output_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or f'Verifier comparison: {reference_dir.name}'
    conflicts = [r for r in all_rows if r['status'] in {'bad_accept', 'false_reject', 'code_mismatch'}]
    transcript_differences = [
        r for r in all_rows if r['transcript_status'] in {'different', 'missing_candidate'}
    ]
    write_csv(output_dir / 'pairs.csv', all_rows)
    write_csv(output_dir / 'conflicts.csv', conflicts)
    write_csv(output_dir / 'transcript_differences.csv', transcript_differences)
    report = [f'# {title}', '', f'Reference: `{reference_dir}`', '',
              'Metrics measure agreement with this reference, not human ground truth. Only valid, matched clips are scored.',
              'Coverage = scored pairs / valid reference clips. Missing or failed results are excluded from decision metrics.',
              'Each candidate uses its own matched subset; compare coverage before comparing scores. N/A means no observations.', '',
              '| Candidate | Matched / reference valid | Coverage | Defect recall | False rejection | Agreement | Bad accepts | Transcript coverage | Exact transcript | Emotion coverage | Exact emotion |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for summary in summaries:
        metrics = summary['overall']
        report.append(f"| {markdown_cell(summary['candidate'])} | {summary['matched_clips']} / {summary['reference_valid']} | "
                      f"{percent(summary['coverage'])} | {percent(metrics['defect_recall'])} | "
                      f"{percent(metrics['false_rejection_rate'])} | {percent(metrics['accuracy'])} | "
                      f"{summary['confusion_matrix']['bad_accepts_missed']} | "
                      f"{percent(summary['transcripts']['coverage'])} | "
                      f"{percent(summary['transcripts']['exact_match_rate'])} | "
                      f"{percent(summary['emotions']['coverage'])} | "
                      f"{percent(summary['emotions']['exact_match_rate'])} |")
    report.extend(['', 'Per-family metrics, input inventories, error counts and unmatched candidate paths are in `summary.json`.',
                   'Every reference clip appears once per candidate in `pairs.csv`, including missing/invalid/hash-mismatched pairs.',
                   'Different or missing candidate transcripts are listed in `transcript_differences.csv` and do not alter verifier agreement.',
                   'Defect codes are compared only when present; a speaker-only verifier does not measure every acoustic criterion.', ''])
    (output_dir / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    conflict_report = [f'# Conflicts: {title}', '', f'Reference: `{reference_dir}`', '',
                       'These are disagreements with the reference. Review the audio before treating them as model errors.', '',
                       '| Candidate | Family / clip | Type | Reference reason | Candidate reason | Audio |',
                       '| --- | --- | --- | --- | --- | --- |']
    for row in conflicts:
        audio = markdown_file_link(row.get('audio_path'), output_dir, 'audio')
        conflict_report.append('| ' + ' | '.join([
            markdown_cell(row['candidate']), markdown_cell(f"{row['family']}/{row['key']}"),
            row['status'], markdown_cell(row['ref_reason']), markdown_cell(row['cand_reason']), audio,
        ]) + ' |')
    if not conflicts:
        conflict_report.extend(['', 'No disagreements among the matched valid clips. See pairs.csv for coverage gaps.'])
    (output_dir / 'conflicts.md').write_text('\n'.join(conflict_report) + '\n', encoding='utf-8')
    plots = [] if args.no_plots else render_plots(summaries, output_dir / 'plots')
    write_json(output_dir / 'summary.json', {
        'schema_version': 3, 'title': title, 'reference': reference_info, 'candidates': inputs,
        'families': families, 'summaries': per_family, 'global_summaries': summaries,
        'plots': plots, 'pairs_csv': str(output_dir / 'pairs.csv'),
        'conflicts_csv': str(output_dir / 'conflicts.csv'), 'conflicts_md': str(output_dir / 'conflicts.md'),
        'transcript_differences_csv': str(output_dir / 'transcript_differences.csv'),
    })
    print('\n'.join(report[2:]), file=sys.stderr)
    progress('COMPARE_DONE', f'Reports: {output_dir}')
    print(output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
