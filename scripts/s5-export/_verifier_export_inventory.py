"""Shared inventory and production-verdict integrity checks for offline exporters."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / 's4-agent' / 'verifier'))
from _common.files import AUDIO_SUFFIXES, digest, progress, read_json, resolve_stored_path
from _verdicts import _known_prompts, _validate_verdict
from _verifier_artifacts import response_complete, verifier_json_paths


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def expected_inputs(paths: list[Path]) -> tuple[dict, bool, dict]:
    """Accept existing indexed audio manifests or exported segments manifests."""
    expected = {}
    durations = {}
    complete = bool(paths)
    for path in paths:
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError('Expected inventory must be a JSON object')
        complete = complete and data.get('complete') is True and not data.get('failures')
        if isinstance(data.get('entries'), list):
            if not all(isinstance(item, dict) for item in data['entries']):
                raise ValueError('Expected inventory entries must be objects')
            entries = [(item.get('path'), item.get('sha256'), False, item.get('duration_s')) for item in data['entries']]
        elif isinstance(data.get('turns'), list):
            if not all(isinstance(item, dict) for item in data['turns']):
                raise ValueError('Expected segment turns must be objects')
            entries = [(item.get('clip'), item.get('clip_sha256'), True,
                        item['end_s'] - item['start_s'] if all(isinstance(item.get(key), (int, float))
                        for key in ('start_s', 'end_s')) else None) for item in data['turns']]
        else:
            raise ValueError('Expected inventory must contain entries or exported turns')
        for name, sha, relative_clip, duration in entries:
            if not isinstance(name, str) or not name or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
                raise ValueError('Every expected clip needs a path and SHA-256; use an exported or indexed manifest')
            audio = ((path.parent / name).resolve() if relative_clip
                     else resolve_stored_path(name, base=path.parent))
            if audio in expected:
                raise ValueError('Duplicate audio in expected inventories; select non-overlapping manifests')
            expected[audio] = sha
            if isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= 0:
                durations[audio] = duration
    if paths and not expected:
        raise ValueError('Expected inventory is empty')
    return expected, complete, durations


def collect_run(directory: Path, manifests: list[Path], configuration: str | None = None,
                *, progress_tag: str = 'EXPORT_SELECT') -> tuple[list[dict], dict]:
    expected, inventory_complete, durations = expected_inputs(manifests)
    candidates = []
    ignored = 0
    configurations = {}
    for path in verifier_json_paths(directory):
        try:
            data = read_json(path)
        except (ValueError, OSError):
            raise ValueError(f'Unreadable JSON in run: {path.name}; repair or isolate it before export') from None
        if not isinstance(data, dict):
            raise ValueError(f'Invalid JSON object in run: {path.name}')
        if data.get('operation') != 'verify':
            if 'verdict' in data or 'decision' in data:
                raise ValueError('Legacy verdict without verify/source contract; regenerate a production artifact')
            continue
        source = data.get('source', {})
        if not isinstance(source, dict) or not isinstance(source.get('path'), str) or not source['path']:
            raise ValueError(f'Verifier artifact has no source identity: {path.name}')
        audio = resolve_stored_path(source['path'], base=path.parent)
        if expected and audio not in expected:
            ignored += 1
            continue
        params = data.get('parameters')
        if not isinstance(params, dict):
            raise ValueError('Verifier parameters are missing or malformed')
        config = {'backend': data.get('model'), 'parameters': params}
        config_hash = fingerprint(config)
        configurations[config_hash] = {'backend': data.get('model'), 'model': params.get('model_id', ''),
                                     'settings_sha256': config_hash,
                                     'prompt_sha256': fingerprint(params.get('prompt'))}
        candidates.append((audio, path, data, config_hash))
    if not candidates:
        raise ValueError('No verifier results match the supplied inventory. Check --input-dir and source paths; no files were exported.')
    if ignored:
        progress(progress_tag, f'Ignored {ignored} verdicts outside the supplied inventory')
    if configuration:
        matches = [key for key in configurations if key.startswith(configuration)]
        if len(matches) != 1:
            raise ValueError('--configuration must uniquely match one settings hash: ' + ', '.join(sorted(configurations)))
        candidates = [item for item in candidates if item[3] == matches[0]]
    artifacts = {}
    conflicts = []
    for audio, path, data, config_hash in candidates:
        if audio in artifacts:
            conflicts.append(audio)
        artifacts[audio] = (path, data, config_hash)
    if conflicts:
        lines = [f'{len(set(conflicts))} clips have multiple verifier results. No result was chosen automatically.',
                 'Narrow --input-dir to a listed folder or add --configuration HASH (unique prefix accepted):']
        for key in sorted({item[3] for item in candidates}):
            group = [item for item in candidates if item[3] == key]
            folders = sorted({item[1].parent.relative_to(directory).as_posix() for item in group})
            lines.append(f'  --configuration {key}: {len(group)} results; folders: {", ".join(folders[:5])}')
        raise ValueError('\n'.join(lines))
    selected = sorted({item[3] for item in candidates})
    configurations = {key: configurations[key] for key in selected}
    if len(configurations) > 1:
        progress(progress_tag, f'{len(configurations)} settings groups; each selected clip has exactly one result.')
    known = _known_prompts()
    rows = []
    for audio in sorted(expected.keys() | artifacts.keys(), key=str):
        row = dict(audio_path='', duration_s='', transcript='', emotion='')
        row.update(verifier_status='missing', reason='No verifier result',
                   transcript_status='needs_transcription', schema_profile='', source_name=audio.name,
                   source_sha256=expected.get(audio, ''), verdict_sha256='')
        if audio in artifacts:
            path, data, config_hash = artifacts[audio]
            row["configuration_sha256"] = config_hash
            source_sha = data['source'].get('sha256', '')
            if not isinstance(source_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', source_sha):
                raise ValueError('Verifier source hash is missing or invalid')
            if audio in expected and expected[audio] != source_sha:
                raise ValueError('Verifier source hash disagrees with expected inventory')
            row.update(source_sha256=source_sha, verdict_sha256=digest(path),
                       verdict_file=path.relative_to(directory).as_posix())
            verdict = data.get('verdict')
            profile, schema_error = _validate_verdict(verdict, data['parameters'].get('prompt'),
                                                      data.get('model', ''), known)
            row['schema_profile'] = profile
            if data.get('status') not in (None, 'success') or data.get('error'):
                invalid = data.get('invalid_verdict')
                uncertain = isinstance(invalid, dict) and invalid.get('decision') == 'uncertain'
                row.update(verifier_status='uncertain' if uncertain else 'failed', reason='Verifier processing failed')
                error = data.get('error')
                if isinstance(error, dict):
                    row['reason'] += ': ' + str(error.get('code', 'unknown'))
            elif schema_error:
                uncertain = isinstance(verdict, dict) and verdict.get('decision') == 'uncertain'
                row.update(verifier_status='uncertain' if uncertain else 'invalid', reason='Invalid verdict: ' + schema_error)
            elif not response_complete(path, data):
                row.update(verifier_status='incomplete', reason='Raw response missing or hash mismatch')
            else:
                transcript = verdict.get('transcript')
                transcript = transcript if isinstance(transcript, str) and transcript.strip() else ''
                emotion = verdict.get('emotion')
                row.update(verifier_status=verdict['decision'], reason=str(verdict.get('reason', '')),
                           transcript=transcript, emotion=emotion if isinstance(emotion, str) else '',
                           transcript_status='provided' if transcript else 'needs_transcription')
        if not audio.is_file():
            row['audio_issue'] = 'Audio file missing'
        elif audio.suffix.lower() not in AUDIO_SUFFIXES:
            row['audio_issue'] = 'Unsupported audio extension'
        elif digest(audio) != row['source_sha256']:
            row['audio_issue'] = 'Audio hash mismatch'
        else:
            row['_source'] = audio
            row['duration_s'] = durations.get(audio, '')
        if row.get('audio_issue'):
            if row['verifier_status'] == 'pass':
                raise ValueError(f'A passed clip has missing or changed audio: {audio}; restore it before export')
            row['reason'] = '; '.join(filter(None, (row['reason'], row['audio_issue'])))
        # Path identity distinguishes identical bytes submitted from different recordings.
        row['clip_id'] = 'clip_' + fingerprint({'path': str(audio), 'sha256': row['source_sha256']})[:20]
        if '_source' in row:
            row['audio_path'] = f"audio/{row['clip_id']}{audio.suffix}"
        rows.append(row)
    counts = dict(Counter(row['verifier_status'] for row in rows))
    unresolved = sum(row['verifier_status'] not in ('pass', 'reject') or bool(row.get('audio_issue')) for row in rows)
    complete = inventory_complete and not unresolved
    summary = {'processing_status': 'complete' if complete else 'partial',
               'coverage': 'verified' if inventory_complete else 'unknown_or_incomplete_inventory',
               'counts': counts, 'total_clips': len(rows),
               'unresolved_clips': unresolved,
               'needs_transcription': sum(row['transcript_status'] == 'needs_transcription' for row in rows),
               'audio_files': sum(bool(row['audio_path']) for row in rows),
               'duration_s': round(sum(row['duration_s'] or 0 for row in rows), 3),
               'configurations': list(configurations.values()),
               'unknown_duration_clips': sum(bool(row['audio_path']) and row['duration_s'] == '' for row in rows),
               'schema_profiles': sorted({row['schema_profile'] for row in rows if row['schema_profile']}),
               'inventory_sha256': [digest(path) for path in manifests]}
    return rows, summary
