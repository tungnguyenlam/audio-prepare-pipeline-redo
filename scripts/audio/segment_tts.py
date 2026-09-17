"""Experimental TTS cut planning from aligned ASR words and cached Silero JIT probabilities.

Strategy: cut at every sentence-ending word whose surroundings contain a sustained
Silero low-probability pause, split any remaining piece longer than --hard-max at the
nearest word pause, then greedily merge contiguous fragments toward --target-min/max.

Writes an inspectable manifest only. Render it separately with audio/export_segments.
No ASR, VAD inference, speaker inference, or verifier is run implicitly.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, probe, progress, read_json, write_json
from _common.merge import parse_bool
from _common.segments import normalize_turns, source_path


SENTENCE_END = ('.', '!', '?', '…')


def low_spans(probabilities, hop_s, a, b, threshold, min_silence):
    """Return [start_s, end_s) spans inside [a, b] where every frame is below threshold."""
    first, stop = max(0, math.ceil(a / hop_s)), min(len(probabilities), math.floor(b / hop_s))
    spans, begin = [], None
    for i in range(first, stop + 1):
        quiet = i < stop and probabilities[i] < threshold
        if quiet and begin is None:
            begin = i
        if not quiet and begin is not None:
            if (i - begin) * hop_s >= min_silence:
                spans.append((begin * hop_s, i * hop_s))
            begin = None
    return spans


def cut_candidate(left, right, probabilities, hop_s, params):
    """Return a boundary between two words, or None when VAD shows no pause nearby.

    Alignment smears words over pauses, so the search window extends past the
    aligned gap into either word's extent, never beyond the neighbouring words.
    """
    search = params['cut_search_ms'] / 1000
    a = max(left['start'], left['end'] - search)
    b = min(right['end'], right['start'] + search)
    spans = low_spans(probabilities, hop_s, a, b, params['silence_threshold'], params['min_silence_ms'] / 1000)
    if not spans:
        return None
    gap_mid = (left['end'] + right['start']) / 2
    span_start, span_end = min(spans, key=lambda s: abs((s[0] + s[1]) / 2 - gap_mid))
    cut = (span_start + span_end) / 2
    collar, max_silence = params['collar_ms'] / 1000, params['max_edge_silence_ms'] / 1000
    frames = probabilities[math.floor(span_start / hop_s):math.ceil(span_end / hop_s)]
    sentence = left['word'].rstrip().endswith(SENTENCE_END)
    return {'left_end': min(cut, max(span_start + collar, span_start + max_silence)),
            'right_start': max(cut, min(span_end - collar, span_end - max_silence)),
            'cut_s': cut, 'pause_start_s': span_start, 'pause_end_s': span_end,
            'aligned_gap_start_s': left['end'], 'aligned_gap_end_s': right['start'],
            'inside_aligned_word': cut < left['end'] or cut > right['start'],
            'max_speech_probability': max(frames) if frames else None,
            'silence_duration_s': span_end - span_start,
            'method': 'sentence_pause' if sentence else 'word_pause'}


def run_edges(words, probabilities, hop_s, params):
    """Return protected run start/end nodes with VAD-supported outer edges."""
    collar, search = params['collar_ms'] / 1000, params['edge_search_ms'] / 1000
    first_word, last_word = words[0], words[-1]
    onset, offset = first_word['start'], last_word['end']
    a = max(first_word['speaker_start_s'], onset - search)
    b = min(last_word['speaker_end_s'], offset + search)
    for k in range(max(0, math.floor(a / hop_s)), min(len(probabilities), math.ceil(b / hop_s))):
        if probabilities[k] < params['vad_threshold']:
            continue
        frame_start, frame_end = k * hop_s, (k + 1) * hop_s
        if frame_start < first_word['start'] and frame_end > a:
            onset = min(onset, max(a, frame_start))
        if frame_end > last_word['end'] and frame_start < b:
            offset = max(offset, min(b, frame_end))
    start = max(first_word['speaker_start_s'], onset - collar)
    end = min(last_word['speaker_end_s'], offset + collar)
    return ({'index': 0, 'left_end': start, 'right_start': start, 'method': 'run_start',
             'aligned_s': first_word['start'], 'protected_s': onset,
             'retained_collar_s': max(0., onset - start), 'requested_collar_s': collar},
            {'index': len(words), 'left_end': end, 'right_start': end, 'method': 'run_end',
             'aligned_s': last_word['end'], 'protected_s': offset,
             'retained_collar_s': max(0., end - offset), 'requested_collar_s': collar})


def plan_run(words, probabilities, hop_s, rate, params):
    """Cut at every VAD-confirmed sentence end, split over-long pieces at word
    pauses, then optionally merge contiguous fragments toward the target band."""
    hard_max = params['hard_max']
    candidates = {}
    for index, (left, right) in enumerate(zip(words, words[1:]), 1):
        node = cut_candidate(left, right, probabilities, hop_s, params)
        if node:
            candidates[index] = {'index': index, **node}
    start_node, end_node = run_edges(words, probabilities, hop_s, params)
    cuts = [i for i, n in candidates.items() if n['method'] == 'sentence_pause']
    pieces = [(a, b) for a, b in zip([0, *cuts], [*cuts, len(words)])]
    # Fallback: split sentences longer than hard_max at the word pause nearest their middle.
    final, pending = [], list(pieces)
    while pending:
        a, b = pending.pop(0)
        left = start_node if a == 0 else candidates[a]
        right = end_node if b == len(words) else candidates[b]
        if right['left_end'] - left['right_start'] <= hard_max:
            final.append((a, b))
            continue
        middle = (left['right_start'] + right['left_end']) / 2
        options = [i for i in candidates if a < i < b and candidates[i]['method'] == 'word_pause']
        if not options:
            final.append((a, b))
            continue
        split = min(options, key=lambda i: abs(candidates[i]['cut_s'] - middle))
        pending[:0] = [(a, split), (split, b)]
    final.sort()
    fragments = []
    for a, b in final:
        left = start_node if a == 0 else candidates[a]
        right = end_node if b == len(words) else candidates[b]
        fragments.append({'first': a, 'last': b, 'left': left, 'right': right,
                          'duration_s': right['left_end'] - left['right_start']})
    if params['merge']:
        groups, current = [], None
        for fragment in fragments:
            if current is None:
                current = dict(fragment)
                continue
            combined = fragment['right']['left_end'] - current['left']['right_start']
            if combined <= params['target_max'] or (current['duration_s'] < params['target_min'] and combined <= hard_max):
                current.update(last=fragment['last'], right=fragment['right'], duration_s=combined)
            else:
                groups.append(current)
                current = dict(fragment)
        if current is not None:
            if (groups and current['duration_s'] < params['hard_min'] and
                    current['right']['left_end'] - groups[-1]['left']['right_start'] <= hard_max):
                groups[-1].update(last=current['last'], right=current['right'],
                                  duration_s=current['right']['left_end'] - groups[-1]['left']['right_start'])
            else:
                groups.append(current)
    else:
        groups = fragments
    chunks, rejected = [], []
    collar = params['collar_ms'] / 1000
    for group in groups:
        left, right = group['left'], group['right']
        selected = words[group['first']:group['last']]
        refs = [{'turn_index': w['turn_index'], 'word_index': w['word_index']} for w in selected]
        a, b = round(left['right_start'] * rate), round(right['left_end'] * rate)
        duration = (b - a) / rate
        if not params['hard_min'] <= duration <= hard_max:
            rejected.append({'reason': 'too_short' if duration < params['hard_min'] else 'no_pause_within_hard_max',
                             'start_s': selected[0]['start'], 'end_s': selected[-1]['end'], 'word_refs': refs})
            continue
        text = ' '.join(w['word'] for w in selected)
        flags = [f'{side}_collar_truncated' for side, node in [('start', left), ('end', right)]
                 if node.get('retained_collar_s', collar) < collar - 1 / rate]
        chunks.append({'speaker_id': selected[0]['speaker_id'], 'start_s': a / rate, 'end_s': b / rate,
                       'confidence': None, '_transcript': text, 'text': text,
                       '_words': [{**w, 'text': w['word'], 'clip_start_s': w['start'] - a / rate,
                                   'clip_end_s': w['end'] - a / rate} for w in selected],
                       'lineage': {'word_refs': refs},
                       'boundary': {'start': left, 'end': right},
                       'quality': {'status': 'candidate', 'flags': flags, 'human_verified': False}})
    nodes = [start_node, *candidates.values(), end_node]
    audit_fragments = [{'first_word': f['first'], 'last_word': f['last'], 'duration_s': f['duration_s'],
                        'start_method': f['left']['method'], 'end_method': f['right']['method']} for f in fragments]
    return chunks, rejected, nodes, audit_fragments


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='ASR JSON with nested words')
    p.add_argument('--input-file', type=Path, help='Same-timeline source override')
    p.add_argument('--speaker-manifest', type=Path, help='Optional diarization; overrides ASR speaker labels')
    p.add_argument('--vad-report', type=Path, required=True, help='evaluate/silero_jit JSON, same source hash')
    p.add_argument('--vad-device', default='cpu', help='Probability track in the VAD report; cuda:0 also selects ROCm')
    p.add_argument('--output-file', type=Path, default=ROOT / '.data/audio/segment_tts/segments.json')
    for name, default, doc in [
            ('target-min', 7., 'merge fragments until at least this many seconds'),
            ('target-max', 10., 'do not merge past this many seconds unless still under target-min'),
            ('hard-min', 1.5, 'reject shorter results'), ('hard-max', 15., 'never exceed; split at word pauses'),
            ('collar-ms', 40., 'minimum pause retained at each cut edge'),
            ('max-edge-silence-ms', 150., 'maximum pause retained at each cut edge'),
            ('max-join-gap', 1., 'longer aligned gaps end a speaker run'),
            ('silence-threshold', .10, 'every Silero frame in a pause must be below this'),
            ('min-silence-ms', 64., 'minimum pause length'),
            ('cut-search-ms', 400., 'search this far into either word around a candidate gap'),
            ('edge-search-ms', 250., 'protect VAD speech this far outside a run'),
            ('vad-threshold', .35, 'Silero speech threshold for run-edge protection')]:
        p.add_argument('--' + name, type=float, default=default, help=f'{doc} (default: {default})')
    p.add_argument('--merge', nargs='?', const=True, default=True, type=parse_bool, metavar='BOOL',
                   help='Merge contiguous fragments toward the target band (default: true; --merge false keeps raw fragments)')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true')
    args = p.parse_args()
    keys = ('target_min', 'target_max', 'hard_min', 'hard_max', 'collar_ms',
            'max_edge_silence_ms', 'max_join_gap', 'silence_threshold', 'min_silence_ms',
            'cut_search_ms', 'edge_search_ms', 'vad_threshold')
    params = {k: getattr(args, k) for k in keys}
    params['merge'] = args.merge
    if not all(math.isfinite(params[k]) and params[k] > 0 for k in keys):
        p.error('Numeric settings must be finite and positive')
    if not 1.5 <= args.hard_min <= args.target_min <= args.target_max <= args.hard_max <= 15:
        p.error('Require 1.5 <= hard-min <= target-min <= target-max <= hard-max <= 15')
    if (args.collar_ms > args.max_edge_silence_ms or args.max_edge_silence_ms > 150 or
            not args.silence_threshold < args.vad_threshold < 1):
        p.error('Require collar <= max edge silence <= 150 ms and silence threshold < VAD threshold < 1')
    inputs = [args.input_manifest, args.vad_report] + ([args.speaker_manifest] if args.speaker_manifest else [])
    destination = args.output_file.resolve()
    manifest = read_json(args.input_manifest)
    source = source_path(manifest, args.input_manifest, args.input_file)
    if destination in {path.resolve() for path in [*inputs, source]}:
        p.error('Output cannot replace an input')
    if destination.exists() and not args.overwrite:
        p.error('Output exists; choose another path or --overwrite')
    audio_identity = identity(source)
    report = read_json(args.vad_report)
    if report.get('operation') != 'evaluate_silero_jit' or not report.get('complete'):
        p.error('VAD report must be a completed native JIT report')
    if report['source']['sha256'] != audio_identity['sha256'] or manifest['source']['sha256'] != audio_identity['sha256']:
        p.error('ASR, VAD and audio must identify the same source bytes')
    track = report['devices'].get(args.vad_device, {})
    if track.get('status') != 'ok':
        p.error('Requested VAD track did not complete successfully')
    probabilities = track['probabilities']
    info = probe(source)
    hop_s = report['parameters']['frame_samples'] / report['parameters']['sample_rate']
    if len(probabilities) != report['audio']['frame_count'] or not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities):
        p.error('Invalid probability track')
    speaker_turns = None
    if args.speaker_manifest:
        speakers = read_json(args.speaker_manifest)
        if speakers['source']['sha256'] != audio_identity['sha256']:
            p.error('Speaker manifest and audio hashes differ')
        speaker_turns = normalize_turns(speakers['turns'], source)

    words, rejected = [], []
    for ti, turn in enumerate(manifest['turns']):
        if not turn.get('words'):
            rejected.append({'reason': 'missing_words', 'turn_index': ti,
                             'start_s': float(turn['start_time']), 'end_s': float(turn['end_time'])})
        for wi, raw in enumerate(turn.get('words', [])):
            ref = {'turn_index': ti, 'word_index': wi}
            a, b = float(raw['start']), float(raw['end'])
            text = str(raw.get('word', raw.get('text', ''))).strip()
            if not text or not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= info['duration_s']):
                rejected.append({'reason': 'invalid_word', **ref})
                continue
            if speaker_turns is None:
                owner = {'speaker_id': str(turn['speaker_id']),
                         'start_s': max(0., min(float(turn['start_time']), a)),
                         'end_s': min(info['duration_s'], max(float(turn['end_time']), b))}
                conflicts = [t for k, t in enumerate(manifest['turns']) if k != ti and
                             str(t['speaker_id']) != owner['speaker_id'] and float(t['start_time']) < b and float(t['end_time']) > a]
            else:
                matches = [t for t in speaker_turns if t['start_s'] <= a and b <= t['end_s']]
                owner = matches[0] if len(matches) == 1 else None
                conflicts = [t for t in speaker_turns if owner is not None and t['speaker_id'] != owner['speaker_id']
                             and t['start_s'] < b and t['end_s'] > a]
            if owner is None or conflicts or owner['speaker_id'] in (None, '', 'unknown'):
                rejected.append({'reason': 'unassigned_or_overlapping_speaker', 'start_s': a, 'end_s': b, **ref})
                continue
            bounds = speaker_turns if speaker_turns is not None else [
                {'speaker_id': str(t['speaker_id']), 'start_s': float(t['start_time']), 'end_s': float(t['end_time'])}
                for t in manifest['turns']]
            safe_start, safe_end = owner['start_s'], owner['end_s']
            for other in bounds:
                if other['speaker_id'] != owner['speaker_id']:
                    if other['end_s'] <= a:
                        safe_start = max(safe_start, other['end_s'])
                    if other['start_s'] >= b:
                        safe_end = min(safe_end, other['start_s'])
            words.append({**raw, 'word': text, 'start': a, 'end': b, **ref,
                          'speaker_id': owner['speaker_id'], 'speaker_start_s': safe_start, 'speaker_end_s': safe_end})
    words.sort(key=lambda w: (w['start'], w['end']))
    runs, current = [], []
    rejected_spans = [(r['start_s'], r['end_s']) for r in rejected if 'start_s' in r]
    for word in words:
        if current:
            prev = current[-1]
            gap_a, gap_b = prev['end'], word['start']
            timeline = speaker_turns if speaker_turns is not None else [
                {'speaker_id': str(t['speaker_id']), 'start_s': float(t['start_time']), 'end_s': float(t['end_time'])}
                for t in manifest['turns']]
            competitor = any(t['speaker_id'] != word['speaker_id'] and t['start_s'] < word['end'] and
                             t['end_s'] > prev['start'] for t in timeline)
            blocked = any(a < gap_b and b > gap_a for a, b in rejected_spans)
            if (word['speaker_id'] != prev['speaker_id'] or gap_b < gap_a or
                    gap_b - gap_a > args.max_join_gap or competitor or blocked):
                runs.append(current)
                current = []
        current.append(word)
    if current:
        runs.append(current)
    # An acoustic collar must not reach a word omitted at a neighboring run or
    # rejection boundary, even when both carry the same coarse speaker label.
    for run in runs:
        left, right = run[0], run[-1]
        for word in words:
            if word is not left and word['end'] <= left['start']:
                left['speaker_start_s'] = max(left['speaker_start_s'], word['end'])
            if word is not right and word['start'] >= right['end']:
                right['speaker_end_s'] = min(right['speaker_end_s'], word['start'])
        for a, b in rejected_spans:
            if b <= left['start']:
                left['speaker_start_s'] = max(left['speaker_start_s'], b)
            if a >= right['end']:
                right['speaker_end_s'] = min(right['speaker_end_s'], a)
    chunks, audits = [], []
    for ri, run in enumerate(runs):
        accepted, dropped, nodes, fragments = plan_run(run, probabilities, hop_s, info['sample_rate'], params)
        chunks.extend(accepted)
        rejected.extend(dropped)
        audits.append({'run_index': ri, 'word_count': len(run), 'boundaries': nodes, 'fragments': fragments})
    chunks = normalize_turns(chunks, source)
    output = {'schema_version': 1, 'operation': 'segment_tts', 'model': manifest.get('model'),
              'source': audio_identity, 'parameters': {**params, 'algorithm_version': 'experimental-v3',
              'implementation': identity(Path(__file__)),
              'input_manifest': identity(args.input_manifest), 'vad_report': identity(args.vad_report),
              'vad_device': args.vad_device, 'speaker_manifest': identity(args.speaker_manifest) if args.speaker_manifest else None},
              'source_sample_rate': info['sample_rate'], 'timestamp_origin': 'diarized_input',
              'speaker_ids': sorted({c['speaker_id'] for c in chunks}), 'turns': chunks,
              'rejected': rejected, 'audit': audits, 'clips_valid': False, 'complete': True,
              'limitations': ['Experimental boundary planner; no phoneme completeness guarantee.',
                              'Transcripts join ASR word units; punctuation/character-span fidelity needs review.',
                              'Outer-edge VAD search is bounded; quiet phonemes and timestamp errors still require review.']}
    write_json(destination, output)
    durations = [c['end_s'] - c['start_s'] for c in chunks]
    fragment_count = sum(len(a['fragments']) for a in audits)
    sentence_cuts = sum(n['method'] == 'sentence_pause' for a in audits for n in a['boundaries'])
    sentence_words = sum(w['word'].endswith(SENTENCE_END) for w in words)
    mean = sum(durations) / len(durations) if durations else 0.
    progress('TTS_PLAN', f'{len(runs)} speaker runs; {sentence_cuts}/{sentence_words} sentence ends passed the VAD gate; '
             f'{fragment_count} fragments -> {len(chunks)} candidates (mean {mean:.2f}s, '
             f'{sum(args.target_min <= d <= args.target_max for d in durations)} in target band); '
             f'{len(rejected)} rejection records')
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
