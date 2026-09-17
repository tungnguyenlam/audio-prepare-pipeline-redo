"""Experimental TTS cut planning from aligned ASR words and cached Silero JIT probabilities.

Writes an inspectable manifest only. Render it separately with audio/export_segments.
No ASR, VAD inference, speaker inference, or verifier is run implicitly.
"""
from __future__ import annotations

from decimal import Decimal
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, ROOT, identity, probe, progress, read_json, write_json
from _common.segments import normalize_turns, source_path


def pause_boundaries(left, right, probabilities, hop_s, collar, max_silence, threshold, min_silence):
    """Return a protected boundary in each sustained low-probability gap interval."""
    gap_start, gap_end = left['end'], right['start']
    first = max(0, math.ceil(gap_start / hop_s))
    stop = min(len(probabilities), math.floor(gap_end / hop_s))
    spans, begin = [], None
    for i in range(first, stop + 1):
        quiet = i < stop and probabilities[i] < threshold
        if quiet and begin is None:
            begin = i
        if not quiet and begin is not None:
            if (i - begin) * hop_s >= min_silence:
                a = max(begin * hop_s, gap_start + collar)
                b = min(i * hop_s, gap_end - collar)
                if a <= b:
                    choices = [k for k in range(begin, i) if a <= (k + 0.5) * hop_s <= b]
                    cut = ((min(choices, key=lambda k: (probabilities[k], abs((k + .5) * hop_s - (a + b) / 2))) + .5) * hop_s
                           if choices else (a + b) / 2)
                    sentence = left['word'].rstrip().endswith(('.', '!', '?', '…'))
                    mean = sum(probabilities[begin:i]) / (i - begin)
                    spans.append({'left_end': min(cut, gap_start + max_silence),
                                  'right_start': max(cut, gap_end - max_silence),
                                  'cut_s': cut, 'gap_start_s': gap_start, 'gap_end_s': gap_end,
                                  'mean_speech_probability': mean,
                                  'silence_duration_s': (i - begin) * hop_s,
                                  'method': 'sentence_silence' if sentence else 'word_silence',
                                  'cost': 2 * mean + .5 * (1 - min((i - begin) * hop_s / .2, 1)) + (0 if sentence else 2)})
            begin = None
    return spans


def plan_run(words, probabilities, hop_s, rate, params):
    collar = params['collar_ms'] / 1000
    max_silence = params['max_edge_silence_ms'] / 1000
    # Word alignment is not an acoustic envelope. Preserve adjacent VAD support
    # before applying collars, constrained by the known speaker-safe interval.
    search = params['edge_search_ms'] / 1000
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
    nodes = [{'index': 0, 'left_end': start, 'right_start': start, 'method': 'run_start',
              'aligned_s': first_word['start'], 'protected_s': onset,
              'retained_collar_s': max(0., onset - start), 'requested_collar_s': collar, 'cost': 0}]
    for index, (left, right) in enumerate(zip(words, words[1:]), 1):
        for boundary in pause_boundaries(left, right, probabilities, hop_s, collar, max_silence,
                                         params['silence_threshold'], params['min_silence_ms'] / 1000):
            nodes.append({'index': index, **boundary})
    nodes.append({'index': len(words), 'left_end': end, 'right_start': end, 'method': 'run_end',
                  'aligned_s': last_word['end'], 'protected_s': offset,
                  'retained_collar_s': max(0., end - offset), 'requested_collar_s': collar, 'cost': 0})
    minimum = math.ceil(Decimal(str(params['hard_min'])) * rate)
    maximum = math.floor(Decimal(str(params['hard_max'])) * rate)
    prefix = [0.0]
    for word in words:
        prefix.append(prefix[-1] + word['end'] - word['start'])
    # First minimize rejected aligned speech seconds, then target/acoustic cost.
    costs, previous = [(math.inf, math.inf)] * len(nodes), [None] * len(nodes)
    costs[0] = (0.0, 0.0)
    for j in range(1, len(nodes)):
        right = nodes[j]
        for i in range(j):
            left = nodes[i]
            if left['index'] >= right['index'] or not math.isfinite(costs[i][0]):
                continue
            rejected = prefix[right['index']] - prefix[left['index']]
            fallback = (costs[i][0] + rejected, costs[i][1])
            if fallback < costs[j]:
                costs[j], previous[j] = fallback, (i, False)
            a, b = round(left['right_start'] * rate), round(right['left_end'] * rate)
            if not minimum <= b - a <= maximum:
                continue
            d = (b - a) / rate
            target = (params['target_min'] + params['target_max']) / 2
            duration_cost = max(0, params['target_min'] - d) ** 2 + max(0, d - params['target_max']) ** 2 + .02 * (d - target) ** 2
            accepted = (costs[i][0], costs[i][1] + 1 + duration_cost + right['cost'])
            if accepted < costs[j]:
                costs[j], previous[j] = accepted, (i, True)
    chunks, rejected = [], []
    j = len(nodes) - 1
    while j:
        i, accepted = previous[j]
        left, right = nodes[i], nodes[j]
        selected = words[left['index']:right['index']]
        refs = [{'turn_index': w['turn_index'], 'word_index': w['word_index']} for w in selected]
        if accepted:
            a, b = round(left['right_start'] * rate), round(right['left_end'] * rate)
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
        else:
            rejected.append({'reason': 'no_duration_feasible_protected_partition',
                             'start_s': selected[0]['start'], 'end_s': selected[-1]['end'], 'word_refs': refs})
        j = i
    return list(reversed(chunks)), list(reversed(rejected)), nodes


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='ASR JSON with nested words')
    p.add_argument('--input-file', type=Path, help='Same-timeline source override')
    p.add_argument('--speaker-manifest', type=Path, help='Optional diarization; overrides ASR speaker labels')
    p.add_argument('--vad-report', type=Path, required=True, help='evaluate/silero_jit JSON, same source hash')
    p.add_argument('--vad-device', default='cpu', help='Probability track in the VAD report; cuda:0 also selects ROCm')
    p.add_argument('--output-file', type=Path, default=ROOT / '.data/audio/segment_tts/segments.json')
    for name, default in [('target-min', 7.), ('target-max', 10.), ('hard-min', 1.5), ('hard-max', 15.),
                          ('collar-ms', 40.), ('max-edge-silence-ms', 150.), ('max-join-gap', 1.),
                          ('silence-threshold', .20), ('min-silence-ms', 64.),
                          ('edge-search-ms', 250.), ('vad-threshold', .35)]:
        p.add_argument('--' + name, type=float, default=default)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    keys = ('target_min', 'target_max', 'hard_min', 'hard_max', 'collar_ms',
            'max_edge_silence_ms', 'max_join_gap', 'silence_threshold', 'min_silence_ms',
            'edge_search_ms', 'vad_threshold')
    params = {k: getattr(args, k) for k in keys}
    if not all(math.isfinite(x) and x > 0 for x in params.values()):
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
        accepted, dropped, nodes = plan_run(run, probabilities, hop_s, info['sample_rate'], params)
        chunks.extend(accepted)
        rejected.extend(dropped)
        audits.append({'run_index': ri, 'word_count': len(run), 'boundaries': nodes})
    chunks = normalize_turns(chunks, source)
    output = {'schema_version': 1, 'operation': 'segment_tts', 'model': manifest.get('model'),
              'source': audio_identity, 'parameters': {**params, 'algorithm_version': 'experimental-v2',
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
    progress('TTS_PLAN', f'{len(chunks)} candidates; {len(rejected)} rejection records; {len(runs)} speaker runs')
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
