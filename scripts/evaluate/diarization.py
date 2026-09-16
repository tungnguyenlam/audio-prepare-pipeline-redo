"""Backend-independent evaluation for speaker diarization turns."""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable



from concurrent.futures import ThreadPoolExecutor


def _normalize_single_turn(turn: dict[str, Any]) -> dict[str, Any]:
    if isinstance(turn, dict):
        speaker_id = str(turn.get("speaker_id") or "").strip()
        try:
            start_s = float(turn.get("start_s"))
            end_s = float(turn.get("end_s"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Every turn requires numeric start_s and end_s") from exc
    else:
        raise TypeError("turns must contain JSON objects")
    if not speaker_id:
        raise ValueError("Every turn requires a non-empty speaker_id")
    if not isfinite(start_s) or not isfinite(end_s) or start_s < 0 or end_s <= start_s:
        raise ValueError("Every turn must have finite timestamps with 0 <= start_s < end_s")
    return {"speaker_id": speaker_id, "start_s": start_s, "end_s": end_s}


def _normalized_turns(turns: Iterable[dict[str, Any]], *, batch_size: int = 1, concurrency: int = 1) -> list[dict[str, Any]]:
    turn_list = list(turns)
    if concurrency > 1 and len(turn_list) > batch_size:
        batches = [turn_list[i:i + batch_size] for i in range(0, len(turn_list), batch_size)]
        with ThreadPoolExecutor(max_workers=min(concurrency, len(batches))) as ex:
            results = ex.map(lambda chunk: [_normalize_single_turn(t) for t in chunk], batches)
            return [t for chunk_res in results for t in chunk_res]
    return [_normalize_single_turn(t) for t in turn_list]


def _maximum_weight_assignment(
    row_ids: list[str],
    column_ids: list[str],
    weights: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Return a maximum-weight one-to-one row-to-column assignment."""
    size = max(len(row_ids), len(column_ids))
    if size == 0:
        return {}
    max_weight = max(weights.values(), default=0.0)
    cost = [
        [
            max_weight
            - (
                weights.get((row_ids[row], column_ids[column]), 0.0)
                if row < len(row_ids) and column < len(column_ids)
                else 0.0
            )
            for column in range(size)
        ]
        for row in range(size)
    ]

    # Hungarian algorithm for a square cost matrix, using one-based work arrays.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    matched_row = [0] * (size + 1)
    predecessor = [0] * (size + 1)
    for row in range(1, size + 1):
        matched_row[0] = row
        column0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = matched_row[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    predecessor[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[matched_row[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched_row[column0] == 0:
                break
        while True:
            column1 = predecessor[column0]
            matched_row[column0] = matched_row[column1]
            column0 = column1
            if column0 == 0:
                break

    assignment = {}
    for column in range(1, size + 1):
        row = matched_row[column] - 1
        column_index = column - 1
        if row < len(row_ids) and column_index < len(column_ids):
            assignment[row_ids[row]] = column_ids[column_index]
    return assignment


def evaluate_diarization(
    reference_turns: Iterable[dict[str, Any]],
    hypothesis_turns: Iterable[dict[str, Any]],
    *,
    duration_s: float,
    collar_s: float = 0.0,
    skip_overlap: bool = False,
    concurrency: int = 1,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate hypothesis turns against a manually annotated reference.

    The calculation uses exact interval boundaries instead of time sampling.
    Hypothesis speakers are mapped one-to-one to reference speakers by maximum
    scored temporal overlap before missed speech, false alarm, confusion, DER,
    and JER are computed.

    Args:
        reference_turns: Ground-truth speaker turns.
        hypothesis_turns: Model-produced speaker turns.
        duration_s: Duration of the shared source audio.
        collar_s: Forgiveness excluded on each side of every reference boundary.
        skip_overlap: Exclude regions with multiple active reference speakers.
        concurrency: Number of worker threads for parallel turn normalization.
        batch_size: Batch size for chunked turn processing.

    Returns:
        JSON-compatible evaluation metrics and speaker mapping.

    Raises:
        TypeError: If numeric settings have invalid types.
        ValueError: If settings or turns are invalid.
    """
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)):
        raise TypeError("duration_s must be a number")
    if isinstance(collar_s, bool) or not isinstance(collar_s, (int, float)):
        raise TypeError("collar_s must be a number")
    duration_s = float(duration_s)
    collar_s = float(collar_s)
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and greater than zero")
    if not isfinite(collar_s) or collar_s < 0:
        raise ValueError("collar_s must be finite and non-negative")

    reference = _normalized_turns(reference_turns, batch_size=batch_size, concurrency=concurrency)
    hypothesis = _normalized_turns(hypothesis_turns, batch_size=batch_size, concurrency=concurrency)
    if not reference:
        raise ValueError("reference_turns must contain at least one annotated turn")
    if any(turn["end_s"] > duration_s + 0.05 for turn in reference + hypothesis):
        raise ValueError("A turn exceeds the shared audio duration")

    reference_ids = sorted({turn["speaker_id"] for turn in reference})
    hypothesis_ids = sorted({turn["speaker_id"] for turn in hypothesis})
    boundaries = {0.0, duration_s}
    excluded_collars: list[tuple[float, float]] = []
    for turn in reference:
        boundaries.update((max(0.0, turn["start_s"]), min(duration_s, turn["end_s"])))
        if collar_s:
            for boundary in (turn["start_s"], turn["end_s"]):
                start = max(0.0, boundary - collar_s)
                end = min(duration_s, boundary + collar_s)
                excluded_collars.append((start, end))
                boundaries.update((start, end))
    for turn in hypothesis:
        boundaries.update((max(0.0, turn["start_s"]), min(duration_s, turn["end_s"])))

    intervals: list[tuple[float, set[str], set[str]]] = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        if any(left <= midpoint < right for left, right in excluded_collars):
            continue
        active_reference = {
            turn["speaker_id"]
            for turn in reference
            if turn["start_s"] <= midpoint < turn["end_s"]
        }
        if skip_overlap and len(active_reference) > 1:
            continue
        active_hypothesis = {
            turn["speaker_id"]
            for turn in hypothesis
            if turn["start_s"] <= midpoint < turn["end_s"]
        }
        intervals.append((end - start, active_reference, active_hypothesis))

    overlap_weights = {
        (hypothesis_id, reference_id): sum(
            length
            for length, active_reference, active_hypothesis in intervals
            if reference_id in active_reference and hypothesis_id in active_hypothesis
        )
        for hypothesis_id in hypothesis_ids
        for reference_id in reference_ids
    }
    hypothesis_to_reference = _maximum_weight_assignment(
        hypothesis_ids, reference_ids, overlap_weights
    )
    hypothesis_to_reference = {
        hypothesis_id: reference_id
        for hypothesis_id, reference_id in hypothesis_to_reference.items()
        if overlap_weights.get((hypothesis_id, reference_id), 0.0) > 0
    }

    scored_audio_s = sum(length for length, _, _ in intervals)
    reference_speaker_s = 0.0
    hypothesis_speaker_s = 0.0
    missed_s = 0.0
    false_alarm_s = 0.0
    confusion_s = 0.0
    correct_s = 0.0
    reference_duration = {speaker_id: 0.0 for speaker_id in reference_ids}
    hypothesis_duration = {speaker_id: 0.0 for speaker_id in hypothesis_ids}
    speaker_intersection = {speaker_id: 0.0 for speaker_id in reference_ids}

    for length, active_reference, active_hypothesis in intervals:
        reference_count = len(active_reference)
        hypothesis_count = len(active_hypothesis)
        mapped_hypothesis = {
            hypothesis_to_reference[hypothesis_id]
            for hypothesis_id in active_hypothesis
            if hypothesis_id in hypothesis_to_reference
        }
        correct_count = len(active_reference & mapped_hypothesis)
        reference_speaker_s += reference_count * length
        hypothesis_speaker_s += hypothesis_count * length
        missed_s += max(0, reference_count - hypothesis_count) * length
        false_alarm_s += max(0, hypothesis_count - reference_count) * length
        confusion_s += (min(reference_count, hypothesis_count) - correct_count) * length
        correct_s += correct_count * length
        for speaker_id in active_reference:
            reference_duration[speaker_id] += length
            if speaker_id in mapped_hypothesis:
                speaker_intersection[speaker_id] += length
        for speaker_id in active_hypothesis:
            hypothesis_duration[speaker_id] += length

    reference_to_hypothesis = {
        reference_id: hypothesis_id
        for hypothesis_id, reference_id in hypothesis_to_reference.items()
    }
    per_speaker = []
    speaker_error_rates = []
    for reference_id in reference_ids:
        hypothesis_id = reference_to_hypothesis.get(reference_id)
        ref_s = reference_duration[reference_id]
        hyp_s = hypothesis_duration.get(hypothesis_id, 0.0) if hypothesis_id else 0.0
        intersection_s = speaker_intersection[reference_id]
        union_s = ref_s + hyp_s - intersection_s
        error_rate = 1.0 - (intersection_s / union_s) if union_s else 0.0
        speaker_error_rates.append(error_rate)
        per_speaker.append(
            {
                "reference_speaker_id": reference_id,
                "hypothesis_speaker_id": hypothesis_id,
                "reference_s": round(ref_s, 6),
                "hypothesis_s": round(hyp_s, 6),
                "intersection_s": round(intersection_s, 6),
                "coverage_pct": round((intersection_s / ref_s * 100) if ref_s else 0.0, 4),
                "jer_pct": round(error_rate * 100, 4),
            }
        )

    denominator = reference_speaker_s
    diarization_error_s = missed_s + false_alarm_s + confusion_s
    return {
        "der_pct": round((diarization_error_s / denominator * 100) if denominator else 0.0, 4),
        "jer_pct": round(
            (sum(speaker_error_rates) / len(speaker_error_rates) * 100)
            if speaker_error_rates
            else 0.0,
            4,
        ),
        "missed_speech_s": round(missed_s, 6),
        "false_alarm_s": round(false_alarm_s, 6),
        "speaker_confusion_s": round(confusion_s, 6),
        "correct_speaker_s": round(correct_s, 6),
        "reference_speaker_s": round(reference_speaker_s, 6),
        "hypothesis_speaker_s": round(hypothesis_speaker_s, 6),
        "scored_audio_s": round(scored_audio_s, 6),
        "collar_s": collar_s,
        "skip_overlap": bool(skip_overlap),
        "speaker_mapping": [
            {
                "hypothesis_speaker_id": hypothesis_id,
                "reference_speaker_id": reference_id,
                "overlap_s": round(overlap_weights.get((hypothesis_id, reference_id), 0.0), 6),
            }
            for hypothesis_id, reference_id in sorted(hypothesis_to_reference.items())
            if overlap_weights.get((hypothesis_id, reference_id), 0.0) > 0
        ],
        "unmapped_hypothesis_speakers": sorted(
            set(hypothesis_ids) - set(hypothesis_to_reference)
        ),
        "per_speaker": per_speaker,
    }


def main() -> int:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _common.files import LoggingArgumentParser, identity, positive_int, progress, read_json, write_json
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-rm', '--reference-manifest', type=Path, required=True, help='Path to reference segments.json containing ground-truth turns')
    p.add_argument('-im', '--input-manifest', type=Path, required=True, help='Path to hypothesis segments.json containing model-predicted turns')
    p.add_argument('--duration', type=float, required=True, help='Total duration of audio in seconds for DER evaluation')
    p.add_argument('--collar', type=float, default=0.0, help='Forgiveness collar in seconds around reference boundaries (default: 0.0)')
    p.add_argument('--skip-overlap', action='store_true', help='Exclude overlapping speech regions from evaluation')
    p.add_argument('-o', '-of', '--output-file', type=Path, required=True, help='Output JSON file path for evaluation metrics')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', help='Overwrite existing output file if present')
    p.add_argument('-c', '--concurrency', type=positive_int, default=1, help='Number of worker threads for parallel manifest loading and turn normalization (default: 1)')
    p.add_argument('-b', '-bs', '--batch-size', type=positive_int, default=1, help='Batch size for chunked turn processing (default: 1)')
    args = p.parse_args()
    dest = args.output_file.resolve()
    if dest in {args.reference_manifest.resolve(), args.input_manifest.resolve()}:
        p.error('Output cannot overwrite an input')
    metadata = {'operation': 'diarization_metrics', 'reference': identity(args.reference_manifest),
                'source': identity(args.input_manifest), 'parameters': {'duration_s': args.duration,
                'collar_s': args.collar, 'skip_overlap': args.skip_overlap,
                'concurrency': args.concurrency, 'batch_size': args.batch_size}}
    if dest.exists() and not args.overwrite:
        old = read_json(dest)
        if all(old.get(k) == v for k, v in metadata.items()) and 'metrics' in old:
            print(dest)
            return 0
        p.error('Conflicting output; use --overwrite')
    progress('EVAL_DIAR', f'Evaluating DER: {args.input_manifest.name} vs {args.reference_manifest.name} (collar={args.collar}s)')
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_ref = ex.submit(read_json, args.reference_manifest)
            fut_hyp = ex.submit(read_json, args.input_manifest)
            ref_data = fut_ref.result()
            hyp_data = fut_hyp.result()
    else:
        ref_data = read_json(args.reference_manifest)
        hyp_data = read_json(args.input_manifest)
    metrics = evaluate_diarization(ref_data['turns'], hyp_data['turns'], duration_s=args.duration,
                                   collar_s=args.collar, skip_overlap=args.skip_overlap,
                                   concurrency=args.concurrency, batch_size=args.batch_size)
    write_json(dest, {**metadata, 'metrics': metrics})
    der_val = metrics.get('der_pct', metrics.get('der', 0.0) * 100)
    miss_val = metrics.get('missed_speech_s', metrics.get('miss', 0.0) * 100)
    fa_val = metrics.get('false_alarm_s', metrics.get('false_alarm', 0.0) * 100)
    conf_val = metrics.get('speaker_confusion_s', metrics.get('speaker_confusion', 0.0) * 100)
    progress('EVAL_DIAR_DONE', f'DER: {der_val:.2f}%, Miss: {miss_val:.2f}s, FA: {fa_val:.2f}s, Conf: {conf_val:.2f}s -> {dest.name}')
    print(dest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
