"""Merge same-speaker turns across acoustic silence before duration filtering."""
from __future__ import annotations

from pathlib import Path

from _manifest import load, save
from _common.files import LoggingArgumentParser, progress
from _common.merge import add_merge_arguments, merge_parameters, merge_turns


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, required=True, help='Input segments.raw.json manifest')
    p.add_argument('-om', '--output-manifest', type=Path,
                   help='Output manifest (default: .data/purity/merge/<family>/segments.json)')
    p.add_argument('-i', '-if', '--input-file', type=Path,
                   help='Source audio override on the same timeline; bypasses the recorded source hash')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', help='Overwrite a conflicting output manifest')
    p.add_argument('-c', '--concurrency', type=int, default=1,
                   help='Shared CLI option; this ordered merge runs sequentially')
    p.add_argument('-b', '-bs', '--batch-size', type=int, default=1,
                   help='Shared CLI option; this ordered merge reads one frame at a time')
    add_merge_arguments(p)
    args = p.parse_args()
    parameters = merge_parameters(args, p)
    if args.concurrency < 1 or args.batch_size < 1:
        p.error('--concurrency and --batch-size must be positive')
    manifest, source = load(args)
    if not manifest.get('complete'):
        p.error('Input manifest must be complete')
    if manifest.get('operation') == 'diarize' and manifest.get('duration_filter_applied') is not False:
        progress('MERGE_INPUT_WARNING', 'Input may already be duration-filtered; use segments.raw.json to retain short turns')
    turns, audit = merge_turns(source, manifest['turns'], **parameters)
    statistics = {'input_turns': len(manifest['turns']), 'output_turns': len(turns),
                  'merged_gaps': sum(item['reason'] == 'merged' for item in audit)}
    progress('MERGE_COMPLETE', f"{statistics['input_turns']} turns -> {statistics['output_turns']} turns; no duration filter applied")
    save(args, manifest, source, turns, 'merge', parameters,
         duration_filter_applied=False, merge_statistics=statistics, merge_audit=audit)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
