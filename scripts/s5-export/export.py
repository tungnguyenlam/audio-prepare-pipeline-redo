"""Export a file manifest as JSONL or CSV."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, completed, digest, identity, progress, read_json, request, write_json


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('-im', '--input-manifest', type=Path, required=True, help='Path to input dataset manifest JSON')
    p.add_argument('-o', '-of', '--output-file', type=Path, required=True, help='Output JSONL or CSV export file')
    p.add_argument('-f', '--format', choices=('jsonl', 'csv'), required=True, help='Export format choice ("jsonl" or "csv")')
    p.add_argument('-w', '-ow', '--overwrite', action='store_true', help='Overwrite existing output file and sidecars')
    p.add_argument('-c', '--concurrency', type=int, default=1, help='Number of concurrent workers for formatting entries. Set > 1 to enable concurrent execution')
    p.add_argument('-b', '-bs', '--batch-size', type=int, default=1, help='Number of entries to format per batch chunk')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')
    destination = args.output_file.resolve()
    sidecar = destination.with_suffix('.json')
    if args.input_manifest.resolve() in {destination, sidecar} or destination == sidecar:
        p.error('Output and its JSON sidecar must differ from the input manifest and each other')
    metadata = request(identity(args.input_manifest), 'dataset_export', {'format': args.format})
    if completed(destination, metadata, args.overwrite):
        progress('EXPORT_CACHED', f'Output already complete: {destination.name}')
        print(destination)
        return 0
    manifest = read_json(args.input_manifest)
    entries = manifest['entries']
    progress('EXPORT_START', f'Exporting {len(entries)} entries to {args.format.upper()}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            if args.format == 'jsonl':
                def _fmt_jsonl_batch(batch_items):
                    return ''.join(json.dumps(entry, ensure_ascii=False) + '\n' for entry in batch_items)

                batches = [entries[i:i + args.batch_size] for i in range(0, len(entries), args.batch_size)]
                if args.concurrency > 1 and len(batches) > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
                        for chunk in pool.map(_fmt_jsonl_batch, batches):
                            stream.write(chunk)
                else:
                    for b in batches:
                        stream.write(_fmt_jsonl_batch(b))
            else:
                fields = sorted({key for entry in entries for key in entry})
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()

                def _fmt_csv_batch(batch_items):
                    return [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in entry.items()} for entry in batch_items]

                batches = [entries[i:i + args.batch_size] for i in range(0, len(entries), args.batch_size)]
                if args.concurrency > 1 and len(batches) > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
                        for chunk in pool.map(_fmt_csv_batch, batches):
                            writer.writerows(chunk)
                else:
                    for b in batches:
                        writer.writerows(_fmt_csv_batch(b))
        write_json(sidecar, {**metadata, 'output': {}})
        os.replace(temporary, destination)
        write_json(sidecar, {**metadata, 'output': {'sha256': digest(destination), 'rows': len(entries)}})
    finally:
        Path(temporary).unlink(missing_ok=True)
    progress('EXPORT_DONE', f'Wrote {len(entries)} rows -> {destination.name}')
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
