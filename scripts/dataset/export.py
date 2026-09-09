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
from _common.files import completed, digest, identity, read_json, request, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--output-file', type=Path, required=True)
    p.add_argument('--format', choices=('jsonl', 'csv'), required=True)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    destination = args.output_file.resolve()
    sidecar = destination.with_suffix('.json')
    if args.input_manifest.resolve() in {destination, sidecar} or destination == sidecar:
        p.error('Output and its JSON sidecar must differ from the input manifest and each other')
    metadata = request(identity(args.input_manifest), 'dataset_export', {'format': args.format})
    if completed(destination, metadata, args.overwrite):
        print(destination)
        return 0
    manifest = read_json(args.input_manifest)
    entries = manifest['entries']
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            if args.format == 'jsonl':
                for entry in entries:
                    stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
            else:
                fields = sorted({key for entry in entries for key in entry})
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in entry.items()} for entry in entries)
        write_json(sidecar, {**metadata, 'output': {}})
        os.replace(temporary, destination)
        write_json(sidecar, {**metadata, 'output': {'sha256': digest(destination), 'rows': len(entries)}})
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
