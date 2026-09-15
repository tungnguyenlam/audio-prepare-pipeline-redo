"""Bundle existing manifest audio and a portable manifest into a ZIP archive."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, completed, digest, identity, progress, read_json, request, safe_name, write_json


def main() -> int:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True, help='Path to input dataset manifest JSON')
    p.add_argument('--output-file', type=Path, required=True, help='Output ZIP archive destination path')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing output ZIP archive and sidecars')
    p.add_argument('--concurrency', type=int, default=1, help='Number of concurrent workers for checking file hashes. Set > 1 to enable concurrent execution')
    p.add_argument('--batch-size', type=int, default=1, help='Batch size of entries to verify per worker task')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')
    destination = args.output_file.resolve()
    sidecar = destination.with_suffix('.json')
    manifest = read_json(args.input_manifest)

    def _verify_item(entry):
        path = Path(entry['path'])
        resolved = path.resolve() if path.is_absolute() else (args.input_manifest.parent / path).resolve()
        if entry.get('sha256') and digest(resolved) != entry['sha256']:
            raise ValueError(f'Indexed audio has changed: {resolved}; re-index before bundling')
        return resolved

    def _verify_batch(batch_items):
        return [_verify_item(e) for e in batch_items]

    batches = [manifest['entries'][i:i + args.batch_size] for i in range(0, len(manifest['entries']), args.batch_size)]
    files = []
    try:
        if args.concurrency > 1 and len(batches) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
                for b_files in pool.map(_verify_batch, batches):
                    files.extend(b_files)
        else:
            for b in batches:
                files.extend(_verify_batch(b))
    except ValueError as exc:
        p.error(str(exc))
    if {destination, sidecar} & {args.input_manifest.resolve(), *files} or destination == sidecar:
        p.error('Output and sidecar must differ from every input')
    metadata = request(identity(args.input_manifest), 'dataset_bundle', {'sources': [identity(path) for path in files]})
    if completed(destination, metadata, args.overwrite):
        progress('BUNDLE_CACHED', f'Bundle already complete: {destination.name}')
        print(destination)
        return 0
    total = len(files)
    progress('BUNDLE_START', f'Bundling {total} files into {destination.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.zip')
    os.close(fd)
    step = max(1, total // 10)
    try:
        entries = []
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for i, (entry, source) in enumerate(zip(manifest['entries'], files), 1):
                name = f'audio/{i - 1:06d}-{safe_name(source.stem)}{source.suffix}'
                archive.write(source, name)
                entries.append({**entry, 'path': name, 'relative_path': name})
                if i == 1 or i == total or i % step == 0:
                    progress('BUNDLE_FILE', f'{name}', current=i, total=total)
            archive.writestr('manifest.json', json.dumps({'schema_version': 1, 'entries': entries}, ensure_ascii=False, indent=2))
        write_json(sidecar, {**metadata, 'output': {}})
        os.replace(temporary, destination)
        write_json(sidecar, {**metadata, 'output': {'sha256': digest(destination), 'files': len(entries)}})
    finally:
        Path(temporary).unlink(missing_ok=True)
    progress('BUNDLE_DONE', f'Wrote {len(entries)} files -> {destination.name}')
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
