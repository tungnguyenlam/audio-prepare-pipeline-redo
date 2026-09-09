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
from _common.files import completed, digest, identity, read_json, request, safe_name, write_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--output-file', type=Path, required=True)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    destination = args.output_file.resolve()
    sidecar = destination.with_suffix('.json')
    manifest = read_json(args.input_manifest)
    files = []
    for entry in manifest['entries']:
        path = Path(entry['path'])
        path = path.resolve() if path.is_absolute() else (args.input_manifest.parent / path).resolve()
        if entry.get('sha256') and digest(path) != entry['sha256']:
            p.error(f'Indexed audio has changed: {path}; re-index before bundling')
        files.append(path)
    if {destination, sidecar} & {args.input_manifest.resolve(), *files} or destination == sidecar:
        p.error('Output and sidecar must differ from every input')
    metadata = request(identity(args.input_manifest), 'dataset_bundle', {'sources': [identity(path) for path in files]})
    if completed(destination, metadata, args.overwrite):
        print(destination)
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.zip')
    os.close(fd)
    try:
        entries = []
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for i, (entry, source) in enumerate(zip(manifest['entries'], files)):
                name = f'audio/{i:06d}-{safe_name(source.name)}'
                archive.write(source, name)
                entries.append({**entry, 'path': name, 'relative_path': name})
            archive.writestr('manifest.json', json.dumps({'schema_version': 1, 'entries': entries}, ensure_ascii=False, indent=2))
        write_json(sidecar, {**metadata, 'output': {}})
        os.replace(temporary, destination)
        write_json(sidecar, {**metadata, 'output': {'sha256': digest(destination), 'files': len(entries)}})
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
