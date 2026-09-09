"""Private file contracts shared by standalone commands; no model orchestration."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unicodedata
import re

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('HF_HOME', str(ROOT / '.data/huggingface'))
AUDIO_SUFFIXES = {'.wav', '.mp3', '.flac', '.ogg', '.opus', '.m4a', '.aac', '.aiff', '.wma', '.mp4', '.webm'}


class FileContractError(ValueError):
    """An input or destination violates the command's file contract."""


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def log_config(title: str, args: argparse.Namespace | dict) -> None:
    if getattr(args, 'quiet', False):
        return
    name = title or (sys.argv[0] if sys.argv else 'command')
    if '/' in name or '\\' in name:
        name = Path(name).name
    items = vars(args) if isinstance(args, argparse.Namespace) else args
    border = '=' * 60
    lines = [border, f'[{name}] CONFIGURATION:']
    for k in sorted(items.keys()):
        if k.startswith('_'):
            continue
        val = items[k]
        if k == 'sample_rate' and val is None:
            val = 'None (preserve source)'
        elif k == 'output_dir' and val is None:
            val = 'None (dynamic per audio family)'
        lines.append(f'  {k:<24}: {val}')
    lines.append(border)
    print('\n'.join(lines), file=sys.stderr, flush=True)


class LoggingArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that logs parsed arguments to stderr upon successful parsing."""
    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args=args, namespace=namespace)
        log_config(self.prog or (sys.argv[0] if sys.argv else 'command'), ns)
        return ns


def progress(action: str, detail: str = '', *, current: int | None = None, total: int | None = None, elapsed_s: float | None = None) -> None:
    """Print real-time progress update to stderr."""
    items = []
    if current is not None and total is not None and total > 0:
        pct = (current / total) * 100.0
        items.append(f'[{current}/{total}] ({pct:5.1f}%)')
    elif current is not None:
        items.append(f'[{current}]')
    items.append(action.upper())
    if detail:
        items.append(detail)
    if elapsed_s is not None:
        items.append(f'({elapsed_s:.2f}s)')
    timestamp = time.strftime('%H:%M:%S')
    print(f'[{timestamp}] ' + ' : '.join(items), file=sys.stderr, flush=True)


def safe_name(value: str, limit: int | None = None, default: str = 'audio') -> str:
    """Sanitize string for filenames: strictly [a-zA-Z0-9_-], spaces -> hyphens, no dots."""
    if not value:
        return default
    # Map Vietnamese stroked-d characters before unicode decomposition
    value = value.replace('đ', 'd').replace('Đ', 'D')
    # Decompose unicode characters and strip combining diacritical marks
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    # Replace spaces, dots, and any non-alphanumeric character (except underscore and hyphen) with hyphen
    value = re.sub(r'[^a-zA-Z0-9_]+', '-', value)
    # Clean up awkward combinations of - and _ like -_ or _-
    value = re.sub(r'-*_-*', '_', value)
    value = re.sub(r'-+', '-', value)
    value = re.sub(r'_+', '_', value)
    value = value.strip('-_')
    if limit and len(value) > limit:
        value = value[:limit].rstrip('-_')
    return value or default


def family_name(video_id: str, title: str) -> str:
    """Return canonical family identifier: {safe_id}_{safe_title_10}."""
    return f"{safe_name(video_id)}_{safe_name(title, 10, default='video')}"


def family_audio_name(video_id: str, title: str, sample_rate: int | None = None, extension: str = '.wav') -> str:
    """Return standard download audio filename: {safe_id}_{safe_title_10}[-{sample_rate}].wav."""
    fam = family_name(video_id, title)
    rate_suffix = f"-{sample_rate}" if sample_rate else ""
    ext = extension if extension.startswith('.') else f".{extension}"
    return f"{fam}{rate_suffix}{ext}"


def infer_audio_family(path: Path) -> str:
    """Infer the audio family identifier from an audio or manifest path."""
    json_path = path if path.suffix.lower() == '.json' else path.with_suffix('.json')
    if json_path.is_file():
        try:
            data = read_json(json_path)
            src_info = data.get('source') or {}
            origin_info = src_info.get('origin') or data.get('origin') or {}
            vid = src_info.get('video_id') or origin_info.get('video_id')
            title = src_info.get('title') or origin_info.get('title')
            if vid:
                return family_name(str(vid), str(title or 'video'))
        except (ValueError, OSError):
            pass

    manifest_path = (path / 'segments.json') if path.is_dir() else (path.parent / 'segments.json')
    if manifest_path.is_file() and path != manifest_path:
        try:
            return infer_audio_family(manifest_path)
        except Exception:
            pass

    stem = path.stem
    if stem in {'segments', 'output', 'source', 'vocals', 'accompaniment', 'mixture'}:
        if path.parent.name and path.parent.name not in {'out', 'work', 'download', 'separate', 'diarize', 'verify'}:
            return path.parent.name

    # Check for <id>_<title10> pattern (e.g. 11-char YT id, or id_title10-<sample_rate>)
    m = re.match(r'^([a-zA-Z0-9_-]{11})_([a-zA-Z0-9-]{1,10})(?:-\d+)?(?:_.*)?$', stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.match(r'^([a-zA-Z0-9_-]+?)_([a-zA-Z0-9-]{1,10})-\d+(?:_.*)?$', stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"

    # Check for parent directory family pattern (e.g. 11-char YT id, or id_title10)
    if path.parent.name and path.parent.name not in {'out', 'work', 'download', 'separate', 'diarize', 'verify', 'audio', '.'}:
        m_parent = re.match(r'^([a-zA-Z0-9_-]{11})(?:_([a-zA-Z0-9-]{1,10}))?(?:_.*)?$', path.parent.name)
        if m_parent:
            return f"{m_parent.group(1)}_{m_parent.group(2)}" if m_parent.group(2) else m_parent.group(1)

    return safe_name(stem)


def resolve_output_dir(args: argparse.Namespace, src: Path) -> Path:
    if getattr(args, 'output_dir', None) is not None:
        return args.output_dir.resolve()
    base = getattr(args, '_default_base', None)
    if base is None:
        op = getattr(args, '_operation', 'audio')
        mod = getattr(args, '_model', None)
        base = ROOT / '.data' / op
        if mod:
            base /= mod
    family = infer_audio_family(src)
    return (base / family).resolve()


def parser(description: str, operation: str, model: str | None = None, *, segments: bool = False) -> LoggingArgumentParser:
    p = LoggingArgumentParser(description=description)
    p.add_argument('--input-file', type=Path)
    p.add_argument('--input-dir', type=Path)
    if not segments:
        p.add_argument('--output-file', type=Path)
    base = ROOT / '.data' / operation
    if model:
        base /= model
    p.add_argument('--output-dir', type=Path, default=None,
                   help='Output directory (default: dynamic per audio family under .data/<operation>/<model>/<family>)')
    p.add_argument('--work-dir', type=Path, default=base / 'work')
    p.add_argument('--overwrite', action='store_true')
    p.set_defaults(_operation=operation, _model=model, _default_base=base)
    return p


def inputs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    out = args.output_dir.resolve() if getattr(args, 'output_dir', None) is not None else None
    if args.input_file is not None:
        src = args.input_file.resolve()
        if not src.is_file():
            raise FileContractError(f'Input file not found: {src}')
        return [(src, Path(src.name))]
    if getattr(args, 'output_file', None) is not None:
        raise FileContractError('--output-file requires --input-file')
    if args.input_dir is None:
        raise FileContractError('Supply --input-file or --input-dir')
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise FileContractError(f'Input directory not found: {root}')
    if out is not None and root == out:
        raise FileContractError('Input and output roots must differ')
    return [(p, p.relative_to(root)) for p in sorted(root.rglob('*'))
            if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
            and (out is None or not p.resolve().is_relative_to(out))]


def destinations(args: argparse.Namespace, suffix: str = '', extension: str = '.wav') -> list[tuple[Path, Path]]:
    safe_parent = lambda rel: Path(*[safe_name(p) for p in rel.parent.parts]) if rel.parent.parts else Path('.')
    explicit_out = getattr(args, 'output_dir', None)
    if explicit_out is not None:
        out_root = explicit_out.resolve()
    elif getattr(args, 'input_dir', None) is not None:
        out_root = resolve_output_dir(args, args.input_dir)
    else:
        out_root = None

    pairs = []
    for src, rel in inputs(args):
        if getattr(args, 'output_file', None) is not None:
            dest = args.output_file.resolve()
        else:
            base_dir = out_root if out_root is not None else resolve_output_dir(args, src)
            dest = (base_dir / safe_parent(rel) / f'{safe_name(rel.stem)}{suffix}{extension}').resolve()
        pairs.append((src, dest))
    sources = {src.resolve() for src, _ in pairs}
    seen = set()
    for src, dest in pairs:
        if dest in sources:
            raise FileContractError(f'Cannot overwrite an input: {dest}')
        if dest in seen:
            raise FileContractError(f'Multiple inputs map to {dest}; use separate invocations or rename inputs')
        seen.add(dest)
    return pairs


def manifest_destinations(args: argparse.Namespace, filename: str = 'segments.json') -> list[tuple[Path, Path]]:
    safe_parent = lambda rel: Path(*[safe_name(p) for p in rel.parent.parts]) if rel.parent.parts else Path('.')
    explicit_out = getattr(args, 'output_dir', None)
    all_inputs = inputs(args)
    if explicit_out is not None:
        pairs = [(src, (explicit_out.resolve() / safe_parent(rel) / safe_name(rel.stem) / filename).resolve())
                 for src, rel in all_inputs]
    else:
        families = [infer_audio_family(src) for src, _ in all_inputs]
        has_multiple_same_family = len(families) != len(set(families))
        pairs = []
        for src, rel in all_inputs:
            base_dir = resolve_output_dir(args, src)
            if has_multiple_same_family:
                dest = (base_dir / safe_name(rel.stem) / filename).resolve()
            else:
                dest = (base_dir / filename).resolve()
            pairs.append((src, dest))
    seen = set()
    for src, dest in pairs:
        if dest in seen:
            raise FileContractError(f'Multiple inputs map to {dest}; use separate invocations or specify distinct output directories')
        seen.add(dest)
    return pairs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise FileContractError(f'Expected a JSON object: {path}')
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def identity(path: Path) -> dict:
    result = {'path': str(path.resolve()), 'sha256': digest(path)}
    sidecar = path.with_suffix('.json')
    if sidecar.is_file():
        try:
            metadata = read_json(sidecar)
            if metadata.get('output', {}).get('sha256') == result['sha256']:
                result['origin'] = metadata.get('source', {})
        except (ValueError, OSError):
            pass
    return result


def request(source: dict, operation: str, parameters: dict, model: str | None = None) -> dict:
    return {'schema_version': 1, 'source': source, 'operation': operation,
            'model': model, 'parameters': parameters}


def completed(dest: Path, wanted: dict, overwrite: bool) -> bool:
    sidecar = dest.with_suffix('.json')
    if dest == sidecar:
        raise FileContractError('Audio destination cannot have a .json suffix')
    if not dest.exists() and not sidecar.exists():
        return False
    try:
        old = read_json(sidecar)
    except (OSError, ValueError):
        old = {}
    matches = all(old.get(k) == v for k, v in wanted.items())
    if matches:
        if dest.is_file() and old.get('output', {}).get('sha256') == digest(dest):
            return True
        return False
    if not overwrite:
        raise FileContractError(f'Conflicting or unrecognized destination: {dest}; use --overwrite')
    return False


def probe(path: Path) -> dict:
    import soundfile as sf
    try:
        info = sf.info(str(path))
        return {'sample_rate': info.samplerate, 'channels': info.channels,
                'frames': info.frames, 'duration_s': info.duration, 'format': info.format}
    except (RuntimeError, sf.LibsndfileError):
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                                 '-show_entries', 'stream=sample_rate,channels,duration:format=duration,format_name',
                                 '-of', 'json', str(path)], check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        rate = int(stream['sample_rate'])
        duration = float(stream.get('duration') or data['format']['duration'])
        return {'sample_rate': rate, 'channels': int(stream['channels']),
                'frames': round(rate * duration), 'duration_s': duration, 'format': data['format']['format_name']}


def publish(staged: Path, dest: Path, metadata: dict, *, audio: bool = True) -> None:
    properties = probe(staged) if audio else {'format': dest.suffix.lstrip('.')}
    properties['sha256'] = digest(staged)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Install a matching incomplete record before the audio. A retry can recognize it.
    write_json(dest.with_suffix('.json'), {**metadata, 'output': {}})
    fd, name = tempfile.mkstemp(dir=dest.parent, suffix='.wav')
    os.close(fd)
    try:
        import shutil
        shutil.copyfile(staged, name)
        os.replace(name, dest)
        write_json(dest.with_suffix('.json'), {**metadata, 'output': properties})
    finally:
        Path(name).unlink(missing_ok=True)


def convert(src: Path, dest: Path, sample_rate: int, channels: int, *, start: float | None = None, end: float | None = None, floating: bool = False) -> None:
    if src.resolve() == dest.resolve():
        raise FileContractError('In-place audio writes are forbidden')
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src)]
    if start is not None:
        cmd += ['-ss', str(start)]
    if end is not None:
        cmd += ['-t', str(end - (start or 0))]
    cmd += ['-vn', '-ar', str(sample_rate), '-ac', str(channels), '-c:a', 'pcm_f32le' if floating else 'pcm_s16le', '-f', 'wav', str(dest)]
    subprocess.run(cmd, check=True, stdout=sys.stderr)


def batch(pairs: list[tuple[Path, Path]], process) -> int:
    total = len(pairs)
    if total == 0:
        progress('BATCH', '0 items to process')
        return 0
    failed = 0
    t_start = time.perf_counter()
    progress('BATCH', f'Starting batch processing of {total} item(s)')
    for idx, (src, dest) in enumerate(pairs, 1):
        item_start = time.perf_counter()
        progress('ITEM_START', f'{src.name} -> {dest.name}', current=idx, total=total)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                process(src, dest)
            elapsed = time.perf_counter() - item_start
            progress('ITEM_DONE', f'{src.name}', current=idx, total=total, elapsed_s=elapsed)
            print(dest, flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - item_start
            failed += 1
            progress('ITEM_FAIL', f'{src.name}: {exc}', current=idx, total=total, elapsed_s=elapsed)
    total_elapsed = time.perf_counter() - t_start
    progress('BATCH_COMPLETE', f'{total - failed} succeeded; {failed} failed', elapsed_s=total_elapsed)
    return int(failed > 0)
