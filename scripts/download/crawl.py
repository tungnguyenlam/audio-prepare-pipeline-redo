"""Discover, filter, deduplicate, and optionally download videos from a source collection."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (FileContractError, LoggingArgumentParser, ROOT,
                           positive_int, progress, read_json, safe_name,
                           write_json)
from playlist import entry_url, list_entries
from youtube import RateLimitAbort, add_download_arguments, download, raise_if_rate_limited


def arguments() -> LoggingArgumentParser:
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument('--source-file', type=Path, required=True,
                   help='JSON collection containing named playlist, channel, or yt-dlp search sources')
    p.add_argument('--source', action='append', default=None,
                   help='Only crawl this exact source name; repeat to select multiple sources')
    p.add_argument('--output-manifest', type=Path, default=None,
                   help='Candidate/audit JSON (default: .data/download/<collection>/crawl.json)')
    p.add_argument('--metadata-only', action='store_true',
                   help='Write the filtered crawl manifest without downloading audio')
    p.add_argument('--max-items', '--limit', dest='max_items', type=positive_int, default=None,
                   help='Maximum accepted unique videos across the selected sources')
    return add_download_arguments(p)


def _string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise FileContractError(f'{label} must be an array of non-empty strings')
    return value


def _compile_patterns(filters: dict, key: str, label: str) -> list[re.Pattern[str]]:
    patterns = []
    for expression in _string_list(filters.get(key), f'{label}.{key}'):
        try:
            patterns.append(re.compile(expression, re.IGNORECASE))
        except re.error as exc:
            raise FileContractError(f'Invalid regex in {label}.{key}: {expression!r}: {exc}') from exc
    return patterns


def _filters(global_filters: dict, source: dict) -> dict:
    local = source.get('filters') or {}
    if not isinstance(local, dict):
        raise FileContractError(f'Source {source["name"]!r} filters must be an object')
    label = f'source {source["name"]!r}.filters'
    include = _compile_patterns(global_filters, 'include_title_regex', 'filters')
    include += _compile_patterns(local, 'include_title_regex', label)
    exclude = _compile_patterns(global_filters, 'exclude_title_regex', 'filters')
    exclude += _compile_patterns(local, 'exclude_title_regex', label)
    excluded_ids = set(_string_list(global_filters.get('exclude_video_ids'), 'filters.exclude_video_ids'))
    excluded_ids.update(_string_list(local.get('exclude_video_ids'), f'{label}.exclude_video_ids'))
    return {
        'include': include,
        'exclude': exclude,
        'excluded_ids': excluded_ids,
        'min_duration_s': local.get('min_duration_s', global_filters.get('min_duration_s')),
        'max_duration_s': local.get('max_duration_s', global_filters.get('max_duration_s')),
        'exclude_live': local.get('exclude_live', global_filters.get('exclude_live', True)),
    }


def _rejection(entry: dict, filters: dict) -> str | None:
    video_id = str(entry.get('id') or '')
    title = str(entry.get('title') or video_id)
    if not video_id:
        return 'missing_video_id'
    if video_id in filters['excluded_ids']:
        return 'excluded_video_id'
    if filters['include'] and not any(pattern.search(title) for pattern in filters['include']):
        return 'title_did_not_match_include_regex'
    if any(pattern.search(title) for pattern in filters['exclude']):
        return 'title_matched_exclude_regex'
    if filters['exclude_live'] and (entry.get('is_live') or entry.get('live_status') in {'is_live', 'is_upcoming'}):
        return 'live_or_upcoming'
    duration = entry.get('duration')
    if duration is not None:
        duration = float(duration)
        if filters['min_duration_s'] is not None and duration < float(filters['min_duration_s']):
            return 'shorter_than_min_duration'
        if filters['max_duration_s'] is not None and duration > float(filters['max_duration_s']):
            return 'longer_than_max_duration'
    return None


def _candidate(entry: dict, source: dict) -> dict:
    return {
        'video_id': str(entry['id']),
        'title': entry.get('title') or str(entry['id']),
        'url': entry_url(entry),
        'duration_s': entry.get('duration'),
        'channel': entry.get('channel') or entry.get('uploader'),
        'channel_id': entry.get('channel_id') or entry.get('uploader_id'),
        'upload_date': entry.get('upload_date'),
        'live_status': entry.get('live_status'),
        'sources': [{
            'name': source['name'],
            'url': source['url'],
            'priority': source.get('priority'),
        }],
        'download': {'status': 'pending'},
    }


def _load_collection(path: Path, selected_names: list[str] | None) -> tuple[dict, list[dict]]:
    if not path.is_file():
        raise FileContractError(f'Source file not found: {path}')
    data = read_json(path)
    if data.get('schema_version') != 1:
        raise FileContractError('Source collection schema_version must be 1')
    if not isinstance(data.get('collection'), str) or not data['collection']:
        raise FileContractError('Source collection requires a non-empty collection name')
    if not isinstance(data.get('sources'), list) or not data['sources']:
        raise FileContractError('Source collection requires a non-empty sources array')
    names = set()
    sources = []
    for index, source in enumerate(data['sources']):
        if not isinstance(source, dict) or not isinstance(source.get('name'), str) or not source['name']:
            raise FileContractError(f'sources[{index}] requires a non-empty name')
        if not isinstance(source.get('url'), str) or not source['url']:
            raise FileContractError(f'Source {source["name"]!r} requires a non-empty URL')
        if not isinstance(source.get('priority', 1000), int):
            raise FileContractError(f'Source {source["name"]!r} priority must be an integer')
        if source.get('max_items') is not None and (
                not isinstance(source['max_items'], int) or source['max_items'] <= 0):
            raise FileContractError(f'Source {source["name"]!r} max_items must be a positive integer')
        if source['name'] in names:
            raise FileContractError(f'Duplicate source name: {source["name"]}')
        names.add(source['name'])
        if selected_names is None or source['name'] in selected_names:
            sources.append(source)
    if selected_names:
        missing = sorted(set(selected_names) - names)
        if missing:
            raise FileContractError(f'Unknown source name(s): {", ".join(missing)}')
    return data, sorted(sources, key=lambda item: item.get('priority', 1000))


def _download_candidates(candidates: list[dict], args, output_groups: dict[str, str]) -> int:
    total = len(candidates)
    lock = threading.Lock()

    def process(index: int, candidate: dict) -> int:
        raise_if_rate_limited()
        try:
            with lock:
                progress('ITEM_START', candidate['title'], current=index, total=total)
            source_info = {'id': candidate['video_id'], 'title': candidate['title'],
                           'webpage_url': candidate['url']}
            dest = download(candidate['url'], args, source_info=source_info,
                            output_group=output_groups.get(candidate['video_id']))
            candidate['download'] = {'status': 'complete', 'path': str(dest)}
            with lock:
                progress('ITEM_DONE', dest.name, current=index, total=total)
                print(dest, flush=True)
            return 0
        except RateLimitAbort as exc:
            candidate['download'] = {'status': 'failed', 'error': str(exc)}
            with lock:
                progress('RATE_LIMITED', str(exc), current=index, total=total)
            raise
        except Exception as exc:
            candidate['download'] = {'status': 'failed', 'error': str(exc)}
            with lock:
                progress('ITEM_FAIL', str(exc), current=index, total=total)
            return 1

    indexed = list(enumerate(candidates, 1))
    batches = [indexed[i:i + args.batch_size] for i in range(0, total, args.batch_size)]

    def process_batch(items: list[tuple[int, dict]]) -> int:
        return sum(process(index, candidate) for index, candidate in items)

    try:
        if args.concurrency > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batches))) as pool:
                return sum(pool.map(process_batch, batches))
        return sum(process_batch(items) for items in batches)
    except RateLimitAbort:
        for candidate in candidates:
            if candidate['download'].get('status') == 'pending':
                candidate['download'] = {
                    'status': 'failed',
                    'error': 'YouTube rate limiting aborted this run; remaining items were skipped',
                }
        raise


def main() -> int:
    parser = arguments()
    args = parser.parse_args()
    try:
        collection, sources = _load_collection(args.source_file, args.source)
        global_filters = collection.get('filters') or {}
        if not isinstance(global_filters, dict):
            raise FileContractError('filters must be an object')
        manifest_path = args.output_manifest
        if manifest_path is None:
            manifest_path = ROOT / '.data/download' / safe_name(collection['collection']) / 'crawl.json'
        manifest_path = manifest_path.resolve()
        if manifest_path.suffix.lower() != '.json':
            raise FileContractError('--output-manifest must have a .json suffix')

        accepted: dict[str, dict] = {}
        rejected = []
        source_results = []
        discovery_failed = 0
        duplicate_occurrences = 0
        truncated = False
        aborted: RateLimitAbort | None = None
        output_groups: dict[str, str] = {}
        for source in sources:
            if args.max_items is not None and len(accepted) >= args.max_items:
                truncated = True
                break
            per_source_limit = source.get('max_items')
            progress('SOURCE', f'{source["name"]}: {source["url"]}')
            try:
                target, raw_count, entries, output_group = list_entries(
                    source['url'], args, limit=per_source_limit)
                source_result = {'name': source['name'], 'url': source['url'],
                                 'resolved_url': target, 'listed': raw_count,
                                 'available': len(entries), 'resolved_name': output_group,
                                 'status': 'complete'}
                source_results.append(source_result)
            except RateLimitAbort as exc:
                discovery_failed += 1
                source_results.append({'name': source['name'], 'url': source['url'],
                                       'status': 'failed', 'error': str(exc)})
                progress('RATE_LIMITED', f'{source["name"]}: {exc}')
                aborted = exc
                break
            except Exception as exc:
                discovery_failed += 1
                source_results.append({'name': source['name'], 'url': source['url'],
                                       'status': 'failed', 'error': str(exc)})
                progress('SOURCE_FAIL', f'{source["name"]}: {exc}')
                continue

            active_filters = _filters(global_filters, source)
            for entry in entries:
                reason = _rejection(entry, active_filters)
                if reason:
                    rejected.append({'video_id': entry.get('id'), 'title': entry.get('title'),
                                     'url': entry_url(entry) if entry.get('id') or entry.get('url') else None,
                                     'source': source['name'], 'reason': reason})
                    continue
                video_id = str(entry['id'])
                if video_id in accepted:
                    duplicate_occurrences += 1
                    source_reference = {
                        'name': source['name'], 'url': source['url'],
                        'priority': source.get('priority')}
                    if source_reference not in accepted[video_id]['sources']:
                        accepted[video_id]['sources'].append(source_reference)
                    continue
                accepted[video_id] = _candidate(entry, source)
                output_groups[video_id] = output_group
                if args.max_items is not None and len(accepted) >= args.max_items:
                    truncated = True
                    break

        candidates = list(accepted.values())
        if args.metadata_only:
            for candidate in candidates:
                candidate['download'] = {'status': 'not_requested'}

        manifest = {
            'schema_version': 1,
            'operation': 'crawl',
            'collection': collection['collection'],
            'description': collection.get('description'),
            'source_file': str(args.source_file.resolve()),
            'filters': global_filters,
            'summary': {
                'sources_selected': len(sources),
                'sources_processed': len(source_results),
                'source_failures': discovery_failed,
                'accepted_unique': len(candidates),
                'duplicate_occurrences': duplicate_occurrences,
                'rejected_occurrences': len(rejected),
                'download_failures': 0,
                'metadata_only': args.metadata_only,
                'truncated': truncated,
            },
            'sources': source_results,
            'videos': candidates,
            'rejected': rejected,
        }
        write_json(manifest_path, manifest)

        download_failed = 0
        if aborted is None and not args.metadata_only and candidates:
            try:
                download_failed = _download_candidates(candidates, args, output_groups)
            except RateLimitAbort as exc:
                aborted = exc
                download_failed = sum(
                    1 for candidate in candidates if candidate['download'].get('status') == 'failed')
                progress('RATE_LIMITED', str(exc))
            manifest['summary']['download_failures'] = download_failed
            write_json(manifest_path, manifest)
        print(manifest_path)
        progress('CRAWL_COMPLETE',
                 f'{len(candidates)} accepted; {len(rejected)} rejected occurrences; '
                 f'{discovery_failed} source failures; {download_failed} download failures')
        return int(discovery_failed > 0 or download_failed > 0 or aborted is not None)
    except (FileContractError, ValueError) as exc:
        progress('ERROR', str(exc))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
