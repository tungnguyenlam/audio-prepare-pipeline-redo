"""Package selected verifier results for offline audio and transcript review (HTML, CSV, XLSX)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile
from xml.etree import ElementTree as ET

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from _common.files import ROOT, LoggingArgumentParser, digest, progress, safe_name, write_json
from _verifier_export_inventory import collect_run, fingerprint

FIELDS = ('clip_id', 'audio_path', 'duration_s', 'verifier_status', 'reason',
          'transcript_status', 'transcript', 'emotion', 'review_status',
          'corrected_transcript', 'review_notes')
REVIEW_STATES = ('pending', 'approved', 'corrected', 'exclude')


def csv_value(value: object) -> object:
    if isinstance(value, str) and (value.lstrip().startswith(('=', '+', '-', '@')) or value.startswith(('\t', '\r', '\n'))):
        return "'" + value
    return value


def write_catalog(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: csv_value(row[key]) for key in FIELDS} for row in rows)


def write_excel(path: Path, rows: list[dict]) -> None:
    """Write a small OOXML workbook with text cells and relative audio hyperlinks.

    Uses stdlib ZIP/XML so export needs no additional package installation.
    """
    if len(rows) > 1048575:
        raise ValueError('Excel row limit exceeded; split the run before exporting')
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    pkgns = 'http://schemas.openxmlformats.org/package/2006/relationships'
    ET.register_namespace('', ns)
    ET.register_namespace('r', relns)
    def element(parent, name, attributes=None, text=None):
        result = ET.SubElement(parent, '{' + ns + '}' + name, attributes or {})
        result.text = text
        return result
    sheet = ET.Element('{' + ns + '}worksheet')
    views = element(sheet, 'sheetViews')
    view = element(views, 'sheetView', {'workbookViewId': '0'})
    element(view, 'pane', {'ySplit': '1', 'topLeftCell': 'A2', 'activePane': 'bottomLeft', 'state': 'frozen'})
    cols = element(sheet, 'cols')
    for i, key in enumerate(FIELDS, 1):
        element(cols, 'col', {'min': str(i), 'max': str(i), 'width': '60' if key in ('transcript', 'corrected_transcript', 'review_notes', 'audio_path', 'reason') else '24', 'customWidth': '1'})
    body = element(sheet, 'sheetData')
    for number, values in enumerate([dict(zip(FIELDS, FIELDS)), *rows], 1):
        row = element(body, 'row', {'r': str(number)})
        for col, field in enumerate(FIELDS):
            value = str(values[field])
            if len(value.encode('utf-16-le')) // 2 > 32767:
                raise ValueError('Excel cell text limit exceeded; shorten source metadata before exporting')
            if any((ord(char) < 32 and char not in '\t\n\r') or char in '\ufffe\uffff' for char in value):
                raise ValueError('Text contains control characters not supported in Excel')
            value = re.sub(r'_x[0-9A-Fa-f]{4}_', lambda match: '_x005F_' + match.group()[1:], value).replace('\r', '_x000D_')
            cell = element(row, 'c', {'r': f'{chr(65 + col)}{number}', 't': 'inlineStr', 's': '1' if number == 1 else '0'})
            element(element(cell, 'is'), 't', {'{http://www.w3.org/XML/1998/namespace}space': 'preserve'}, value)
    element(sheet, 'autoFilter', {'ref': f'A1:L{len(rows) + 1}'})
    validations = element(sheet, 'dataValidations', {'count': '1'})
    validation = element(validations, 'dataValidation', {'type': 'list', 'allowBlank': '0', 'showErrorMessage': '1', 'sqref': f'I2:I{len(rows) + 1}'})
    element(validation, 'formula1', text='"' + ','.join(REVIEW_STATES) + '"')
    links = element(sheet, 'hyperlinks') if any(row['audio_path'] for row in rows) else None
    rels = ET.Element('Relationships', xmlns=pkgns)
    for i, row in enumerate(rows, 2):
        if row['audio_path']:
            element(links, 'hyperlink', {'ref': f'B{i}', '{' + relns + '}id': f'audio{i}'})
            ET.SubElement(rels, 'Relationship', Id=f'audio{i}', Type=relns + '/hyperlink', Target=row['audio_path'], TargetMode='External')
    xml = lambda node: ET.tostring(node, encoding='utf-8', xml_declaration=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', '''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>''')
        archive.writestr('_rels/.rels', f'<Relationships xmlns="{pkgns}"><Relationship Id="rId1" Type="{relns}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{relns}"><sheets><sheet name="Transcript review" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships xmlns="{pkgns}"><Relationship Id="rId1" Type="{relns}/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="{relns}/styles" Target="styles.xml"/></Relationships>')
        archive.writestr('xl/styles.xml', f'<styleSheet xmlns="{ns}"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        archive.writestr('xl/worksheets/sheet1.xml', xml(sheet))
        archive.writestr('xl/worksheets/_rels/sheet1.xml.rels', xml(rels))


def main() -> int:
    parser = LoggingArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', '-id', type=Path, required=True, help='Verifier folder to search recursively; manifest selects its clips')
    parser.add_argument('--input-manifest', '-im', type=Path, action='append', default=[], help='Expected indexed audio or exported segments manifest; repeat for multiple families')
    parser.add_argument('--output-file', '-of', type=Path, help='Exact ZIP destination under .data/; default .data/s5-export/<dataset>_<version>.zip')
    parser.add_argument('--dataset-name', help='Delivery name; defaults to manifest family or input folder name')
    parser.add_argument('--configuration', help='Select one settings hash or unique prefix when clips have multiple results')
    parser.add_argument('--version', default='v1', help='Delivery version')
    parser.add_argument('--allow-partial', action='store_true', help='Explicitly deliver unresolved processing or unknown input coverage')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing ZIP')
    args = parser.parse_args()
    directory = args.input_dir.resolve()
    if not args.dataset_name:
        manifest = args.input_manifest[0].resolve() if args.input_manifest else None
        args.dataset_name = (manifest.parent.name if manifest and manifest.name == 'segments.json'
                             else manifest.stem if manifest else directory.name)
    default_name = safe_name(args.dataset_name + '_' + args.version) + '.zip'
    destination = (args.output_file or ROOT / '.data' / 's5-export' / default_name).resolve()
    if not directory.is_dir():
        parser.error('Input directory does not exist')
    if not destination.is_relative_to(ROOT / '.data') or destination.suffix.lower() != '.zip':
        parser.error('Output must be an exact .zip path under .data/, e.g. .data/s5-export/review_v1.zip. Omit --output-file to use ' + str(ROOT / '.data' / 's5-export' / default_name))
    if destination.is_relative_to(directory):
        parser.error('Output must be outside the verifier run')
    if destination.exists() and not args.overwrite:
        parser.error(f'Output already exists: {destination}. Use --version v2, --output-file, or --overwrite.')
    try:
        progress('HANDOFF_OUTPUT', str(destination))
        progress('HANDOFF_SCAN', 'Checking verifier artifacts and expected inputs')
        rows, summary = collect_run(directory, args.input_manifest, args.configuration,
                                    progress_tag='HANDOFF_SELECT')
        summary['human_review_status'] = 'pending'
        for row in rows:
            row.update(review_status='pending', corrected_transcript='', review_notes='')
            if '_source' in row:
                folder = 'passed' if row['verifier_status'] == 'pass' else 'needs_attention'
                row['audio_path'] = f"audio/{folder}/{row['clip_id']}{row['_source'].suffix.lower()}"
        if summary['processing_status'] != 'complete' and not args.allow_partial:
            raise ValueError(f"Cannot label this delivery complete: coverage={summary['coverage']}; outcomes={summary['counts']}; unresolved={summary['unresolved_clips']}. Provide the complete expected --input-manifest and resolve those inputs, or add --allow-partial for an explicitly partial review package.")
        summary.update(dataset_name=args.dataset_name, version=args.version,
                       created_at=datetime.now(timezone.utc).isoformat())
        package_name = safe_name(args.dataset_name + '_' + args.version)
        if summary['processing_status'] == 'partial':
            package_name += '_PARTIAL'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='handoff-', dir=destination.parent) as temp:
            root = Path(temp) / package_name
            support = root / 'supporting_files'
            support.mkdir(parents=True)
            for row in rows:
                source = row.pop('_source', None)
                if source:
                    target = root / row['audio_path']
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    if digest(target) != row['source_sha256']:
                        raise ValueError('Audio changed during export; retry with a stable run')
            write_catalog(root / 'clip_catalog.csv', rows)
            write_excel(root / 'clip_catalog.xlsx', rows)
            write_catalog(support / 'needs_attention.csv', [row for row in rows if row['verifier_status'] != 'pass' or row['transcript_status'] != 'provided'])
            release_id = fingerprint({'summary': summary, 'clips': rows})
            payload = {'release_id': release_id, 'fields': FIELDS, 'summary': summary, 'clips': rows}
            template = Path(__file__).with_name('verifier_handoff_review.html').read_text(encoding='utf-8')
            encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
            (root / 'START_HERE.html').write_text(template.replace('__DELIVERY_DATA__', encoded), encoding='utf-8')
            summary_text = (f"{args.dataset_name} — {args.version}\n"
                            f"Verifier processing: {summary['processing_status'].upper()}\n"
                            f"Input coverage: {summary['coverage']}\nHuman transcript review: PENDING\n"
                            f"Clips: {len(rows)}; audio files: {summary['audio_files']}; known hours: {summary['duration_s'] / 3600:.3f}\n"
                            f"Verdicts: {json.dumps(summary['counts'])}\n"
                            f"Needs transcription: {summary['needs_transcription']}\n"
                            f"Verifier configurations: {len(summary['configurations'])} (see release_manifest.json)\n"
                            f"Clips with unknown duration: {summary['unknown_duration_clips']}\n"
                            f"Profiles: {', '.join(summary['schema_profiles'])}\n")
            (support / 'delivery_summary.txt').write_text(summary_text, encoding='utf-8')
            (root / 'READ_ME.txt').write_text(summary_text + '''
1. Extract the entire ZIP; keep the folder structure together.
2. Open START_HERE.html in a browser to listen and review, or open clip_catalog.xlsx
   in Excel. CSV is also provided with the same relative audio paths and transcripts.
3. Choose ONE editing method. Return the saved Excel/CSV file, or use the HTML
   Download review CSV and Save progress JSON buttons. HTML edits stay in memory
   until downloaded; load the saved JSON to resume. Excel/CSV edits do not sync to
   HTML. Keep the original clip_id and audio_path columns unchanged.

Review each clip: set approved if the original transcript is correct; corrected
if you entered a complete replacement in corrected_transcript; exclude if the
clip should not be used. Put explanations in review_notes. Blank transcripts mean
needs transcription, not silence. Missing emotion means not provided.

All transcripts and labels are model-produced, not human-approved. A complete
verifier run may contain valid rejections. Human review starts pending regardless
of verifier status. Narrow verifier profiles do not guarantee full acoustic QC.
Rejected and unresolved clips are in audio/needs_attention where verified audio
is available. They must not be treated as approved training data. A reviewer
opinion does not repair failed verifier processing or change the original verdict.
The sender must resolve processing failures separately and issue a new version.

Audio is copied unchanged. Browser playback depends on the original format;
use the relative audio link in your system player if the browser cannot play it.
Excel links may show the normal application confirmation before opening audio.
Keep XLSX/CSV at the package root so relative audio paths remain valid.
CSV text starting with formula characters is prefixed with an apostrophe for
spreadsheet safety. XLSX and release_manifest.json preserve original text.
Checksums describe the delivered originals, not subsequently edited review files.
''', encoding='utf-8')
            write_json(support / 'release_manifest.json', {'schema_version': 1, **payload})
            files = sorted(path for path in root.rglob('*') if path.is_file())
            (support / 'checksums.sha256').write_text(''.join(f'{digest(path)}  {path.relative_to(root).as_posix()}\n' for path in files), encoding='utf-8')
            temporary_zip = Path(temp) / 'delivery.zip'
            with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(root.rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(Path(temp)).as_posix())
            if args.overwrite:
                os.replace(temporary_zip, destination)
            else:
                os.link(temporary_zip, destination)
        progress('HANDOFF_DONE', f"{summary['processing_status']}; {len(rows)} clips; human review pending")
        print(destination)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
