"""Generate standalone interactive HTML report for ViCut-Bench inspection.

Provides an audio-enabled dashboard for:
  1. Edge Screen Verification (30 clean VIVOS anchors vs 20 screen-flagged PhoAudiobook clips)
  2. Defect Synthesis AB Audition (paired clean twin vs synthesized defects across severities)
  3. Research State & Benchmark Metrics
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import LoggingArgumentParser, persist_path, progress, resolve_stored_path

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def select_audit_candidates(measured_path: Path, anchors_path: Path, seed: int = 42) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    anchors = load_jsonl(anchors_path)
    measured = load_jsonl(measured_path)
    stop_anchors = [r for r in anchors if r.get("is_stop_coda")]
    open_anchors = [r for r in anchors if not r.get("is_stop_coda")]
    rng.shuffle(stop_anchors)
    rng.shuffle(open_anchors)
    clean_vivos = stop_anchors[:15] + open_anchors[:15]
    flagged_pho = [r for r in measured if r.get("dataset") == "phoaudiobook" and r.get("screen_flags")]
    rng.shuffle(flagged_pho)
    flagged_sample = flagged_pho[:20]
    return clean_vivos, flagged_sample

def format_audio_src(raw_rel_path: str, report_dir: Path) -> str:
    if "anchors/" in raw_rel_path:
        part = raw_rel_path.split("anchors/")[-1]
        return f"../anchors/{part}"
    if "defects/" in raw_rel_path:
        part = raw_rel_path.split("defects/")[-1]
        return f"../defects/{part}"
    if raw_rel_path.startswith(".data/vicut_bench/"):
        return raw_rel_path.replace(".data/vicut_bench/", "../")
    if raw_rel_path.startswith("vivos/"):
        return f"../anchors/{raw_rel_path}"
    if raw_rel_path.startswith("phoaudiobook/"):
        return f"../anchors/{raw_rel_path}"
    return raw_rel_path

def build_card(category: str, r: dict, report_dir: Path) -> str:
    is_clean = category == "clean"
    audio_src = format_audio_src(r.get("rel_path", ""), report_dir)
    clip_id = r.get("id", "")
    dataset = r.get("dataset", "")
    speaker = r.get("speaker", "")
    text = html.escape(r.get("text", ""))
    final_word = html.escape(r.get("final_word", ""))
    is_stop = r.get("is_stop_coda", False)
    flags = r.get("screen_flags", [])
    duration = r.get("duration_s", 0.0)
    lead_silence = r.get("lead_silence_s", 0.0) * 1000.0
    trail_silence = r.get("trail_silence_s", 0.0) * 1000.0
    lead_zero = r.get("lead_zero_s", 0.0) * 1000.0
    trail_zero = r.get("trail_zero_s", 0.0) * 1000.0
    snr = r.get("snr_db", 0.0)
    badge_class = "badge-clean" if is_clean else "badge-flagged"
    badge_text = "Clean Anchor" if is_clean else "Screen Rejected"
    flags_html = "".join([f"<span class='flag-pill'>{f}</span>" for f in flags])
    coda_html = "<span class='coda-pill stop'>Stop Coda (-p/-t/-c/-ch)</span>" if is_stop else "<span class='coda-pill open'>Open/Nasal Coda</span>"
    zero_note = f" <span class='zero-warn'>({lead_zero:.0f}ms zero pad)</span>" if lead_zero > 0 else ""
    trail_zero_note = f" <span class='zero-warn'>({trail_zero:.0f}ms zero pad)</span>" if trail_zero > 0 else ""
    flag_div = f"<div class='flags-row'><span class='flags-title'>Detection Flags:</span> {flags_html}</div>" if flags else ""
    item_class = "clean-item" if is_clean else "flagged-item"
    coda_class = "stop-coda" if is_stop else "open-coda"
    return f"""<div class="card edge-card {item_class} {coda_class}">
    <div class="card-header">
        <div class="card-title-group">
            <span class="badge {badge_class}">{badge_text}</span>
            <span class="source-tag">{dataset.upper()} &bull; {speaker}</span>
        </div>
        <span class="duration-tag">{duration:.2f}s</span>
    </div>
    <div class="card-body">
        <div class="transcript-box">
            <p class="transcript">&ldquo;{text}&rdquo;</p>
            <div class="word-tags">
                <span class="final-word-label">Final Word: <strong>{final_word}</strong></span>
                {coda_html}
            </div>
        </div>
        <div class="metrics-grid">
            <div class="metric-item">
                <span class="metric-label">Lead Silence</span>
                <span class="metric-val">{lead_silence:.1f} ms{zero_note}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Trail Silence</span>
                <span class="metric-val">{trail_silence:.1f} ms{trail_zero_note}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Envelope SNR</span>
                <span class="metric-val">{snr:.1f} dB</span>
            </div>
        </div>
        {flag_div}
        <div class="audio-control-wrap">
            <audio controls preload="none" src="{audio_src}" class="custom-audio"></audio>
        </div>
    </div>
    <div class="card-footer">
        <span class="clip-id">{clip_id}</span>
    </div>
</div>"""

def build_defect_card(d: dict, report_dir: Path) -> str:
    defect_type = d.get("defect_type", "")
    severity = d.get("severity", "")
    clip_id = d.get("id", "")
    anchor_id = d.get("anchor_id", "")
    defect_src = format_audio_src(d.get("audio_path", ""), report_dir)
    twin_src = format_audio_src(d.get("twin_audio_path", ""), report_dir)
    text = html.escape(d.get("transcript", ""))
    final_word = html.escape(d.get("final_word", "")) if "final_word" in d else ""
    speaker = d.get("speaker", "")
    if defect_type == "clipped_word_end":
        desc = f"Truncated <strong>{severity}</strong> before measured speech offset (final word: <em>{final_word}</em>)."
        sev_class = "sev-end"
    elif defect_type == "clipped_word_start":
        desc = f"Truncated <strong>{severity}</strong> after measured speech onset."
        sev_class = "sev-start"
    elif defect_type == "secondary_speaker":
        desc = f"Interfering speaker (<em>{d.get('donor_speaker')}</em>) overlaid at <strong>{severity}</strong>."
        sev_class = "sev-spk"
    elif defect_type == "background_music":
        desc = f"Instrumental background music mixed at <strong>{severity}</strong>."
        sev_class = "sev-music"
    else:
        desc = f"Severity: <strong>{severity}</strong>"
        sev_class = "sev-other"
    title_defect = defect_type.replace("_", " ").upper()
    return f"""<div class="card defect-card" data-defect-type="{defect_type}" data-severity="{severity}">
    <div class="card-header">
        <div class="card-title-group">
            <span class="badge badge-defect {sev_class}">{title_defect}</span>
            <span class="severity-pill">{severity}</span>
        </div>
        <span class="source-tag">{speaker}</span>
    </div>
    <div class="card-body">
        <div class="defect-desc-box">
            <p class="defect-desc">{desc}</p>
        </div>
        <div class="transcript-box">
            <p class="transcript">&ldquo;{text}&rdquo;</p>
        </div>
        <div class="ab-compare-container">
            <div class="ab-player-lane clean-lane">
                <div class="lane-header">
                    <span class="lane-title">A &bull; Clean Twin</span>
                </div>
                <audio controls preload="none" src="{twin_src}" class="custom-audio ab-audio clean-audio"></audio>
            </div>
            <div class="ab-player-lane defect-lane">
                <div class="lane-header">
                    <span class="lane-title">B &bull; Defect Clip</span>
                </div>
                <audio controls preload="none" src="{defect_src}" class="custom-audio ab-audio defect-audio"></audio>
            </div>
        </div>
    </div>
    <div class="card-footer">
        <span class="clip-id">{clip_id} &bull; twin: {anchor_id}</span>
    </div>
</div>"""

def main() -> int:
    default_measured = Path(".data/vicut_bench/anchors/measured_candidates.jsonl")
    default_anchors = Path(".data/vicut_bench/anchors/vicut_bench_anchors_300.jsonl")
    default_defects = Path(".data/vicut_bench/defects/vicut_bench_defects.jsonl")
    default_report = Path(".data/vicut_bench/reports/vicut_bench_dashboard.html")
    p = LoggingArgumentParser(description=__doc__)
    p.add_argument("--measured-manifest", type=Path, default=default_measured)
    p.add_argument("--anchors-manifest", type=Path, default=default_anchors)
    p.add_argument("--defects-manifest", type=Path, default=default_defects)
    p.add_argument("-or", "--output-report", type=Path, default=default_report)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    measured_path = resolve_stored_path(args.measured_manifest)
    anchors_path = resolve_stored_path(args.anchors_manifest)
    defects_path = resolve_stored_path(args.defects_manifest)
    report_path = resolve_stored_path(args.output_report)
    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    clean_vivos, flagged_pho = select_audit_candidates(measured_path, anchors_path, args.seed)
    defects = load_jsonl(defects_path)
    edge_cards = [build_card("clean", r, report_dir) for r in clean_vivos] + [build_card("flagged", r, report_dir) for r in flagged_pho]
    defect_cards = [build_defect_card(d, report_dir) for d in defects[:80]]
    edge_joined = "\n".join(edge_cards)
    defect_joined = "\n".join(defect_cards) if defect_cards else "<p style='padding: 30px; color: #9ca3af;'>No defects synthesized yet.</p>"
    clean_count = len(clean_vivos)
    flagged_count = len(flagged_pho)
    total_edge_count = clean_count + flagged_count
    raw_template = Path(__file__).resolve().parent / "report_template.html"
    template_text = raw_template.read_text(encoding="utf-8")
    final_html = template_text.replace("{{EDGE_CARDS}}", edge_joined)
    final_html = final_html.replace("{{DEFECT_CARDS}}", defect_joined)
    final_html = final_html.replace("{{TOTAL_EDGE_COUNT}}", str(total_edge_count))
    final_html = final_html.replace("{{CLEAN_COUNT}}", str(clean_count))
    final_html = final_html.replace("{{FLAGGED_COUNT}}", str(flagged_count))
    report_path.write_text(final_html, encoding="utf-8")
    progress("COMPLETE", f"Saved interactive HTML report to {report_path}")
    print(persist_path(report_path))
    return 0

if __name__ == "__main__":
    sys.exit(main())
