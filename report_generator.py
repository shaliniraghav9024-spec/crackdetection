"""
Generate stand-alone HTML and PDF defect-inspection reports.

The reports combine:

* the per-class defect summary (severity, recommendation, where it
  appeared in the image / when in the video),
* an annotated keyframe / annotated photo,
* a transcript of any voice commentary, with timestamps and
  highlighted defect-keyword mentions.

HTML is always produced.  PDF is produced via ReportLab when available
and is the canonical "downloadable report".
"""

from __future__ import annotations

import base64
import datetime as dt
import html
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from defect_analyzer import ImageReport, VideoReport, SEVERITY
from audio_transcriber import TranscriptResult, Segment


def match_transcript_to_defect(
    transcript: TranscriptResult | None,
    defect_label: str,
    time_sec: float | None = None,
    window_sec: float = 4.0,
) -> list[Segment]:
    """Return the transcript segments most relevant to a given defect.

    Two matching strategies are combined:

    1. **Keyword match** -- segments whose ``detected_defects`` field
       contains ``defect_label`` (regardless of timestamp).
    2. **Time match** -- if ``time_sec`` is provided, any segment that
       overlaps a ``+/- window_sec`` window around it is included so the
       voice commentary aligned with the defect frame is captured even
       when the inspector didn't say the defect's exact name.
    """
    if transcript is None or not transcript.segments:
        return []

    matches: dict[tuple[float, float], Segment] = {}
    for seg in transcript.segments:
        if defect_label in seg.detected_defects:
            matches[(seg.start, seg.end)] = seg

    if time_sec is not None:
        lo = time_sec - window_sec
        hi = time_sec + window_sec
        for seg in transcript.segments:
            if seg.end >= lo and seg.start <= hi:
                matches[(seg.start, seg.end)] = seg

    return sorted(matches.values(), key=lambda s: s.start)


SEVERITY_COLORS = {
    "High":   "#d9534f",
    "Medium": "#f0ad4e",
    "Low":    "#5bc0de",
    "None":   "#6c757d",
    "Unknown": "#6c757d",
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_time(sec: float) -> str:
    if sec is None:
        return "-"
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def _img_to_base64(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ----------------------------------------------------------------------- HTML

_HTML_STYLE = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  margin: 0; padding: 32px; background: #f5f7fa; color: #1f2933;
}
.container { max-width: 980px; margin: 0 auto; background: #fff;
  border-radius: 12px; padding: 32px; box-shadow: 0 4px 16px rgba(0,0,0,.06); }
h1 { margin-top: 0; font-size: 28px; }
h2 { border-bottom: 2px solid #e4e7eb; padding-bottom: 6px; margin-top: 32px; }
.meta { color: #52606d; font-size: 14px; }
.summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px; margin: 24px 0; }
.card { background: #f8fafc; border-radius: 10px; padding: 16px;
  border-left: 4px solid #3d5afe; }
.card .num { font-size: 28px; font-weight: 700; }
.card .lbl { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
  color: #52606d; }
.defect-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
.defect-table th, .defect-table td { padding: 10px 12px; text-align: left;
  border-bottom: 1px solid #e4e7eb; vertical-align: top; }
.defect-table th { background: #f0f4f8; font-size: 13px; }
.severity-pill { display: inline-block; padding: 3px 10px; border-radius: 999px;
  color: #fff; font-size: 12px; font-weight: 600; }
.confbar { background: #e4e7eb; border-radius: 6px; height: 8px; overflow: hidden;
  width: 120px; }
.confbar > span { display: block; height: 100%; background: #3d5afe; }
.transcript { background: #fafbfc; border-radius: 8px; padding: 14px;
  border: 1px solid #e4e7eb; max-height: 360px; overflow-y: auto;
  font-family: 'SFMono-Regular', Menlo, monospace; font-size: 13px; }
.seg { margin: 4px 0; }
.seg .ts { color: #3d5afe; font-weight: 600; margin-right: 8px; }
.seg.has-defect { background: #fff8e1; padding: 4px 6px; border-radius: 4px; }
.kw { background: #ffd54f; padding: 0 3px; border-radius: 3px; }
img.snap { max-width: 100%; border-radius: 10px; border: 1px solid #e4e7eb; }
.empty { color: #829ab1; font-style: italic; }
.finding-card { display: grid; grid-template-columns: 1fr 1.2fr; gap: 18px;
  padding: 16px; border: 1px solid #e4e7eb; border-radius: 10px;
  background: #fafbfc; margin: 14px 0; }
.finding-card h3 { margin: 0 0 8px; }
.finding-card h4 { margin: 14px 0 6px; font-size: 14px; color: #3d5afe; }
.finding-img img.snap { width: 100%; }
@media (max-width: 760px) {
  .finding-card { grid-template-columns: 1fr; }
}
.footer { color: #829ab1; font-size: 12px; text-align: center; margin-top: 32px; }
"""


def _highlight_keywords(text: str, keywords: list[str]) -> str:
    safe = html.escape(text)
    for kw in sorted(set(keywords), key=len, reverse=True):
        if not kw:
            continue
        safe = safe.replace(html.escape(kw),
                            f"<span class='kw'>{html.escape(kw)}</span>")
    return safe


def _defect_table_html(defects: list[dict], for_video: bool) -> str:
    if not defects:
        return "<p class='empty'>No defects detected above the confidence threshold.</p>"
    cols = ["Defect", "Severity", "Confidence", "Where", "Recommendation"]
    rows: list[str] = []
    for d in defects:
        sev = d.get("severity", "Unknown")
        sev_color = SEVERITY_COLORS.get(sev, "#6c757d")
        conf = d.get("max_confidence", 0.0)
        if for_video:
            place_count = int(d.get("place_count", 1) or 1)
            where = (f"{place_count} place(s) · "
                     f"first @ {_fmt_time(d.get('first_time_sec', 0))}, "
                     f"last @ {_fmt_time(d.get('last_time_sec', 0))} "
                     f"({d.get('count', 0)} frames)")
        else:
            where = f"{d.get('count', 0)} region(s)"
        rows.append(f"""
        <tr>
          <td><strong>{html.escape(d['label'])}</strong></td>
          <td><span class='severity-pill' style='background:{sev_color}'>{sev}</span></td>
          <td>
            <div>{_fmt_pct(conf)}</div>
            <div class='confbar'><span style='width:{int(conf*100)}%'></span></div>
          </td>
          <td>{html.escape(where)}</td>
          <td style='font-size:13px'>{html.escape(d.get('recommendation', ''))}</td>
        </tr>
        """)
    head = "".join(f"<th>{c}</th>" for c in cols)
    return (f"<table class='defect-table'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def _findings_html(
    defects: list[dict],
    transcript: TranscriptResult | None,
    is_video: bool,
    image_crops: list[dict] | None,
) -> str:
    if not defects:
        return ""

    crop_lookup: dict[str, dict] = {}
    if image_crops:
        for c in image_crops:
            crop_lookup[c["label"]] = c

    panels: list[str] = []
    for d in defects:
        label = d["label"]
        sev = d.get("severity", "Unknown")
        sev_color = SEVERITY_COLORS.get(sev, "#6c757d")
        conf = d.get("max_confidence", 0.0)

        if is_video:
            kf = d.get("keyframe") or {}
            img_path = kf.get("image_path")
            ts = kf.get("time_sec")
            place_count = int(d.get("place_count", 1) or 1)
            where = (
                f"{place_count} place(s) · first @ {_fmt_time(d.get('first_time_sec', 0))}, "
                f"last @ {_fmt_time(d.get('last_time_sec', 0))} "
                f"({d.get('count', 0)} frames)"
            )
        else:
            crop = crop_lookup.get(label, {})
            img_path = crop.get("image_path")
            ts = None
            where = f"{d.get('count', 0)} region(s) in image"

        b64 = _img_to_base64(img_path)
        img_html = (f"<img class='snap' src='{b64}' alt='{html.escape(label)}' />"
                    if b64 else "<p class='empty'>No marked image available.</p>")

        matched: list[Segment] = match_transcript_to_defect(
            transcript, label, time_sec=ts,
        )
        if matched:
            seg_lines = []
            for s in matched:
                kws = [k.replace("_", " ") for k in s.detected_defects]
                seg_lines.append(
                    f"<div class='seg has-defect'>"
                    f"<span class='ts'>[{_fmt_time(s.start)}]</span>"
                    f"{_highlight_keywords(s.text, kws)}"
                    f"</div>"
                )
            transcript_html = (
                "<h4>Voice commentary near this defect</h4>"
                f"<div class='transcript'>{''.join(seg_lines)}</div>"
            )
        elif transcript is not None and transcript.has_audio and not transcript.error:
            transcript_html = ("<h4>Voice commentary near this defect</h4>"
                               "<p class='empty'>Inspector did not comment on "
                               "this defect.</p>")
        else:
            transcript_html = ""

        panels.append(f"""
        <div class='finding-card'>
          <div class='finding-img'>{img_html}</div>
          <div class='finding-body'>
            <h3>{html.escape(label)}
              <span class='severity-pill' style='background:{sev_color};margin-left:8px'>
                {sev}
              </span>
            </h3>
            <p><strong>Confidence:</strong> {_fmt_pct(conf)}<br>
               <strong>Where:</strong> {html.escape(where)}</p>
            <p><strong>Recommendation:</strong>
               {html.escape(d.get('recommendation', ''))}</p>
            {_place_gallery_html(d.get("places") or d.get("place_crops") or [])}
            {transcript_html}
          </div>
        </div>
        """)

    return "<h2>Defect Findings (per-defect detail)</h2>" + "".join(panels)


def _place_gallery_html(place_crops: list[dict], title: str = "Detected places") -> str:
    if len(place_crops) <= 1:
        return ""
    cards: list[str] = []
    for p in place_crops:
        b64 = _img_to_base64(p.get("image_path"))
        if not b64:
            continue
        place_label = f"Place {int(p.get('index', 0) or 0)}"
        if p.get("time_sec") is not None:
            place_label += f" @ {_fmt_time(p.get('time_sec'))}"
        cards.append(f"""
        <div class='card'>
          <div class='lbl'>{html.escape(place_label)}</div>
          <img class='snap' src='{b64}' alt='place crop' />
          <div class='lbl' style='margin-top:6px'>conf {_fmt_pct(p.get('confidence', 0.0))}</div>
        </div>
        """)
    if not cards:
        return ""
    return (
        f"<h4>{html.escape(title)}</h4>"
        "<div class='summary-cards'>" + "".join(cards) + "</div>"
    )


def _unconfirmed_voice_mentions_html(mentions: list[dict]) -> str:
    if not mentions:
        return ""

    cards: list[str] = []
    for m in mentions:
        label = m.get("label", "unknown")
        kf = m.get("keyframe") or {}
        b64 = _img_to_base64(kf.get("image_path"))
        ts = kf.get("time_sec")
        img_html = (f"<img class='snap' src='{b64}' alt='{html.escape(label)}' />"
                    if b64 else "<p class='empty'>No nearby review frame available.</p>")
        quotes = m.get("voice_quotes") or []
        quotes_html = "".join(
            f"<div class='seg has-defect'>{html.escape(q)}</div>"
            for q in quotes
        ) or "<p class='empty'>No transcript excerpt saved.</p>"
        cards.append(f"""
        <div class='finding-card'>
          <div class='finding-img'>{img_html}</div>
          <div class='finding-body'>
            <h3>{html.escape(label)}
              <span class='severity-pill'
                    style='background:{SEVERITY_COLORS["None"]};margin-left:8px'>
                not visually confirmed
              </span>
            </h3>
            <p><strong>Mentioned near:</strong> {_fmt_time(ts) if ts is not None else '-'}</p>
            <p class='empty'>
              Mentioned by the inspector in audio, but not counted as a
              detected defect because the image model did not confirm it.
            </p>
            <h4>Transcript excerpt</h4>
            <div class='transcript'>{quotes_html}</div>
          </div>
        </div>
        """)
    return (
        "<h2>Voice Mentions Not Visually Confirmed</h2>"
        "<p class='empty'>These items are shown for manual review only.</p>"
        + "".join(cards)
    )


def _transcript_html(transcript: TranscriptResult | None) -> str:
    if transcript is None:
        return ""
    if transcript.error:
        return (
            "<h2>Voice Transcript</h2>"
            f"<p class='empty'>{html.escape(transcript.error)}</p>"
        )
    if not transcript.has_audio:
        return ""
    if not transcript.segments:
        return (
            "<h2>Voice Transcript</h2>"
            "<p class='empty'>Audio track found, but no speech was transcribed.</p>"
        )

    segs: list[str] = []
    for seg in transcript.segments:
        kws = [k.replace("_", " ") for k in seg.detected_defects]
        css = "seg has-defect" if seg.detected_defects else "seg"
        segs.append(
            f"<div class='{css}'>"
            f"<span class='ts'>[{_fmt_time(seg.start)}]</span>"
            f"{_highlight_keywords(seg.text, kws)}"
            f"</div>"
        )
    return (
        "<h2>Voice Transcript</h2>"
        f"<div class='transcript'>{''.join(segs)}</div>"
    )


def render_html_report(
    title: str,
    report: ImageReport | VideoReport,
    transcript: TranscriptResult | None = None,
    annotated_image_path: str | None = None,
    image_crops: list[dict] | None = None,
) -> str:
    is_video = isinstance(report, VideoReport)
    defects = report.detected_defects
    unconfirmed_voice_mentions = (
        list(getattr(report, "unconfirmed_voice_mentions", []))
        if is_video else []
    )
    high = sum(1 for d in defects if d.get("severity") == "High")
    medium = sum(1 for d in defects if d.get("severity") == "Medium")
    low = sum(1 for d in defects if d.get("severity") == "Low")

    snap_b64 = _img_to_base64(annotated_image_path)

    media_summary: list[tuple[str, str]] = []
    media_summary.append(("File", html.escape(Path(report.source_path).name)))
    media_summary.append(("Resolution", f"{report.width}×{report.height}"))
    if is_video:
        v: VideoReport = report  # type: ignore[assignment]
        media_summary.extend([
            ("Duration", _fmt_time(v.duration_sec)),
            ("FPS", f"{v.fps:.2f}"),
            ("Frames sampled",
             f"{v.sampled_frames} / {v.total_frames}"),
        ])
    media_summary.append(("Analysis time", f"{report.elapsed_sec:.1f} s"))

    media_html = "".join(
        f"<tr><td><strong>{html.escape(k)}</strong></td>"
        f"<td>{html.escape(v)}</td></tr>"
        for k, v in media_summary
    )

    keyframes_html = ""
    if is_video:
        v = report  # type: ignore[assignment]
        kfs = [k for k in v.keyframe_paths if k.get("image_path")]
        if kfs:
            cards = []
            for kf in kfs:
                b64 = _img_to_base64(kf["image_path"])
                if not b64:
                    continue
                cards.append(f"""
                <div class='card' style='border-left-color:{SEVERITY_COLORS.get(SEVERITY.get(kf['label'],''), '#3d5afe')}'>
                  <div class='lbl'>{html.escape(kf['label'])} @ {_fmt_time(kf['time_sec'])}</div>
                  <img class='snap' src='{b64}' alt='{html.escape(kf['label'])}' />
                  <div class='lbl' style='margin-top:6px'>conf {_fmt_pct(kf['confidence'])}</div>
                </div>
                """)
            if cards:
                keyframes_html = (
                    "<h2>Keyframes</h2>"
                    "<div class='summary-cards'>" + "".join(cards) + "</div>"
                )

    snap_html = ""
    if snap_b64:
        snap_html = (
            "<h2>Annotated Image</h2>"
            f"<img class='snap' src='{snap_b64}' alt='annotated' />"
        )

    body = f"""
    <div class="container">
      <h1>{html.escape(title)}</h1>
      <p class="meta">Generated on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

      <div class="summary-cards">
        <div class="card"><div class="num">{len(defects)}</div>
          <div class="lbl">Defect types</div></div>
        <div class="card" style="border-left-color:#d9534f">
          <div class="num">{high}</div><div class="lbl">High severity</div></div>
        <div class="card" style="border-left-color:#f0ad4e">
          <div class="num">{medium}</div><div class="lbl">Medium severity</div></div>
        <div class="card" style="border-left-color:#5bc0de">
          <div class="num">{low}</div><div class="lbl">Low severity</div></div>
      </div>

      <h2>Media Information</h2>
      <table class="defect-table"><tbody>{media_html}</tbody></table>

      <h2>Detected Defects</h2>
      {_defect_table_html(defects, for_video=is_video)}

      {snap_html}
      {keyframes_html}
      {_findings_html(defects, transcript, is_video, image_crops)}
      {_unconfirmed_voice_mentions_html(unconfirmed_voice_mentions)}
      {_transcript_html(transcript)}

      <p class="footer">Building Defect Inspection Report &middot; YOLO + Whisper</p>
    </div>
    """
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_HTML_STYLE}</style></head><body>{body}</body></html>
"""


# ------------------------------------------------------------------------ PDF

def render_pdf_report(
    pdf_path: str | Path,
    title: str,
    report: ImageReport | VideoReport,
    transcript: TranscriptResult | None = None,
    annotated_image_path: str | None = None,
    image_crops: list[dict] | None = None,
) -> Path:
    """Generate a PDF report at ``pdf_path`` and return the resolved path."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak,
    )

    pdf_path = Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("h1c", parent=styles["Heading1"],
                              alignment=1, textColor=colors.HexColor("#1f2933")))
    styles.add(ParagraphStyle("meta", parent=styles["Normal"],
                              textColor=colors.HexColor("#52606d"),
                              fontSize=9, alignment=1))
    styles.add(ParagraphStyle("section", parent=styles["Heading2"],
                              textColor=colors.HexColor("#3d5afe")))
    styles.add(ParagraphStyle("body9", parent=styles["BodyText"],
                              fontSize=9, leading=12))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title=title,
    )
    story: list = []

    story.append(Paragraph(title, styles["h1c"]))
    story.append(Paragraph(
        f"Generated on {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["meta"],
    ))
    story.append(Spacer(1, 12))

    is_video = isinstance(report, VideoReport)
    unconfirmed_voice_mentions = (
        list(getattr(report, "unconfirmed_voice_mentions", []))
        if is_video else []
    )

    # Media info table
    media_rows = [
        ["File", Path(report.source_path).name],
        ["Resolution", f"{report.width}x{report.height}"],
    ]
    if is_video:
        v = report  # type: ignore[assignment]
        media_rows += [
            ["Duration", _fmt_time(v.duration_sec)],
            ["FPS", f"{v.fps:.2f}"],
            ["Frames sampled", f"{v.sampled_frames} of {v.total_frames}"],
        ]
    media_rows.append(["Analysis time", f"{report.elapsed_sec:.1f} s"])

    media_tbl = Table(media_rows, hAlign="LEFT", colWidths=[4*cm, 12*cm])
    media_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e4e7eb")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(Paragraph("Media Information", styles["section"]))
    story.append(media_tbl)
    story.append(Spacer(1, 14))

    # Summary
    defects = report.detected_defects
    high = sum(1 for d in defects if d.get("severity") == "High")
    medium = sum(1 for d in defects if d.get("severity") == "Medium")
    low = sum(1 for d in defects if d.get("severity") == "Low")
    story.append(Paragraph("Summary", styles["section"]))
    story.append(Paragraph(
        f"<b>{len(defects)}</b> defect type(s) detected — "
        f"<font color='#d9534f'><b>{high}</b> high</font>, "
        f"<font color='#f0ad4e'><b>{medium}</b> medium</font>, "
        f"<font color='#5bc0de'><b>{low}</b> low</font>.",
        styles["body9"],
    ))
    story.append(Spacer(1, 10))

    # Defect table
    story.append(Paragraph("Detected Defects", styles["section"]))
    if not defects:
        story.append(Paragraph(
            "No defects detected above the confidence threshold.",
            styles["body9"],
        ))
    else:
        head = ["Defect", "Severity", "Conf.", "Where", "Recommendation"]
        rows = [head]
        for d in defects:
            if is_video:
                place_count = int(d.get("place_count", 1) or 1)
                where = (f"{place_count} place(s), first {_fmt_time(d.get('first_time_sec',0))}, "
                         f"last {_fmt_time(d.get('last_time_sec',0))} "
                         f"({d.get('count',0)} frames)")
            else:
                where = f"{d.get('count',0)} region(s)"
            rows.append([
                Paragraph(f"<b>{d['label']}</b>", styles["body9"]),
                d.get("severity", "?"),
                _fmt_pct(d.get("max_confidence", 0.0)),
                Paragraph(where, styles["body9"]),
                Paragraph(d.get("recommendation", ""), styles["body9"]),
            ])
        tbl = Table(rows, hAlign="LEFT",
                    colWidths=[3*cm, 2*cm, 2*cm, 4*cm, 6*cm])
        ts = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f4f8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e4e7eb")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
        for i, d in enumerate(defects, start=1):
            color = SEVERITY_COLORS.get(d.get("severity"), "#6c757d")
            ts.add("TEXTCOLOR", (1, i), (1, i), colors.HexColor(color))
            ts.add("FONTNAME", (1, i), (1, i), "Helvetica-Bold")
        tbl.setStyle(ts)
        story.append(tbl)

    story.append(Spacer(1, 14))

    # Annotated image
    if annotated_image_path and Path(annotated_image_path).exists():
        story.append(Paragraph("Annotated Image", styles["section"]))
        try:
            story.append(RLImage(annotated_image_path,
                                 width=16*cm, height=12*cm,
                                 kind="proportional"))
        except Exception:  # noqa: BLE001
            pass
        story.append(Spacer(1, 14))

    # ---- Defect Findings: per-defect image + matched voice commentary ----
    if defects:
        story.append(PageBreak())
        story.append(Paragraph("Defect Findings", styles["section"]))
        story.append(Paragraph(
            "Each detected defect is shown below with the marked-up "
            "image and any voice commentary recorded for that defect.",
            styles["body9"],
        ))
        story.append(Spacer(1, 8))

        crop_lookup: dict[str, dict] = {}
        if image_crops:
            for c in image_crops:
                crop_lookup[c["label"]] = c

        for d in defects:
            label = d["label"]
            sev = d.get("severity", "Unknown")
            sev_color = SEVERITY_COLORS.get(sev, "#6c757d")
            conf = d.get("max_confidence", 0.0)

            if is_video:
                kf = d.get("keyframe") or {}
                img_path = kf.get("image_path")
                ts = kf.get("time_sec")
                place_count = int(d.get("place_count", 1) or 1)
                where = (
                    f"{place_count} place(s), first {_fmt_time(d.get('first_time_sec', 0))}, "
                    f"last {_fmt_time(d.get('last_time_sec', 0))} "
                    f"({d.get('count', 0)} frames)"
                )
            else:
                crop = crop_lookup.get(label, {})
                img_path = crop.get("image_path")
                ts = None
                where = f"{d.get('count', 0)} region(s) in image"

            # Title row with coloured severity tag
            story.append(Paragraph(
                f"<b><font size='12'>{label}</font></b> &nbsp; "
                f"<font color='{sev_color}'><b>[{sev}]</b></font> "
                f"&nbsp; conf {_fmt_pct(conf)}",
                styles["body9"],
            ))
            story.append(Paragraph(
                f"<b>Where:</b> {where}", styles["body9"],
            ))
            story.append(Paragraph(
                f"<b>Recommendation:</b> {d.get('recommendation', '')}",
                styles["body9"],
            ))
            story.append(Spacer(1, 4))
            place_crops = d.get("place_crops") or []
            if len(place_crops) > 1:
                story.append(Paragraph(
                    f"<b>Detected places in keyframe:</b> {len(place_crops)}",
                    styles["body9"],
                ))
                story.append(Spacer(1, 4))

            # Two-column row: marked-up image on the left, transcript on the right
            left_cell: list = []
            if img_path and Path(img_path).exists():
                try:
                    left_cell.append(RLImage(
                        img_path, width=8*cm, height=6*cm,
                        kind="proportional",
                    ))
                except Exception:  # noqa: BLE001
                    left_cell.append(Paragraph(
                        "(image unavailable)", styles["body9"],
                    ))
            else:
                left_cell.append(Paragraph(
                    "(no marked image for this defect)", styles["body9"],
                ))

            matched = match_transcript_to_defect(
                transcript, label, time_sec=ts,
            )
            right_cell: list = []
            if matched:
                right_cell.append(Paragraph(
                    "<b>Voice commentary near this defect</b>",
                    styles["body9"],
                ))
                for s in matched:
                    tag = (" [defects: " + ", ".join(s.detected_defects) + "]"
                           if s.detected_defects else "")
                    right_cell.append(Paragraph(
                        f"<b>[{_fmt_time(s.start)}]</b> "
                        f"{html.escape(s.text)}<i>{html.escape(tag)}</i>",
                        styles["body9"],
                    ))
                    right_cell.append(Spacer(1, 2))
            elif (transcript is not None and transcript.has_audio
                    and not transcript.error):
                right_cell.append(Paragraph(
                    "<i>Inspector did not comment on this defect.</i>",
                    styles["body9"],
                ))
            else:
                right_cell.append(Paragraph(
                    "<i>No voice commentary available for this media.</i>",
                    styles["body9"],
                ))

            row_tbl = Table(
                [[left_cell, right_cell]],
                colWidths=[8.5*cm, 8.5*cm],
                hAlign="LEFT",
            )
            row_tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e4e7eb")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4,
                 colors.HexColor("#e4e7eb")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafbfc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(row_tbl)
            story.append(Spacer(1, 14))

    if unconfirmed_voice_mentions:
        story.append(PageBreak())
        story.append(Paragraph(
            "Voice Mentions Not Visually Confirmed", styles["section"],
        ))
        story.append(Paragraph(
            "These items were mentioned in the audio transcript, but the "
            "visual detector did not confirm them. They are included for "
            "manual review only and are not counted as detected defects.",
            styles["body9"],
        ))
        story.append(Spacer(1, 8))

        for m in unconfirmed_voice_mentions:
            label = m.get("label", "unknown")
            kf = m.get("keyframe") or {}
            img_path = kf.get("image_path")
            ts = kf.get("time_sec")
            story.append(Paragraph(
                f"<b><font size='12'>{html.escape(label)}</font></b> "
                f"&nbsp; <font color='#6c757d'><b>[not visually confirmed]</b></font>",
                styles["body9"],
            ))
            story.append(Paragraph(
                f"<b>Mentioned near:</b> {_fmt_time(ts) if ts is not None else '-'}",
                styles["body9"],
            ))
            quotes = m.get("voice_quotes") or []

            left_cell: list = []
            if img_path and Path(img_path).exists():
                try:
                    left_cell.append(RLImage(
                        img_path, width=8*cm, height=6*cm,
                        kind="proportional",
                    ))
                except Exception:  # noqa: BLE001
                    left_cell.append(Paragraph(
                        "(image unavailable)", styles["body9"],
                    ))
            else:
                left_cell.append(Paragraph(
                    "(no nearby review frame)", styles["body9"],
                ))

            right_cell: list = [Paragraph(
                "<b>Transcript excerpt</b>", styles["body9"],
            )]
            if quotes:
                for q in quotes:
                    right_cell.append(Paragraph(
                        html.escape(q), styles["body9"],
                    ))
                    right_cell.append(Spacer(1, 2))
            else:
                right_cell.append(Paragraph(
                    "<i>No transcript excerpt saved.</i>", styles["body9"],
                ))

            row_tbl = Table(
                [[left_cell, right_cell]],
                colWidths=[8.5*cm, 8.5*cm],
                hAlign="LEFT",
            )
            row_tbl.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e4e7eb")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4,
                 colors.HexColor("#e4e7eb")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafbfc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(row_tbl)
            story.append(Spacer(1, 14))

    if transcript is not None:
        story.append(PageBreak())
        story.append(Paragraph("Voice Transcript", styles["section"]))
        if transcript.error:
            story.append(Paragraph(html.escape(transcript.error), styles["body9"]))
        elif transcript.has_audio and transcript.segments:
            for s in transcript.segments:
                tag = (" [mentions: " + ", ".join(s.detected_defects) + "]"
                       if s.detected_defects else "")
                story.append(Paragraph(
                    f"<b>[{_fmt_time(s.start)}]</b> "
                    f"{html.escape(s.text)}<i>{html.escape(tag)}</i>",
                    styles["body9"],
                ))
                story.append(Spacer(1, 2))
        elif transcript.has_audio:
            story.append(Paragraph(
                "Audio track found, but no speech was transcribed.",
                styles["body9"],
            ))

    doc.build(story)
    return pdf_path
