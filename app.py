"""
Streamlit web app for building-defect inspection (video only).

Upload an inspection video. Frames are sampled every ``N`` seconds and
run through the YOLO detector (architecture chosen by whichever
``best.pt`` is resolved on load), with per-class keyframes and an
optional annotated MP4. ``ffmpeg`` extracts the audio and Whisper
transcribes any voice commentary into time-stamped segments; each
segment is scanned for defect-related keywords and highlighted in the
report.

Run with:

    source venv/bin/activate
    streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from defect_analyzer import (
    CLASS_NAMES,
    CLASS_ALIASES,
    SEVERITY,
    VoiceHint,
    analyze_video,
    find_evidence_frame_around_time,
    render_any_defect_frame_instances,
)
from audio_transcriber import transcribe_video
from defect_matcher import (
    DefectMention,
    SIMILARITY_THRESHOLD,
    WINDOW_SECONDS,
    SAMPLE_FPS,
    DEDUP_BUCKET_SECONDS,
    find_defect_mentions,
)
from report_generator import (
    render_pdf_report,
    match_transcript_to_defect,
)


# -------------------------------------------------------------- helpers
def _build_voice_hints(transcript) -> list[VoiceHint]:
    """Convert a Whisper transcript into a flat list of timestamped
    defect mentions for the analyzer to use as detection hints."""
    if transcript is None or not transcript.segments:
        return []
    hints: list[VoiceHint] = []
    for seg in transcript.segments:
        if not seg.detected_defects:
            continue
        t_mid = 0.5 * (seg.start + seg.end)
        for raw_label in seg.detected_defects:
            label = CLASS_ALIASES.get(raw_label, raw_label)
            if label in CLASS_NAMES:
                hints.append(VoiceHint(
                    time_sec=t_mid, label=label, text=seg.text,
                ))
    return hints


def _save_upload(uploaded_file, work_dir: Path) -> Path:
    """Persist a Streamlit uploaded file into a per-run working dir."""
    suffix = Path(uploaded_file.name).suffix
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=work_dir))
    target = run_dir / f"input{suffix}"
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _severity_pill(sev: str) -> str:
    cls_map = {"High": "pill-high", "Medium": "pill-medium",
               "Low": "pill-low"}
    cls = cls_map.get(sev, "pill-none")
    return f"<span class='result-pill {cls}'>{sev}</span>"


def _fmt_time(sec: float | None) -> str:
    if sec is None:
        return "-"
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


# -------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Building Defect Inspector",
    page_icon="🏗️",
    layout="wide",
)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
      .stApp h1 { color: #1f2933; }
      .small-caption { color: #52606d; font-size: 13px; }
      .result-pill {
        display:inline-block; padding:4px 12px; border-radius:999px;
        font-size:12px; font-weight:600; color:#fff; margin-right:6px;
      }
      .pill-high { background:#d9534f; }
      .pill-medium { background:#f0ad4e; }
      .pill-low { background:#5bc0de; }
      .pill-none { background:#6c757d; }
    </style>
    """,
    unsafe_allow_html=True,
)


WORK_DIR = Path("output/app_runs").resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EXTS = ["avi", "m4v", "mkv", "mov", "mp4", "webm"]


@st.cache_resource(show_spinner=False)
def _warm_model():
    """Load the YOLO model once and keep it cached for the session.
    Returns the resolved weights path so the UI can show what's running."""
    from defect_analyzer import _resolve_default_weights, get_model
    weights = _resolve_default_weights()
    get_model(weights)
    return weights


# --------------------------------------------------- auto-tuned defaults
# All knobs are tuned automatically — no manual settings exposed in the UI.
# The detector runs at low confidence to maximise recall, then a per-class
# adaptive threshold (see _auto_threshold_filter below) suppresses noisy
# detections based on the per-class confidence distribution observed in
# this video.
_AUTO_CONF_FLOOR    = 0.10   # initial recall pass (low = catch more)
_AUTO_IOU           = 0.45   # standard NMS IoU
_AUTO_FRAME_STRIDE  = 0.5    # sample one frame every 0.5s
# faster-whisper + int8 makes "medium" cost about the same on CPU as
# the old openai-whisper "small" did, with noticeably fewer mishearings
# of Indian-English inspection vocabulary.  Set the WHISPER_SIZE env
# var (e.g. ``WHISPER_SIZE=large-v3``) to override -- only worth it if
# you have a GPU; large-v3 on CPU is ~10x slower than medium.
_AUTO_WHISPER_SIZE  = os.environ.get("WHISPER_SIZE", "medium").strip() or "medium"
_AUTO_WHISPER_LANG  = "en"   # forced English (most common for these inspections)
_AUTO_SAVE_VIDEO    = False  # keyframes always saved; full annotated mp4 is slow

# False-positive suppression filters.
#
# 1. Artificial distractor mask: stock COCO YOLO finds clocks, books,
#    cell phones, ties, handbags, etc., and any defect box that overlaps
#    one >= 50% is dropped. Catches "wall clock detected as crack",
#    "phone in inspector's hand detected as hole".
# 2. CLIP zero-shot verifier: each surviving defect crop is scored
#    against text prompts. The distractor prompts include "a photo of
#    a wristwatch" and "printed text on a card", which is exactly what
#    fixes the watch / ID-card false positives that were leaking
#    through into the report.
#
# Both are enabled by default. They can hurt recall on a model that's
# still being trained, so set DISABLE_FP_FILTERS=1 in the environment to
# turn them off temporarily during a re-training cycle.
from defect_analyzer import (
    set_artificial_filter_enabled, set_clip_verifier_enabled,
)
_FP_FILTERS_ON = os.environ.get("DISABLE_FP_FILTERS", "").strip() not in ("1", "true", "yes")
set_artificial_filter_enabled(_FP_FILTERS_ON)
set_clip_verifier_enabled(_FP_FILTERS_ON)


# ----------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Inspector")
st.sidebar.caption(
    "Fully automatic — just upload a video. Confidence, NMS, frame "
    "sampling, and audio settings are all tuned per-clip from the "
    "detections themselves."
)
if _FP_FILTERS_ON:
    st.sidebar.caption(
        "🛡️ False-positive filters **ON** — clocks, watches, ID cards, "
        "phones, books, ties, signs are suppressed."
    )
else:
    st.sidebar.caption(
        "⚠️ FP filters disabled (DISABLE_FP_FILTERS=1)."
    )
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Defect classes**\n\n"
    + "\n".join(f"- `{c}` ({SEVERITY.get(c, '?')})"
                for c in CLASS_NAMES if c != "normal"),
)


# ----------------------------------------------------------------- header
st.title("🏗️ Building Defect Inspector")
st.caption(
    "YOLO multi-defect video detector with bounding boxes, "
    "voice-commentary transcription, and downloadable PDF / HTML / JSON reports."
)

with st.spinner("Loading YOLO model..."):
    try:
        weights_path = _warm_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load model: {e}")
        st.stop()


def _detect_arch_from_weights(p: Path) -> str:
    """Best-effort architecture label from the weights file path or args.yaml.
    Falls back to the file name if nothing else is known."""
    name = p.name
    args_yaml = p.parent.parent / "args.yaml"
    base_model = ""
    if args_yaml.exists():
        try:
            import yaml as _y
            base_model = (_y.safe_load(args_yaml.read_text()) or {}).get("model", "") or ""
        except Exception:
            pass
    probe = (base_model + " " + name).lower()
    if "yolo11" in probe or "yolov11" in probe:
        return "YOLO11"
    if "yolov10" in probe or "yolo10" in probe:
        return "YOLOv10"
    if "yolov9" in probe or "yolo9" in probe:
        return "YOLOv9"
    if "yolov8" in probe or "yolo8" in probe:
        return "YOLOv8"
    if "yolov5" in probe or "yolo5" in probe:
        return "YOLOv5"
    return "YOLO"


_ARCH_LABEL = _detect_arch_from_weights(Path(weights_path))
st.caption(f"Loaded weights: `{Path(weights_path).name}` · architecture: **{_ARCH_LABEL}**")



# ----------------------------------------------------------------- upload
upload = st.file_uploader(
    "Upload an inspection video",
    type=VIDEO_EXTS,
    accept_multiple_files=False,
)
if upload is None:
    st.info("👆 Upload an inspection video to start the analysis.")
    st.stop()


input_path = _save_upload(upload, WORK_DIR)
run_dir = input_path.parent
st.caption(
    f"Loaded **{Path(upload.name).name}**  ·  working dir `{run_dir.name}`"
)


# ----------------------------------------------------------------- analyze
transcript = None
voice_unconfirmed: list[dict] = []
report_title = "Building Defect Inspection Report"

annotated_video_path = (run_dir / "annotated.mp4") if _AUTO_SAVE_VIDEO else None
keyframes_dir = run_dir / "keyframes"

# Voice runs first so its mentions can guide the detector via ranking.
with st.spinner(
    f"Extracting audio and transcribing with Whisper '{_AUTO_WHISPER_SIZE}'..."
):
    transcript = transcribe_video(
        input_path,
        model_size=_AUTO_WHISPER_SIZE,
        language=_AUTO_WHISPER_LANG,
    )

voice_hints = _build_voice_hints(transcript)

# ------------------------------------------------------------------
# Semantic mention pass + real-time mention-driven evidence panel.
#
# This is the "spec" pipeline: embed every transcript segment with
# all-MiniLM-L6-v2, find segments whose cosine similarity to a defect
# class crosses SIMILARITY_THRESHOLD, then for each mention scrub the
# video in a +- WINDOW_SECONDS / SAMPLE_FPS window and grab the best
# detection of the mentioned class. Streamed live via st.status so the
# user sees evidence appear as it is found.
# ------------------------------------------------------------------
semantic_mentions: list[DefectMention] = []
semantic_evidence: list[dict] = []   # one entry per mention -> {mention, evidence}

if (transcript is not None
        and transcript.has_speech
        and transcript.segments):
    with st.spinner(
        "Matching transcript to defect classes (sentence-transformers)..."
    ):
        semantic_mentions = find_defect_mentions(
            transcript.segments,
            threshold=SIMILARITY_THRESHOLD,
            dedup_bucket=DEDUP_BUCKET_SECONDS,
        )

if voice_hints or semantic_mentions:
    st.caption(
        f"🎙️ Voice mentions: **{len(voice_hints)}** keyword cue(s) · "
        f"**{len(semantic_mentions)}** semantic mention(s) "
        f"(sim ≥ {SIMILARITY_THRESHOLD})."
    )

# Fold the semantic mentions into voice_hints so analyze_video's voice-
# corroboration logic sees them too. We keep both lists separately for
# the per-mention evidence panel below.
for m in semantic_mentions:
    label = CLASS_ALIASES.get(m.label, m.label)
    if label in CLASS_NAMES:
        voice_hints.append(VoiceHint(
            time_sec=m.time_sec, label=label, text=m.text,
        ))

# Stream the per-mention evidence pass while the user watches.
if semantic_mentions:
    st.markdown("## 🎙️ Live mention scan")
    with st.status(
        f"Scanning {len(semantic_mentions)} mention(s) for visual evidence...",
        expanded=True,
    ) as status:
        for i, m in enumerate(semantic_mentions, 1):
            status.write(
                f"**{i}/{len(semantic_mentions)}**  "
                f"`{_fmt_time(m.time_sec)}`  **{m.label}**  "
                f"(sim {m.confidence:.2f}) — _{m.text[:120]}_"
            )
            try:
                ev = find_evidence_frame_around_time(
                    input_path,
                    target_class=m.label,
                    time_sec=m.time_sec,
                    weights_path=weights_path,
                    window_seconds=WINDOW_SECONDS,
                    sample_fps=SAMPLE_FPS,
                    conf_threshold=_AUTO_CONF_FLOOR,
                    iou_threshold=_AUTO_IOU,
                )
            except Exception as e:  # noqa: BLE001
                status.write(f"  · evidence search failed: {e}")
                ev = None

            # Render the evidence frame (or nearest fallback) inline so
            # the user sees defects appear in real time.
            if ev is not None:
                if ev.found and ev.annotated_image_bgr is not None:
                    img_path = run_dir / f"mention_{i:02d}_{m.label}.jpg"
                    cv2.imwrite(str(img_path), ev.annotated_image_bgr)
                    cap_text = (
                        f"{m.label} @ {_fmt_time(ev.matched_time_sec or m.time_sec)}"
                        f" — YOLO {ev.confidence * 100:.1f}%"
                        f" · audio match {m.confidence:.2f}"
                    )
                    status.image(str(img_path), caption=cap_text,
                                 width="stretch")
                    semantic_evidence.append({
                        "mention": m,
                        "evidence": ev,
                        "image_path": str(img_path),
                    })
                elif ev.nearest_frame_bgr is not None:
                    img_path = run_dir / f"mention_{i:02d}_{m.label}_nearest.jpg"
                    cv2.imwrite(str(img_path), ev.nearest_frame_bgr)
                    status.image(
                        str(img_path),
                        caption=(
                            f"{m.label} @ {_fmt_time(m.time_sec)}"
                            " — no visual confirmation; nearest frame shown"
                        ),
                        width="stretch",
                    )
                    semantic_evidence.append({
                        "mention": m,
                        "evidence": ev,
                        "image_path": str(img_path),
                    })
        status.update(
            label=f"Mention scan complete — "
                  f"{sum(1 for e in semantic_evidence if e['evidence'].found)}/"
                  f"{len(semantic_mentions)} visually confirmed.",
            state="complete",
        )

progress = st.progress(0.0, text="Sampling frames...")

def _on_progress(p: float) -> None:
    progress.progress(min(max(p, 0.0), 1.0),
                      text=f"Detecting... {p * 100:.0f}%")

with st.spinner(f"Running {_ARCH_LABEL} detection on video frames..."):
    report_obj = analyze_video(
        input_path,
        weights_path=weights_path,
        conf_threshold=_AUTO_CONF_FLOOR,
        iou_threshold=_AUTO_IOU,
        every_n_seconds=_AUTO_FRAME_STRIDE,
        annotated_out=annotated_video_path,
        keyframes_dir=keyframes_dir,
        progress_cb=_on_progress,
        voice_hints=voice_hints,
    )
progress.progress(1.0, text="Detection complete.")

# Auto per-class threshold: classes whose detections are mostly low-conf
# (median below a global cut-off and detection density high) get their
# threshold raised to that median to suppress noise. Classes with strong
# signal keep the recall floor. Voice-mentioned classes get the floor
# even if vision is weak — we trust corroboration.
def _auto_threshold_filter(report, base_floor: float, voice_classes: set[str]) -> dict[str, float]:
    from collections import defaultdict
    confs: dict[str, list[float]] = defaultdict(list)
    for fp in report.frames:
        for det in (fp.detections or []):
            confs[det.label].append(det.confidence)
    thresholds: dict[str, float] = {}
    for label, vals in confs.items():
        if not vals:
            thresholds[label] = base_floor
            continue
        vals_sorted = sorted(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        density = len(vals) / max(1, report.sampled_frames or 1)
        if label in voice_classes:
            thresholds[label] = base_floor
        elif density > 0.5 and median < 0.30:
            thresholds[label] = max(base_floor + 0.05, median)
        else:
            thresholds[label] = max(base_floor, 0.15)
    return thresholds

_voice_classes = {h.label for h in voice_hints}
_per_class_thr = _auto_threshold_filter(report_obj, _AUTO_CONF_FLOOR, _voice_classes)

# Re-filter detected_defects + per-frame is_defect using adaptive thresholds.
_filtered_summary = []
for d in report_obj.detected_defects:
    thr = _per_class_thr.get(d["label"], _AUTO_CONF_FLOOR)
    if d.get("max_confidence", 0.0) >= thr:
        _filtered_summary.append(d)
report_obj.detected_defects = _filtered_summary
defects_summary = _filtered_summary
voice_unconfirmed = report_obj.unconfirmed_voice_mentions

sampled = report_obj.sampled_frames
passed = sum(1 for fp in report_obj.frames if fp.is_defect)
rejected = sampled - passed
voice_passed = sum(1 for fp in report_obj.frames
                   if fp.is_defect and fp.voice_corroborated)
voice_msg = (f" · **{voice_passed}** voice-corroborated"
             if voice_hints else "")
voice_only_msg = (
    f" · **{len(voice_unconfirmed)}** voice mention(s) not visually confirmed"
    if voice_unconfirmed else ""
)
st.caption(
    f"Sampled **{sampled}** frame(s) · **{passed}** with defect(s) · "
    f"**{rejected}** clean{voice_msg}{voice_only_msg}."
)

if annotated_video_path and annotated_video_path.exists():
    st.markdown("### Annotated video")
    st.video(str(annotated_video_path))


# ----------------------------------------------------------------- summary
st.markdown("## 📊 Defect Summary")

c1, c2, c3, c4 = st.columns(4)
high = sum(1 for d in defects_summary if d.get("severity") == "High")
medium = sum(1 for d in defects_summary if d.get("severity") == "Medium")
low = sum(1 for d in defects_summary if d.get("severity") == "Low")
c1.metric("Defect types", len(defects_summary))
c2.metric("High severity", high)
c3.metric("Medium severity", medium)
c4.metric("Low severity", low)

if not defects_summary:
    st.success(
        "No defects detected in this video. The auto-tuned threshold "
        "kept everything below the noise floor for this clip."
    )
else:
    # ------------------------------------------------------------------
    # Flat per-frame summary: one card per unique frame that contains
    # ANY defect (across all classes). All boxes for all classes in
    # that frame are drawn together. Adjacent samples and same-scene
    # duplicates collapse via time-bucket + spatial-IoU dedup so the
    # camera lingering on one wall doesn't fill the page.
    # ------------------------------------------------------------------
    st.caption("🎞️ Each defect-containing frame is shown below.")
    instances_dir = run_dir / "summary_instances"
    flat_instances = render_any_defect_frame_instances(
        input_path,
        report_obj.frames,
        out_dir=instances_dir,
        dedup_seconds=1.5,
        max_instances=3,
    )

    if not flat_instances:
        st.info(
            "No clean defect frames survived dedup. "
            "Defect Findings below still lists the per-class details."
        )
    else:
        for i, inst in enumerate(flat_instances, 1):
            img = inst.get("image_path")
            t = inst.get("time_sec")
            nb = inst.get("num_boxes", 1)
            labels = inst.get("labels") or []
            label_str = ", ".join(f"`{l}`" for l in labels)

            st.markdown(
                f"##### Frame {i} @ `{_fmt_time(t)}` — "
                f"{nb} defect{'s' if nb != 1 else ''} "
                f"<span class='small-caption'>({label_str})</span>",
                unsafe_allow_html=True,
            )
            if img and Path(img).exists():
                st.image(img, width="stretch")
            else:
                st.info("No frame image available.")

            # Pull the closest transcript line for any of the labels in
            # this frame so the inspector's words align with the visuals.
            best_seg = None
            best_dt = float("inf")
            for lab in labels:
                segs = match_transcript_to_defect(
                    transcript, lab, time_sec=t,
                )
                for s in segs:
                    dt = abs(0.5 * (s.start + s.end) - (t or 0.0))
                    if dt < best_dt:
                        best_dt = dt
                        best_seg = s
            if best_seg and (best_seg.text or "").strip():
                st.markdown(
                    f"<div class='small-caption'>"
                    f"🎙️ <code>[{_fmt_time(best_seg.start)}]</code> "
                    f"<i>“{best_seg.text.strip()}”</i></div>",
                    unsafe_allow_html=True,
                )


# ------------------------------------------------- per-defect findings UI
if defects_summary:
    st.markdown("## 🔍 Defect Findings (per-defect detail)")
    for d in defects_summary:
        label = d["label"]

        with st.container(border=True):
            cols = st.columns([1, 1.2])

            kf = d.get("keyframe") or {}
            img_path = kf.get("image_path")
            ts = kf.get("time_sec")
            where = (
                f"first @ {_fmt_time(d.get('first_time_sec', 0))}, "
                f"last @ {_fmt_time(d.get('last_time_sec', 0))} "
                f"({d.get('count', 0)} frames)"
            )

            with cols[0]:
                if img_path and Path(img_path).exists():
                    cap = (
                        f"{label} — {d.get('max_confidence', 0) * 100:.1f}%"
                        + (f" @ {_fmt_time(ts)}" if ts is not None else "")
                    )
                    st.image(img_path, width="stretch", caption=cap)
                else:
                    st.info("No marked image available for this defect.")

            with cols[1]:
                st.markdown(
                    f"**{label}** &nbsp; "
                    f"{_severity_pill(d.get('severity', '?'))}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Confidence:** {d.get('max_confidence', 0) * 100:.1f}%  \n"
                    f"**Where:** {where}  \n"
                    f"**Recommendation:** {d.get('recommendation', '')}"
                )
                # Note: the per-frame card grid now lives in the Defect
                # Summary section above (deduplicated by both time and
                # bounding-box overlap). The old "Detected places"
                # gallery was duplicating those rows, often showing the
                # same physical defect three or four times because
                # consecutive sample frames re-detected it -- removed.
                matched = match_transcript_to_defect(
                    transcript, label, time_sec=ts,
                )
                if matched:
                    # Deduplicate by lowercased text so the same
                    # repeated sentence doesn't appear twice.
                    seen: set[str] = set()
                    unique_segs = []
                    for s in matched:
                        key = (s.text or "").strip().lower().rstrip(".!? ")
                        if key and key not in seen:
                            seen.add(key)
                            unique_segs.append(s)
                    st.markdown(
                        "**🎙️ Voice commentary near this defect**"
                    )
                    for s in unique_segs:
                        tag = (" — _defects: "
                               + ", ".join(s.detected_defects)
                               + "_") if s.detected_defects else ""
                        st.markdown(
                            f"`[{_fmt_time(s.start)}]` {s.text}{tag}"
                        )
                elif (transcript is not None
                      and transcript.has_audio
                      and not transcript.error):
                    st.caption("Inspector did not comment on this defect.")

if voice_unconfirmed:
    st.markdown("## 🎙️ Voice Mentions Not Visually Confirmed")
    st.warning(
        "These items were mentioned in the audio transcript, but the "
        "visual detector did not confirm them. They are shown for review "
        "only and are not counted as detected defects."
    )
    for mention in voice_unconfirmed:
        label = mention["label"]
        kf = mention.get("keyframe") or {}
        img_path = kf.get("image_path")
        ts = kf.get("time_sec")
        quotes = mention.get("voice_quotes") or []
        review_box_drawn = bool(mention.get("review_box_drawn"))

        with st.container(border=True):
            cols = st.columns([1, 1.2])
            with cols[0]:
                if img_path and Path(img_path).exists():
                    caption = (
                        f"{label} review frame"
                        + (f" @ {_fmt_time(ts)}" if ts is not None else "")
                    )
                    st.image(img_path, width="stretch", caption=caption)
                else:
                    st.info("No nearby review frame available.")
            with cols[1]:
                st.markdown(
                    f"**{label}** &nbsp; "
                    "<span class='result-pill pill-none'>not visually confirmed</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Mentioned in audio:** "
                    f"{_fmt_time(mention.get('first_time_sec'))}"
                )
                if not review_box_drawn:
                    st.caption(
                        "Nearest relevant frame shown. No reliable exact "
                        "visual box was found for this spoken defect."
                    )
                if quotes:
                    st.markdown("**🎙️ Transcript**")
                    for q in quotes:
                        st.markdown(f"> _{q}_")

if transcript is not None:
    st.markdown("## 📝 Voice Transcript")
    if transcript.error:
        st.warning(transcript.error)
    elif transcript.has_speech and transcript.segments:
        st.caption(
            f"Language: `{transcript.language or 'unknown'}` · "
            f"{len(transcript.segments)} segment(s)"
        )
        for seg in transcript.segments:
            tag = ""
            if seg.detected_defects:
                tag = " — _mentions: " + ", ".join(seg.detected_defects) + "_"
            st.markdown(f"`[{_fmt_time(seg.start)}]` {seg.text}{tag}")
    elif transcript.has_audio:
        st.caption("Audio track found, but no speech was transcribed.")


# ----------------------------------------------------------------- exports
st.markdown("## 📄 Inspection Report")

pdf_path = run_dir / "report.pdf"
pdf_ok = True
try:
    render_pdf_report(
        pdf_path=pdf_path,
        title=report_title,
        report=report_obj,
        transcript=transcript,
    )
except Exception as e:  # noqa: BLE001
    pdf_ok = False
    st.error(f"PDF generation failed: {e}")

if pdf_ok and pdf_path.exists():
    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_path.read_bytes(),
        file_name="defect_report.pdf",
        mime="application/pdf",
        width="stretch",
    )
else:
    st.button("PDF unavailable", disabled=True, width="stretch")
