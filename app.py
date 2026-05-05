"""
Streamlit web app for building-defect inspection (image + video).

Workflow
--------
* **Image mode** — upload a single photo. The detector runs on the full
  frame plus a configurable NxN tile grid so multiple co-occurring
  defects (crack + stain, etc.) inside one photo are surfaced.
* **Video mode** — upload an inspection video. Frames are sampled every
  ``N`` seconds and run through the YOLOv8 detector, with per-class
  keyframes and an optional annotated MP4. ``ffmpeg`` extracts the
  audio and Whisper transcribes any voice commentary into time-stamped
  segments; each segment is scanned for defect-related keywords and
  highlighted in the report.

Run with:

    source venv/bin/activate
    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from defect_analyzer import (
    CLASS_NAMES,
    CLASS_ALIASES,
    SEVERITY,
    VoiceHint,
    analyze_image,
    analyze_video,
    save_defect_crops,
)
from audio_transcriber import transcribe_video
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
IMAGE_EXTS = ["bmp", "jpg", "jpeg", "png", "webp"]
VIDEO_EXTS = ["avi", "m4v", "mkv", "mov", "mp4", "webm"]


@st.cache_resource(show_spinner=False)
def _warm_model():
    """Load the YOLO model once and keep it cached for the session.
    Returns the resolved weights path so the UI can show what's running."""
    from defect_analyzer import _resolve_default_weights, get_model
    weights = _resolve_default_weights()
    get_model(weights)
    return weights


# ----------------------------------------------------------------- sidebar
st.sidebar.title("⚙️ Settings")

mode = st.sidebar.radio(
    "Input type", ["Video", "Image"], horizontal=True,
    help="Video mode samples frames over time and adds Whisper voice "
         "transcription; Image mode tiles a single photo to surface "
         "multiple co-occurring defects.",
)

conf_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
    help="YOLO box must score at least this to be kept. Lower = more "
         "recall, more false positives.",
)
iou_threshold = st.sidebar.slider(
    "NMS IoU threshold",
    min_value=0.30, max_value=0.80, value=0.45, step=0.05,
    help="Non-max suppression IoU cut-off. Higher = more overlapping "
         "boxes survive.",
)

grid_size = st.sidebar.select_slider(
    "Image tile grid (NxN)",
    options=[1, 2, 3, 4],
    value=3,
    help="Image mode only. 3x3 is a good default for full-wall photos.",
    disabled=(mode != "Image"),
)

every_n_seconds = st.sidebar.slider(
    "Sample one frame every (seconds)",
    min_value=0.1, max_value=3.0, value=0.5, step=0.1,
    disabled=(mode != "Video"),
)

transcribe_audio = st.sidebar.checkbox(
    "Transcribe voice commentary (Whisper)",
    value=True,
    disabled=(mode != "Video"),
    help="Extracts the audio with ffmpeg and runs OpenAI Whisper.",
)

whisper_size = st.sidebar.selectbox(
    "Whisper model size",
    ["tiny", "base", "small", "medium", "large"],
    index=3,   # default to "medium" -- much fewer mishearings than tiny
    disabled=(mode != "Video"),
    help="Larger = more accurate but slower and uses more RAM. "
         "'medium' (~1.5 GB) drastically reduces homophone errors "
         "(\"tent\" -> \"paint\", \"whole\" -> \"hole\") on inspection "
         "audio.  'tiny' / 'base' are CPU-friendlier fallbacks.",
)

whisper_language = st.sidebar.selectbox(
    "Voice commentary language",
    ["English (forced)", "Auto-detect"],
    index=0,
    disabled=(mode != "Video"),
    help="Whisper-medium / large sometimes misclassifies Indian-English "
         "inspection audio as Hindi / Marathi.  Forcing English fixes "
         "that.  Switch to auto-detect only if your commentary is in a "
         "non-English language.",
)
_whisper_lang_arg = "en" if whisper_language.startswith("English") else None

save_annotated_video = st.sidebar.checkbox(
    "Save annotated video (slower)",
    value=False,
    disabled=(mode != "Video"),
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Artificial-thing filter**")

filter_distractors = st.sidebar.checkbox(
    "Suppress clocks / signs / watches / phones (YOLO-COCO mask)",
    value=True,
    help="Runs a stock COCO-trained YOLO to find common wall objects "
         "(clock, tv, phone, person, book, stop sign…). Defect boxes "
         "that overlap any of those are treated as false positives "
         "and dropped. Disable only if your video contains no such "
         "objects -- the detector will then assume every dark rim is "
         "a defect.",
)

filter_clip = st.sidebar.checkbox(
    "Zero-shot CLIP verifier",
    value=True,
    help="For each surviving defect box, OpenAI's CLIP compares it "
         "against ‘a wall with a hole / crack / peeling paint’ vs "
         "‘a clock / sign / watch / poster’. If a distractor prompt "
         "wins, the box is dropped. Loads ~350 MB once on first use.",
)

# Apply toggles to the analyzer module-level flags.
from defect_analyzer import (
    set_artificial_filter_enabled, set_clip_verifier_enabled,
)
set_artificial_filter_enabled(filter_distractors)
set_clip_verifier_enabled(filter_clip)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Detected defect classes**\n\n"
    + "\n".join(f"- `{c}` ({SEVERITY.get(c, '?')})"
                for c in CLASS_NAMES if c != "normal"),
)


# ----------------------------------------------------------------- header
st.title("🏗️ Building Defect Inspector")
st.caption(
    "YOLOv8 multi-defect detector with bounding boxes, voice-commentary "
    "transcription (videos), and downloadable PDF / HTML / JSON reports."
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
st.caption(f"Loaded weights: `{Path(weights_path).name}`")


# ----------------------------------------------------------------- upload
upload = st.file_uploader(
    f"Upload an inspection {mode.lower()}",
    type=(IMAGE_EXTS if mode == "Image" else VIDEO_EXTS),
    accept_multiple_files=False,
)
if upload is None:
    st.info(f"👆 Upload an inspection {mode.lower()} to start the analysis.")
    st.stop()


input_path = _save_upload(upload, WORK_DIR)
run_dir = input_path.parent
st.caption(
    f"Loaded **{Path(upload.name).name}**  ·  working dir `{run_dir.name}`"
)


# ----------------------------------------------------------------- analyze
report_obj = None
transcript = None
defects_summary: list[dict] = []
image_crops: list[dict] | None = None
voice_unconfirmed: list[dict] = []
report_title = "Building Defect Inspection Report"

if mode == "Video":
    annotated_video_path = (run_dir / "annotated.mp4") if save_annotated_video else None
    keyframes_dir = run_dir / "keyframes"

    # Voice runs first so its mentions can guide the detector via ranking.
    if transcribe_audio:
        with st.spinner(
            f"Extracting audio and transcribing with Whisper '{whisper_size}'..."
        ):
            transcript = transcribe_video(
                input_path,
                model_size=whisper_size,
                language=_whisper_lang_arg,
            )

    voice_hints = _build_voice_hints(transcript)
    if voice_hints:
        st.caption(
            f"🎙️ Voice mentions found: **{len(voice_hints)}** defect cue(s)."
        )

    progress = st.progress(0.0, text="Sampling frames...")

    def _on_progress(p: float) -> None:
        progress.progress(min(max(p, 0.0), 1.0),
                          text=f"Detecting... {p * 100:.0f}%")

    with st.spinner("Running YOLOv8 detection on video frames..."):
        report_obj = analyze_video(
            input_path,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            every_n_seconds=every_n_seconds,
            annotated_out=annotated_video_path,
            keyframes_dir=keyframes_dir,
            progress_cb=_on_progress,
            voice_hints=voice_hints,
        )
    progress.progress(1.0, text="Detection complete.")
    defects_summary = report_obj.detected_defects
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

else:  # Image mode
    st.subheader("Uploaded image")
    st.image(str(input_path), use_container_width=True)

    annotated_image_path = run_dir / "annotated.jpg"

    with st.spinner("Running YOLOv8 detection (whole image + tile grid)..."):
        report_obj = analyze_image(
            input_path,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            grid=grid_size,
            annotated_out=annotated_image_path,
        )
    defects_summary = report_obj.detected_defects

    image_crops = save_defect_crops(
        input_path, report_obj,
        out_dir=run_dir / "crops",
        conf_threshold=conf_threshold,
    )
    st.caption(
        f"Whole image + **{grid_size * grid_size}** tile pass(es) · "
        f"**{len(defects_summary)}** defect class(es) found."
    )
    if annotated_image_path.exists():
        st.markdown("### Annotated image")
        st.image(str(annotated_image_path), use_container_width=True)


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
        "No defects detected above the confidence threshold. "
        "Try lowering it in the sidebar if you expect defects."
    )
else:
    rows = []
    for d in defects_summary:
        if mode == "Video":
            where = (f"{d.get('count', 0)} frames · "
                     f"{_fmt_time(d.get('first_time_sec', 0))}–"
                     f"{_fmt_time(d.get('last_time_sec', 0))}")
        else:
            where = f"{d.get('count', 0)} region(s) in image"
        rows.append({
            "Defect": d["label"],
            "Severity": d.get("severity", "?"),
            "Count": d.get("count", 0),
            "Places": d.get("place_count", 1) if mode == "Video" else 1,
            "Max conf.": f"{d.get('max_confidence', 0) * 100:.1f}%",
            "Avg conf.": f"{d.get('avg_confidence', 0) * 100:.1f}%",
            "Where": where,
            "Recommendation": d.get("recommendation", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


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
            if mode == "Video":
                where = (
                    f"first @ {_fmt_time(d.get('first_time_sec', 0))}, "
                    f"last @ {_fmt_time(d.get('last_time_sec', 0))} "
                    f"({d.get('count', 0)} frames)"
                )
            else:
                where = f"{d.get('count', 0)} region(s) in image"

            with cols[0]:
                if img_path and Path(img_path).exists():
                    cap = (
                        f"{label} — {d.get('max_confidence', 0) * 100:.1f}%"
                        + (f" @ {_fmt_time(ts)}" if ts is not None else "")
                    )
                    st.image(img_path, use_container_width=True, caption=cap)
                else:
                    st.info("No marked image available for this defect.")

            with cols[1]:
                place_count = int(d.get("place_count", 1) or 1)
                st.markdown(
                    f"**{label}** &nbsp; "
                    f"{_severity_pill(d.get('severity', '?'))}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Confidence:** {d.get('max_confidence', 0) * 100:.1f}%  \n"
                    f"**Places shown:** {place_count}  \n"
                    f"**Where:** {where}  \n"
                    f"**Recommendation:** {d.get('recommendation', '')}"
                )
                places = d.get("places") or []
                place_cards = places if places else (d.get("place_crops") or [])
                if len(place_cards) > 1:
                    st.markdown("**Detected places**")
                    gcols = st.columns(min(3, len(place_cards)))
                    for idx, place in enumerate(place_cards):
                        with gcols[idx % len(gcols)]:
                            p_img = place.get("image_path")
                            if p_img and Path(p_img).exists():
                                label_text = f"Place {place.get('index', idx + 1)}"
                                if place.get("time_sec") is not None:
                                    label_text += f" @ {_fmt_time(place.get('time_sec'))}"
                                st.image(
                                    p_img,
                                    use_container_width=True,
                                    caption=(
                                        f"{label_text}"
                                        f" — {place.get('confidence', 0.0) * 100:.1f}%"
                                    ),
                                )
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

if mode == "Video" and voice_unconfirmed:
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
                    st.image(img_path, use_container_width=True, caption=caption)
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

if mode == "Video" and transcript is not None:
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

annotated_image_for_report = None
if mode == "Image":
    candidate = run_dir / "annotated.jpg"
    if candidate.exists():
        annotated_image_for_report = str(candidate)

pdf_path = run_dir / "report.pdf"
pdf_ok = True
try:
    render_pdf_report(
        pdf_path=pdf_path,
        title=report_title,
        report=report_obj,
        transcript=transcript,
        annotated_image_path=annotated_image_for_report,
        image_crops=image_crops,
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
        use_container_width=True,
    )
else:
    st.button("PDF unavailable", disabled=True, use_container_width=True)
