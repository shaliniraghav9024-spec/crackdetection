"""
YOLOv8-based defect detector for the BD3 building-defects dataset.

Replaces the previous ResNet-18 + CAM stack. Given an image or a video,
runs a fine-tuned YOLOv8 detector to produce real bounding boxes per
defect, aggregates per-class statistics, saves annotated keyframes and
(for videos) writes an annotated MP4 if requested.

Public surface kept stable for app.py / report_generator.py:
    * CLASS_NAMES, CLASS_ALIASES, SEVERITY, RECOMMENDATIONS
    * VoiceHint dataclass
    * get_model, analyze_image, analyze_video, save_defect_crops
    * ImageReport, VideoReport, FramePrediction, TilePrediction, Detection
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# ----------------------------------------------------- domain constants
CLASS_NAMES: list[str] = [
    "algae", "hole", "major_crack", "minor_crack",
    "normal", "peeling", "spalling", "stain",
]

# Free-text labels that voice transcription may produce, mapped to the
# canonical class name. Keep this in sync with audio_transcriber.DEFECT_KEYWORDS.
CLASS_ALIASES: dict[str, str] = {
    "biological_growth": "algae",
    "moss":              "algae",
    "fungus":            "algae",
    "crack":             "minor_crack",
    "fissure":           "minor_crack",
    "general_crack":     "minor_crack",
    "rust_stain":        "stain",
    "watermark":         "stain",
    "discoloration":     "stain",
    "flaking":           "peeling",
    "blistering":        "peeling",
    "delamination":      "peeling",
    "exposed_rebar":     "spalling",
    "concrete_falling":  "spalling",
    "water":             "stain",
}

SEVERITY: dict[str, str] = {
    "major_crack": "High",
    "spalling":    "High",
    "hole":        "High",
    "minor_crack": "Medium",
    "peeling":     "Medium",
    "algae":       "Low",
    "stain":       "Low",
    "normal":      "None",
}

RECOMMENDATIONS: dict[str, str] = {
    "major_crack": "Structural review recommended; monitor for movement and "
                   "consider epoxy injection / steel stitching.",
    "spalling":    "Remove loose concrete, treat exposed rebar with rust "
                   "inhibitor, and patch with repair mortar.",
    "hole":        "Patch with appropriate filler; check for water ingress "
                   "behind the surface before sealing.",
    "minor_crack": "Seal with flexible crack filler; re-inspect in 6 months.",
    "peeling":     "Strip flaking paint, clean and re-prime the affected "
                   "area before repainting.",
    "algae":       "Clean with fungicidal wash and improve ventilation / "
                   "drainage to prevent recurrence.",
    "stain":       "Cosmetic; clean with mild detergent. Investigate the "
                   "water source if the stain is recurring.",
    "normal":      "No action required.",
}

# Fallback weights when the user hasn't fine-tuned a defect model yet.
# Using YOLOv8m (medium) instead of nano gives noticeably stronger
# general object features which helps the distractor / artificial-thing
# filter and gives the ensemble a better generic backbone to fall back
# on.  Auto-downloads on first use via Ultralytics.
FALLBACK_WEIGHTS_PATH = "yolov8m.pt"


# ---------------------------------------------------------------------
# Pretrained "auxiliary" detection models, downloaded from Hugging Face.
#
# The locally-trained model in ``runs/detect/*/weights/best.pt`` only sees
# whatever data the user has hand-annotated (often just a handful of
# images, which is not enough to generalize).  These auxiliary models
# were trained on thousands of real-world crack images so they catch
# what the in-house detector misses.
#
# Each entry is (weights_path, class_remap, conf_offset).
#   * class_remap maps the auxiliary model's class names to our 8-class
#     schema.  ``None`` means "drop this prediction" (e.g. dry-wall
#     joints / model background classes).
#   * conf_offset is added to the prediction's confidence so we can
#     tune how strongly each model contributes when boxes overlap.
#
# Models are loaded lazily and silently skipped when their weight files
# are missing -- so the project still works without the downloads.
# ---------------------------------------------------------------------
AUX_MODELS: list[tuple[str, dict[str, str | None], float]] = [
    # Wall-specific crack/joint segmentation; high precision on building
    # interiors.  Joints are normal seams, NOT defects -> drop.
    ("models/levanell/yolov8n-seg-cracks-joints.pt",
     {"crack": "minor_crack", "joint": None},
     0.0),
    # OpenSistemas YOLOv8n crack-seg; trained on 4k road+wall images.
    ("models/opensistemas_n/yolov8n/weights/best.pt",
     {"crack": "minor_crack"},
     -0.05),  # slightly down-weight (more aggressive than levanell)
]


# ----------------------------------------------------------- dataclasses
@dataclass
class VoiceHint:
    """A time-stamped defect mention extracted from voice commentary."""
    time_sec: float
    label: str
    text: str


@dataclass
class Detection:
    """A single YOLO bounding-box detection in source-image coordinates."""
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2)


@dataclass
class TilePrediction:
    """Per-tile detections for a tiled image pass (in image coords)."""
    tile_idx: int
    bbox: tuple[int, int, int, int]
    detections: list[Detection]


@dataclass
class FramePrediction:
    """One sampled video frame with its detections."""
    frame_idx: int
    time_sec: float
    detections: list[Detection]
    voice_corroborated: bool = False

    @property
    def is_defect(self) -> bool:
        return any(d.label != "normal" for d in self.detections)


@dataclass
class ImageReport:
    source_path: str
    width: int
    height: int
    elapsed_sec: float
    detected_defects: list[dict] = field(default_factory=list)
    tiles: list[TilePrediction] = field(default_factory=list)
    annotated_path: str | None = None
    sampled_frames: int = 1


@dataclass
class VideoReport:
    source_path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    sampled_frames: int
    elapsed_sec: float
    detected_defects: list[dict] = field(default_factory=list)
    frames: list[FramePrediction] = field(default_factory=list)
    keyframe_paths: list[dict] = field(default_factory=list)
    unconfirmed_voice_mentions: list[dict] = field(default_factory=list)
    annotated_path: str | None = None


# ----------------------------------------------------------- model loading
_MODEL_CACHE: dict[str, object] = {}


def _resolve_default_weights() -> str:
    """Pick the most recent fine-tuned weights, else yolov8n.pt."""
    candidates = sorted(
        Path("runs/detect").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    return FALLBACK_WEIGHTS_PATH


@dataclass
class _Member:
    """One model in the ensemble, with its own class remap + score offset."""
    name: str
    model: object
    remap: dict[str, str | None]
    conf_offset: float


def _load_ultralytics_model(path: str):
    from ultralytics import YOLO  # local import keeps cold-start light
    return YOLO(path)


def _build_aux_members() -> list[_Member]:
    """Load any auxiliary pretrained models that exist on disk.

    Silently skipped if a weights file is missing; this lets the rest
    of the project work even when the downloads haven't been done.
    """
    members: list[_Member] = []
    for w_rel, remap, off in AUX_MODELS:
        w = Path(w_rel)
        if not w.exists():
            continue
        try:
            m = _load_ultralytics_model(str(w.resolve()))
        except Exception:  # noqa: BLE001
            continue
        members.append(_Member(name=w.name, model=m, remap=remap, conf_offset=off))
    return members


def get_model(weights_path: str | Path | None = None) -> list[_Member]:
    """Return the ensemble of detection models to run on each frame.

    The returned list always contains the primary user-trained model
    (``runs/detect/*/weights/best.pt`` or ``yolov8n.pt``) plus any
    auxiliary pretrained models defined in ``AUX_MODELS`` whose weight
    files exist locally.

    Each entry is a ``_Member(name, model, remap, conf_offset)`` so
    callers can apply the per-model class mapping uniformly.
    """
    if weights_path is None:
        weights_path = _resolve_default_weights()
    weights_path = str(Path(weights_path).resolve())

    cache_key = f"ensemble::{weights_path}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]  # type: ignore[return-value]

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_path}.\n"
            f"Train first with `python train_yolo.py` or pass an "
            f"explicit weights path."
        )

    primary = _Member(
        name=Path(weights_path).name,
        model=_load_ultralytics_model(weights_path),
        # Primary model's class names already match our 8-class schema;
        # explicitly map only "crack" (some user-trained variants emit
        # generic "crack") and drop "normal".
        remap={"normal": None, "crack": "minor_crack"},
        conf_offset=0.0,
    )
    members = [primary] + _build_aux_members()
    _MODEL_CACHE[cache_key] = members  # type: ignore[assignment]
    return members


# ----------------------------------------------------------- annotation
_CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "major_crack": (0, 0, 255),       # red
    "spalling":    (0, 80, 220),      # red-orange
    "hole":        (0, 50, 200),      # dark red
    "minor_crack": (0, 165, 255),     # orange
    "peeling":     (0, 215, 255),     # gold
    "algae":       (60, 200, 80),     # green
    "stain":       (200, 180, 60),    # teal
    "normal":      (180, 180, 180),   # grey
}


def _color_for(label: str) -> tuple[int, int, int]:
    return _CLASS_COLORS_BGR.get(label, (0, 215, 255))


def _draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    timestamp_text: str | None = None,
) -> np.ndarray:
    """Draw labelled bounding boxes for ``detections`` onto a copy of ``frame``.

    When ``detections`` is empty we still stamp ``timestamp_text`` on
    the bottom-right corner -- callers like the voice-only fallback
    use that to surface the inspector's quote even when CLIP couldn't
    verify any localized defect region.
    """
    if frame is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    min_side = max(40, int(min(h, w) * 0.05))

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < min_side or y2 - y1 < min_side:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            half = min_side // 2
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w - 1, cx + half), min(h - 1, cy + half)

        color = _color_for(det.label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label_text = f"{det.label} {det.confidence * 100:.0f}%"
        scale, thick = 0.55, 2
        (tw, th), _ = cv2.getTextSize(label_text, font, scale, thick)
        sticker_h = th + 10
        sx = x1
        sy = y1 - sticker_h - 2 if y1 - sticker_h - 2 >= 0 else y2 + 2
        sy = max(0, min(sy, h - sticker_h - 1))
        cv2.rectangle(
            out, (sx, sy),
            (min(w - 1, sx + tw + 12), sy + sticker_h),
            color, -1,
        )
        cv2.putText(
            out, label_text, (sx + 6, sy + th + 5),
            font, scale, (255, 255, 255), thick, cv2.LINE_AA,
        )

    if timestamp_text:
        (tw, th), _ = cv2.getTextSize(timestamp_text, font, 0.5, 1)
        pad_x, pad_y = 6, 4
        rx0 = w - tw - pad_x * 2 - 6
        ry0 = h - th - pad_y * 2 - 6
        cv2.rectangle(
            out, (rx0, ry0),
            (rx0 + tw + pad_x * 2, ry0 + th + pad_y * 2),
            (0, 0, 0), -1,
        )
        cv2.putText(
            out, timestamp_text,
            (rx0 + pad_x, ry0 + th + pad_y),
            font, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
        )
    return out


def _save_detection_place_crops(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    *,
    label: str,
    out_dir: Path,
    stem: str,
) -> list[dict]:
    """Save one annotated crop per detection for a single class."""
    if frame_bgr is None or frame_bgr.size == 0 or not detections:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame_bgr.shape[:2]
    pad = max(18, int(0.05 * min(h, w)))
    places: list[dict] = []
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        crop = frame_bgr[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            continue
        local = Detection(
            label=det.label,
            confidence=det.confidence,
            bbox=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
        )
        annotated = _draw_detections(crop, [local])
        out_path = out_dir / f"{stem}_place{idx}.jpg"
        cv2.imwrite(str(out_path), annotated)
        places.append({
            "index": idx,
            "label": label,
            "confidence": float(det.confidence),
            "bbox": det.bbox,
            "image_path": str(out_path),
        })
    return places


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(1, x2 - x1) * max(1, y2 - y1))


def _track_match_cost(
    prev_bbox: tuple[int, int, int, int],
    new_bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> float | None:
    """Return a matching cost for two detections or ``None`` if too far apart."""
    diag = float(max((width ** 2 + height ** 2) ** 0.5, 1.0))
    cx0, cy0 = _bbox_center(prev_bbox)
    cx1, cy1 = _bbox_center(new_bbox)
    center_dist = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5
    max_center_dist = max(80.0, 0.22 * diag)
    if center_dist > max_center_dist:
        return None
    a0 = _bbox_area(prev_bbox)
    a1 = _bbox_area(new_bbox)
    area_ratio = max(a0, a1) / max(1.0, min(a0, a1))
    if area_ratio > 5.0:
        return None
    return center_dist / max_center_dist + 0.08 * abs(np.log(area_ratio))


def _build_place_tracks(
    frames: list[FramePrediction],
    *,
    label: str,
    width: int,
    height: int,
    max_gap_sec: float = 1.2,
) -> list[dict]:
    """Cluster repeated detections of one class into distinct places."""
    tracks: list[dict] = []
    for fp in sorted(frames, key=lambda x: x.time_sec):
        dets = [d for d in fp.detections if d.label == label]
        if not dets:
            continue
        dets.sort(key=lambda d: _bbox_center(d.bbox)[0])
        assigned_tracks: set[int] = set()
        for det in dets:
            best_idx: int | None = None
            best_cost: float | None = None
            for idx, track in enumerate(tracks):
                if idx in assigned_tracks:
                    continue
                if fp.time_sec - track["last_time_sec"] > max_gap_sec:
                    continue
                cost = _track_match_cost(track["last_bbox"], det.bbox, width, height)
                if cost is None:
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            if best_idx is None:
                tracks.append({
                    "label": label,
                    "events": [{
                        "frame_idx": fp.frame_idx,
                        "time_sec": fp.time_sec,
                        "det": det,
                    }],
                    "last_time_sec": fp.time_sec,
                    "last_bbox": det.bbox,
                    "best": {
                        "frame_idx": fp.frame_idx,
                        "time_sec": fp.time_sec,
                        "det": det,
                    },
                })
                assigned_tracks.add(len(tracks) - 1)
            else:
                track = tracks[best_idx]
                track["events"].append({
                    "frame_idx": fp.frame_idx,
                    "time_sec": fp.time_sec,
                    "det": det,
                })
                track["last_time_sec"] = fp.time_sec
                track["last_bbox"] = det.bbox
                if det.confidence > track["best"]["det"].confidence:
                    track["best"] = {
                        "frame_idx": fp.frame_idx,
                        "time_sec": fp.time_sec,
                        "det": det,
                    }
                assigned_tracks.add(best_idx)

    summarized: list[dict] = []
    for idx, track in enumerate(tracks, start=1):
        events = track["events"]
        confs = [e["det"].confidence for e in events]
        best = track["best"]
        summarized.append({
            "index": idx,
            "label": label,
            "count": len(events),
            "first_time_sec": float(events[0]["time_sec"]),
            "last_time_sec": float(events[-1]["time_sec"]),
            "max_confidence": float(max(confs)),
            "avg_confidence": float(sum(confs) / len(confs)),
            "frame_idx": int(best["frame_idx"]),
            "time_sec": float(best["time_sec"]),
            "bbox": best["det"].bbox,
            "confidence": float(best["det"].confidence),
            "image_path": None,
        })
    summarized.sort(
        key=lambda t: (t["first_time_sec"], -t["max_confidence"]),
    )
    for idx, track in enumerate(summarized, start=1):
        track["index"] = idx
    return summarized


def _save_video_place_tracks(
    video_path: Path,
    *,
    label: str,
    tracks: list[dict],
    keyframes_dir: Path | None,
) -> list[dict]:
    """Save one exact-place crop per tracked video place."""
    if keyframes_dir is None or not tracks:
        return tracks
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return tracks
    try:
        for track in tracks:
            frame = _seek_and_read(cap, int(track["frame_idx"]))
            if frame is None:
                continue
            det = Detection(
                label=label,
                confidence=float(track["confidence"]),
                bbox=tuple(track["bbox"]),
            )
            saved = _save_detection_place_crops(
                frame,
                [det],
                label=label,
                out_dir=keyframes_dir,
                stem=f"{label}_place{int(track['index'])}_t{track['time_sec']:.2f}",
            )
            if saved:
                track["image_path"] = saved[0]["image_path"]
    finally:
        cap.release()
    return tracks


def _filter_place_tracks(tracks: list[dict]) -> list[dict]:
    """Drop very short, weak place tracks while preserving strong ones."""
    if not tracks:
        return []
    kept = [
        t for t in tracks
        if t.get("count", 0) >= 3 or t.get("max_confidence", 0.0) >= 0.5
    ]
    if not kept:
        kept = [max(tracks, key=lambda t: (t.get("max_confidence", 0.0),
                                           t.get("count", 0)))]
    kept.sort(key=lambda t: (t["first_time_sec"], -t["max_confidence"]))
    for idx, track in enumerate(kept, start=1):
        track["index"] = idx
    return kept


# ----------------------------------------------------------- detection helpers
def _bbox_iou(a: tuple[int, int, int, int],
              b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(a_area + b_area - inter, 1e-6)


def _per_class_nms(detections: list[Detection],
                   iou_thresh: float = 0.5) -> list[Detection]:
    """Per-class NMS, used to merge boxes that span tile borders."""
    if not detections:
        return []
    by_label: dict[str, list[Detection]] = {}
    for d in detections:
        by_label.setdefault(d.label, []).append(d)
    kept: list[Detection] = []
    for _, group in by_label.items():
        group.sort(key=lambda d: d.confidence, reverse=True)
        survivors: list[Detection] = []
        for d in group:
            if any(_bbox_iou(d.bbox, s.bbox) >= iou_thresh for s in survivors):
                continue
            survivors.append(d)
        kept.extend(survivors)
    return kept


def _predict_one(member: "_Member", frame_bgr: np.ndarray,
                 conf: float, iou: float, imgsz: int) -> list[Detection]:
    """Run a single member of the ensemble; apply its class remap."""
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    # Use a slightly lowered conf at the model itself so its NMS keeps
    # candidates we may still want to filter out by class remap; we
    # re-apply the user-facing ``conf`` after remapping.
    raw_conf = max(0.05, conf - max(0.0, -member.conf_offset))
    try:
        res = member.model.predict(
            source=frame_bgr, conf=raw_conf, iou=iou, imgsz=imgsz,
            verbose=False, save=False,
        )
    except Exception:  # noqa: BLE001
        return []
    if not res:
        return []
    r = res[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    names = r.names
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    out: list[Detection] = []
    for (x1, y1, x2, y2), cid, c in zip(xyxy, cls_ids, confs):
        if isinstance(names, dict):
            raw_label = names.get(int(cid), str(cid))
        else:
            raw_label = names[int(cid)]
        # Remap to our 8-class schema; ``None`` drops the prediction.
        if raw_label in member.remap:
            mapped = member.remap[raw_label]
        else:
            mapped = raw_label  # passthrough for unmapped classes
        if mapped is None or mapped == "normal":
            continue
        adj_conf = float(c) + member.conf_offset
        if adj_conf < conf:
            continue
        out.append(Detection(
            label=mapped,
            confidence=adj_conf,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
        ))
    return out


# --------------------------------------------------------------------- #
# Artificial-thing filter.
#
# Real-world inspection videos are full of objects that are NOT defects
# but visually look like one to a crack detector: clocks (round dark
# rim + thin black hands inside), watches, signs, name badges, photo
# frames, switch plates, vents, labels, etc.  We use TWO independent
# checks to suppress those false positives without retraining:
#
#   1. YOLOv8-COCO distractor boxes  (yolo_distractors.distractor_boxes)
#      Pretrained on 80 everyday object classes -- catches `clock`, `tv`,
#      `book`, `cell phone`, `person`, `stop sign`, etc.
#
#   2. CLIP zero-shot verification   (clip_verifier.verify_box)
#      For each surviving defect box we compute cosine similarity to
#      ``defect_prompt`` vs ``distractor_prompt`` text embeddings.
#      Boxes where the best distractor wins are dropped.
#
# Both filters are best-effort: if the optional dependencies aren't
# installed they're skipped without errors.
# --------------------------------------------------------------------- #

# Strict mode: a defect box is dropped if ``DISTRACTOR_OVERLAP_THRESH``
# fraction of it sits inside a known distractor box.  0.30 means "if
# 30%+ of the defect box overlaps a clock, treat it as a clock-edge
# false positive".
DISTRACTOR_OVERLAP_THRESH: float = 0.30
# Master toggles, flipped by Streamlit / tests at runtime.
ARTIFICIAL_FILTER_ENABLED: bool = True
CLIP_VERIFIER_ENABLED: bool = True


def set_artificial_filter_enabled(enabled: bool) -> None:
    global ARTIFICIAL_FILTER_ENABLED
    ARTIFICIAL_FILTER_ENABLED = bool(enabled)


def set_clip_verifier_enabled(enabled: bool) -> None:
    global CLIP_VERIFIER_ENABLED
    CLIP_VERIFIER_ENABLED = bool(enabled)


def _box_overlap_fraction(
    box: tuple[int, int, int, int],
    other: tuple[int, int, int, int],
) -> float:
    """Fraction of ``box``'s area that falls inside ``other``."""
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = other
    iw = max(0, min(x2, ox2) - max(x1, ox1))
    ih = max(0, min(y2, oy2) - max(y1, oy1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    area = max(1, (x2 - x1) * (y2 - y1))
    return inter / area


def _drop_distractor_overlapping(
    detections: list[Detection],
    frame_bgr: np.ndarray,
) -> tuple[list[Detection], list[tuple[int, int, int, int, str, float]]]:
    """Drop detections that overlap a known distractor object.

    Returns ``(kept_detections, distractor_boxes)`` so callers can show
    the user which artificial things were filtered out.  Falls back to
    a no-op if ``yolo_distractors`` isn't importable.
    """
    if not detections or not ARTIFICIAL_FILTER_ENABLED:
        return detections, []
    try:
        from yolo_distractors import distractor_boxes
    except Exception:  # noqa: BLE001
        return detections, []
    try:
        d_boxes = distractor_boxes(frame_bgr, conf=0.20, dilate_px=12)
    except Exception:  # noqa: BLE001
        return detections, []
    if not d_boxes:
        return detections, []

    kept: list[Detection] = []
    for det in detections:
        is_distractor = False
        for (x0, y0, x1, y1, _lbl, _c) in d_boxes:
            if _box_overlap_fraction(
                det.bbox, (x0, y0, x1, y1)
            ) >= DISTRACTOR_OVERLAP_THRESH:
                is_distractor = True
                break
        if not is_distractor:
            kept.append(det)
    return kept, d_boxes


def _clip_filter_detections(
    detections: list[Detection],
    frame_bgr: np.ndarray,
    margin: float = 0.02,
) -> list[Detection]:
    """Drop boxes where CLIP thinks the patch is a distractor not a defect.

    Skipped silently when ``open_clip`` isn't installed.
    """
    if not detections or not CLIP_VERIFIER_ENABLED:
        return detections
    try:
        from clip_verifier import is_available, verify_box
    except Exception:  # noqa: BLE001
        return detections
    if not is_available():
        return detections
    kept: list[Detection] = []
    for det in detections:
        try:
            r = verify_box(frame_bgr, det.bbox, margin=margin)
        except Exception:  # noqa: BLE001
            kept.append(det)
            continue
        if r is None or r.is_defect:
            kept.append(det)
    return kept


def _yolo_predict(members, frame_bgr: np.ndarray,
                  conf: float, iou: float,
                  imgsz: int = 640) -> list[Detection]:
    """Run every model in the ensemble and merge the resulting boxes.

    ``members`` is the list returned by ``get_model``; legacy callers
    that pass a single Ultralytics ``YOLO`` instance are also supported
    for back-compatibility (the model is wrapped in a default member).
    Detections that overlap a YOLO-COCO distractor (clock, sign, etc.)
    are dropped, and surviving boxes are zero-shot verified by CLIP.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    # Back-compat shim: wrap a bare Ultralytics model.
    if not isinstance(members, list):
        members = [_Member(name="legacy", model=members,
                           remap={"normal": None}, conf_offset=0.0)]

    all_dets: list[Detection] = []
    for m in members:
        all_dets.extend(_predict_one(m, frame_bgr, conf, iou, imgsz))

    if not all_dets:
        return []
    # Cross-model NMS so two models marking the same crack don't
    # produce two boxes on the keyframe.
    merged = _per_class_nms(all_dets, iou_thresh=max(iou, 0.45))
    # Stage 1: drop boxes overlapping known distractor objects.
    merged, _distractors = _drop_distractor_overlapping(merged, frame_bgr)
    # Stage 2: zero-shot CLIP verification on survivors.
    merged = _clip_filter_detections(merged, frame_bgr)
    return merged


def _frame_sharpness(frame_bgr: np.ndarray) -> float:
    """Laplacian-variance sharpness score (higher = sharper)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Window (in seconds) on either side of a voice mention that we sample
# frames from for the voice-only evidence fallback.  Inspectors often
# describe a defect a few seconds *after* the camera was pointing at
# it (they pan past the defect, then narrate), so the look-BEHIND
# window is intentionally wider than the look-AHEAD window.
VOICE_FALLBACK_LOOKBEFORE_SEC: float = 12.0
VOICE_FALLBACK_LOOKAFTER_SEC: float = 6.0
# Number of candidate frames to sample inside the window.  More samples
# = better chance of finding the right frame.
VOICE_FALLBACK_SAMPLES: int = 36
# When the visual model can't find the voice-named class even at the
# lowest threshold, we still report it -- but with this synthetic
# confidence so it ranks below real visual detections.
VOICE_FALLBACK_CONFIDENCE: float = 0.30


# --------------------------------------------------------------------- #
# Class-specific visual-evidence scorers.
#
# Used by the voice-only fallback to pick the BEST frame for a given
# defect class (instead of just the sharpest), and to draw a heuristic
# bounding box on whatever region of that frame actually looks like the
# named defect (instead of a centred reference box).
# --------------------------------------------------------------------- #

def _dominant_color_lab(
    frame_bgr: np.ndarray,
    sample_step: int = 8,
) -> np.ndarray:
    """Return the median LAB colour of the frame (proxy for "wall colour")."""
    small = frame_bgr[::sample_step, ::sample_step]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    return np.median(lab.reshape(-1, 3), axis=0)


def _distractor_exclude_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) covering YOLO-COCO distractors.

    Used by the heuristic evidence scorers so they don't mistake an
    INDIA card / clock / sign for a peeling patch or stain.  Returns
    an all-zeros mask if YOLO distractors aren't available -- the
    rest of the heuristics still work, just less accurately.
    """
    H, W = frame_bgr.shape[:2]
    zero = np.zeros((H, W), dtype=np.uint8)
    if not ARTIFICIAL_FILTER_ENABLED:
        return zero
    try:
        from yolo_distractors import distractor_mask
        return distractor_mask(frame_bgr, conf=0.20, dilate_px=12)
    except Exception:  # noqa: BLE001
        return zero


def _touches_border(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_px: int,
) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        x1 <= margin_px or y1 <= margin_px
        or x2 >= width - margin_px or y2 >= height - margin_px
    )


def _wall_context_fraction(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_px: int = 24,
    diff_thresh: float = 16.0,
) -> float:
    """How much of the bbox neighborhood still looks like the wall.

    Review-only heuristics should localize defects *on the wall*, not on
    adjacent door frames, paper scraps, or edge clutter. We estimate wall
    color from the frame median and measure what fraction of pixels in an
    expanded neighborhood stay close to that dominant wall color.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    px = max(8, int(pad_px))
    rx1, ry1 = max(0, x1 - px), max(0, y1 - px)
    rx2, ry2 = min(w, x2 + px), min(h, y2 + px)
    roi = frame_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return 0.0
    med = _dominant_color_lab(frame_bgr).astype(np.int16)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.int16)
    diff = np.linalg.norm(lab - med.reshape(1, 1, 3), axis=2)
    return float(np.mean(diff < diff_thresh))


def _peeling_evidence(
    frame_bgr: np.ndarray,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Score & locate paint-peeling-like regions.

    Strategy: peeling shows up as a *patch* whose colour deviates from
    the dominant wall colour (the underlying primer / different layer of
    paint exposed).  We compute LAB-distance from the median wall colour,
    threshold it, and pick the largest connected region.

    Artificial things (clocks / signs / books / phones) are masked out
    first so the scorer doesn't mistake an INDIA card or a watch face
    for a peeling patch.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0, None
    H, W = frame_bgr.shape[:2]
    border_margin = max(6, int(min(H, W) * 0.02))
    exclude = _distractor_exclude_mask(frame_bgr)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    med = _dominant_color_lab(frame_bgr).astype(np.int16)
    diff = np.linalg.norm(lab - med.reshape(1, 1, 3), axis=2)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    # Mode 1: broad paint-loss/discoloration patches.
    thr = float(diff.mean() + 1.2 * diff.std())
    broad_mask = (diff > thr).astype(np.uint8) * 255
    # Knock out distractor pixels so they don't get clustered into
    # candidate peeling patches.
    broad_mask[exclude > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    broad_mask = cv2.morphologyEx(broad_mask, cv2.MORPH_OPEN, k)
    broad_mask = cv2.morphologyEx(broad_mask, cv2.MORPH_CLOSE, k)

    frame_area = H * W
    best_score = 0.0
    best_bbox: tuple[int, int, int, int] | None = None

    n, labels, stats, _ = cv2.connectedComponentsWithStats(broad_mask, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # Discard giant regions (that's the wall itself) and tiny ones.
        if area < frame_area * 0.001 or area > frame_area * 0.40:
            continue
        bbox = (int(x), int(y), int(x + w), int(y + h))
        if _touches_border(bbox, W, H, border_margin):
            continue
        if _wall_context_fraction(frame_bgr, bbox, pad_px=max(w, h)) < 0.45:
            continue
        region_diff = diff[labels == i]
        mean_diff = float(region_diff.mean()) if region_diff.size else 0.0
        score = (area / frame_area) * (1.0 + mean_diff / 35.0)
        if score > best_score:
            best_score = score
            best_bbox = bbox

    # Mode 2: small exposed-primer/paint-chip spots. These are often
    # bright, low-texture outliers against an otherwise uniform wall.
    chip_mask = (
        (diff > max(12.0, diff.mean() + 0.4 * diff.std()))
        & (l_chan > np.median(l_chan) + 10)
        & (np.abs(a_chan - med[1]) + np.abs(b_chan - med[2]) > 18)
    ).astype(np.uint8) * 255
    chip_mask[exclude > 0] = 0
    chip_mask = cv2.morphologyEx(chip_mask, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(
                                     cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(chip_mask, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6 or area > frame_area * 0.01:
            continue
        bbox = (int(x), int(y), int(x + w), int(y + h))
        if _touches_border(bbox, W, H, border_margin):
            continue
        if _wall_context_fraction(frame_bgr, bbox, pad_px=24) < 0.60:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > 4.5:
            continue
        region_diff = diff[labels == i]
        mean_diff = float(region_diff.mean()) if region_diff.size else 0.0
        compact = area / max(1, w * h)
        score = compact * (0.08 + area / 250.0) * (mean_diff / 20.0)
        if score > best_score:
            best_score = score
            best_bbox = bbox

    return float(best_score), best_bbox


def _hole_evidence(
    frame_bgr: np.ndarray,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Score & locate small dark blob-like regions (holes / spalling)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0, None
    H, W = frame_bgr.shape[:2]
    exclude = _distractor_exclude_mask(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Dark spots = pixels well below the local mean.
    blur = cv2.GaussianBlur(gray, (31, 31), 0)
    diff = blur.astype(np.int16) - gray.astype(np.int16)
    mask = (diff > 25).astype(np.uint8) * 255
    mask[exclude > 0] = 0   # ignore dark pixels inside artificial things
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    n, _l, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    best_score = 0.0
    best_bbox: tuple[int, int, int, int] | None = None
    frame_area = H * W
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8 or area > frame_area * 0.02:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > 4.0:   # too elongated to be a hole
            continue
        # Score = compactness * smallness  (small + roundish wins)
        compact = area / max(1, w * h)
        score = compact * (1.0 - area / max(1.0, frame_area * 0.02))
        if score > best_score:
            best_score = score
            best_bbox = (int(x), int(y), int(x + w), int(y + h))
    return float(best_score), best_bbox


def _crack_evidence(
    frame_bgr: np.ndarray,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Score & locate thin meandering dark contours (cracks)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0, None
    H, W = frame_bgr.shape[:2]
    exclude = _distractor_exclude_mask(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    border_margin = max(6, int(min(H, W) * 0.02))

    # Dark-line mask: cracks are usually darker than their local
    # background, while wall/door boundaries mostly sit on the frame edge.
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = blur.astype(np.int16) - gray.astype(np.int16)
    dark_mask = (diff > 8).astype(np.uint8) * 255
    dark_mask[exclude > 0] = 0
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )

    best_score = 0.0
    best_bbox: tuple[int, int, int, int] | None = None
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6:
            continue
        bbox = (int(x), int(y), int(x + w), int(y + h))
        if _touches_border(bbox, W, H, border_margin):
            continue
        long_side = max(w, h)
        short_side = max(1, min(w, h))
        if _wall_context_fraction(frame_bgr, bbox, pad_px=max(18, long_side)) < 0.52:
            continue
        ar = long_side / short_side
        if ar < 2.2 or long_side < 10:
            continue
        if ar > 14.0 and short_side <= 6 and long_side > 0.20 * H:
            continue
        fill = area / max(1, w * h)
        if fill > 0.55:
            continue
        mean_darkness = float(diff[y:y+h, x:x+w][dark_mask[y:y+h, x:x+w] > 0].mean())
        score = (
            (long_side / max(W, H))
            * (1.0 - fill)
            * (mean_darkness / 8.0)
        )
        if score > best_score:
            best_score = score
            best_bbox = bbox
    return float(best_score), best_bbox


def _stain_evidence(
    frame_bgr: np.ndarray,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Score & locate large discoloured patches (stains, water marks)."""
    # Stains are essentially "peeling" without the patch-size limit.
    score, bbox = _peeling_evidence(frame_bgr)
    return score * 0.8, bbox  # slightly weaker score so peeling wins ties


def _algae_evidence(
    frame_bgr: np.ndarray,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Score & locate green / mossy regions (algae)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return 0.0, None
    H, W = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Green hue 35-85 in OpenCV's 0-180 range.
    mask = cv2.inRange(hsv, (30, 40, 30), (90, 255, 220))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, _l, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    best_area = 0
    best_bbox: tuple[int, int, int, int] | None = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 200:
            continue
        if area > best_area:
            best_area = area
            best_bbox = (int(x), int(y), int(x + w), int(y + h))
    return float(best_area / max(1, H * W)), best_bbox


def _evidence_for_class(
    frame_bgr: np.ndarray,
    label: str,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Dispatch to the per-class evidence scorer.

    Returns ``(score, bbox)`` -- score is unitless; higher = stronger
    visual evidence for the named class in this frame.  bbox is the
    region we'd draw a heuristic box on (or ``None`` if no candidate).
    """
    fn = {
        "peeling":     _peeling_evidence,
        "stain":       _stain_evidence,
        "hole":        _hole_evidence,
        "spalling":    _hole_evidence,
        "minor_crack": _crack_evidence,
        "major_crack": _crack_evidence,
        "algae":       _algae_evidence,
    }.get(label)
    if fn is None:
        return 0.0, None
    return fn(frame_bgr)


def _seek_and_read(cap, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def _box_passes_clip_for_class(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
) -> bool:
    """Return True iff CLIP thinks the bbox patch is a real defect of
    type ``label`` (not a clock, sign, flag, etc.).

    Falls back to ``True`` (accept the box) when CLIP isn't available --
    so the project still works without the optional CLIP weights.
    """
    if not CLIP_VERIFIER_ENABLED:
        return True
    try:
        from clip_verifier import is_available, verify_box
    except Exception:  # noqa: BLE001
        return True
    if not is_available():
        return True
    try:
        r = verify_box(frame_bgr, bbox, margin=0.02)
    except Exception:  # noqa: BLE001
        return True
    if r is None:
        return True
    # Accept when CLIP rates a defect prompt highest AND, if it has a
    # specific class opinion, agrees with the named ``label`` (or with
    # any aliased equivalent -- "hole" -> "minor_crack" etc.).
    if not r.is_defect:
        return False
    aliases = {label, CLASS_ALIASES.get(label, label)}
    return r.best_defect_class in aliases or r.best_defect_class in CLASS_NAMES


def _voice_only_evidence_pass(
    *,
    video_path: Path,
    voice_hints: list[VoiceHint] | None,
    already_found: set[str],
    keyframes_dir: Path | None,
    model,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    fps: float,
) -> list[dict]:
    """Build defect entries for voice-mentioned classes the model missed.

    For every (timestamp, label) in ``voice_hints`` whose ``label`` is
    not already in ``already_found``, we:

      1. Sample ``VOICE_FALLBACK_SAMPLES`` frames within
         ``VOICE_FALLBACK_WINDOW_SEC`` of the spoken timestamp.
      2. Pick the sharpest one (Laplacian variance).
      3. Re-run the YOLO ensemble at half the user's confidence
         threshold to give the model a second, more permissive chance.
      4. If that still finds nothing, save the sharp frame itself with
         the inspector's quote stamped on top -- so the user sees what
         the inspector was looking at when they named the defect.

    The resulting report entry is tagged ``source = "voice"`` so the
    UI can render it differently (e.g. with a "🎙️ voice-only" pill).
    """
    if not voice_hints:
        return []

    # Group voice hints by label so we don't process the same class
    # twice for two nearby mentions.  Keep the earliest timestamp
    # (typically when the inspector first calls it out) and the set of
    # quotes for the report caption.  Quotes are deduplicated so the
    # PDF doesn't show "There is a crack involved." three times when
    # Whisper merged a couple of repeated segments.
    by_label: dict[str, dict] = {}
    for h in voice_hints:
        if h.label in already_found:
            continue
        slot = by_label.setdefault(h.label, {
            "first_t": h.time_sec,
            "all_t": [],
            "quotes": [],
            "_seen_quotes": set(),
        })
        slot["first_t"] = min(slot["first_t"], h.time_sec)
        slot["all_t"].append(h.time_sec)
        if h.text:
            q = h.text.strip()
            key = q.lower().rstrip(".!? ").strip()
            if key and key not in slot["_seen_quotes"]:
                slot["_seen_quotes"].add(key)
                slot["quotes"].append(q)

    if not by_label:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Strength threshold: if the best frame in the local voice window
    # has a class-specific evidence score below this, we do a second
    # pass over the entire video for stronger evidence.  Inspectors
    # frequently describe a defect several seconds after panning past
    # it, so this rescue pass dramatically improves keyframe accuracy.
    GLOBAL_RESCUE_EVIDENCE_THRESHOLD = 0.020
    # Even when local evidence is above the threshold, we still upgrade
    # the keyframe if a globally-much-stronger frame exists.  This
    # bound says "use the global frame iff it's at least Nx stronger
    # than the best local frame".  3x is conservative; bumps the
    # keyframe to far-away frames only when they're clearly better.
    GLOBAL_RESCUE_RATIO = 3.0

    voice_only: list[dict] = []
    for label, slot in by_label.items():
        t_center = slot["first_t"]
        # 1) Sample candidate frames in a wide window around the voice
        #    timestamp.  We scan from -LOOKBEFORE to +LOOKAFTER seconds.
        f_center = int(round(t_center * fps))
        f_before = int(round(VOICE_FALLBACK_LOOKBEFORE_SEC * fps))
        f_after  = int(round(VOICE_FALLBACK_LOOKAFTER_SEC * fps))
        n_steps  = max(2, VOICE_FALLBACK_SAMPLES - 1)
        step     = max(1, (f_before + f_after) // n_steps)

        def _score_at(fi: int, off_from_center: int) -> dict | None:
            fr = _seek_and_read(cap, fi)
            if fr is None:
                return None
            sharp = _frame_sharpness(fr)
            if sharp < 4.0:
                # Skip motion-blur frames -- never make a good keyframe
                # and confuse the heuristic detectors.
                return None
            ev_score, ev_bbox = _evidence_for_class(fr, label)
            return {
                "fi": fi,
                "frame": fr,
                "sharp": sharp,
                "ev_score": ev_score,
                "ev_bbox": ev_bbox,
                # Ranking is dominated by class evidence -- a frame
                # where the class actually shows up beats a sharp blank
                # wall every time.  Sharpness only acts as a small
                # tiebreaker between frames with equal evidence.
                "rank": (
                    ev_score
                    + 0.0005 * min(sharp, 200)
                    - 0.0001 * abs(off_from_center) / max(1, f_after)
                ),
            }

        scored: list[dict] = []
        for off in range(-f_before, f_after + 1, step):
            fi = max(0, min(total_frames - 1, f_center + off))
            row = _score_at(fi, off)
            if row is not None:
                scored.append(row)
        if not scored:
            continue
        scored.sort(key=lambda r: r["rank"], reverse=True)

        # 1b) Global-rescue pass.  Run a sparse scan of the WHOLE video
        #     and, if a frame outside the local window has clearly
        #     stronger class-specific evidence, prefer it as the
        #     keyframe.  This handles the very common case where the
        #     inspector pans past a defect, then narrates a few seconds
        #     later -- the camera is no longer on the defect when they
        #     speak.  Defect classes like "peeling" / "crack" are
        #     global wall properties, not time-locked to the second
        #     they were named.
        local_best_score = scored[0]["ev_score"]
        run_global = (
            local_best_score < GLOBAL_RESCUE_EVIDENCE_THRESHOLD
        )
        # Sample one frame every ~0.3 s across the whole video.  Dense
        # enough to hit the actual peak of class evidence -- a defect
        # often shows up as a sharp peak in a single frame as the camera
        # pans across it.  Coarse step misses the peak entirely.
        global_rows: list[dict] = []
        visited = {r["fi"] for r in scored}
        global_step = max(1, int(round(0.3 * fps)))
        for fi in range(0, total_frames, global_step):
            if fi in visited:
                continue
            row = _score_at(fi, abs(fi - f_center))
            if row is not None:
                global_rows.append(row)
        global_best = max(
            (r for r in global_rows), key=lambda r: r["ev_score"],
            default=None,
        )

        # Decide whether to prefer the global frame over the local one.
        if global_best is not None:
            ratio_trigger = (
                local_best_score < 1e-6
                or global_best["ev_score"] >= GLOBAL_RESCUE_RATIO * local_best_score
            )
            if run_global or ratio_trigger:
                scored = [global_best] + scored
        # Final ranking now uses pure class evidence (sharpness already
        # filtered out blur), so we always pick the visually-richest
        # frame for the named defect.
        scored.sort(key=lambda r: r["ev_score"], reverse=True)

        # 2) On the top-ranked frame ONLY, try a permissive YOLO pass:
        #    if YOLO actually finds the spoken class with reasonable
        #    confidence on the same frame the heuristic picked, we use
        #    YOLO's box (more precise).  YOLO box must (a) overlap the
        #    heuristic bbox so it agrees with the CV evidence, and
        #    (b) pass CLIP verification so we don't paste a "peeling"
        #    label on the INDIA card / clock / sign.
        soft_conf = max(0.05, conf_threshold * 0.5)
        chosen_fi: int | None = None
        chosen_frame: np.ndarray | None = None
        chosen_dets: list[Detection] = []
        if scored:
            top = scored[0]
            dets = _yolo_predict(
                model, top["frame"],
                conf=soft_conf, iou=iou_threshold, imgsz=imgsz,
            )
            ev_bbox = top.get("ev_bbox")
            for d in dets:
                if d.label != label:
                    continue
                overlap_ok = (
                    ev_bbox is None
                    or _box_overlap_fraction(d.bbox, ev_bbox) > 0.20
                )
                clip_ok = _box_passes_clip_for_class(
                    top["frame"], d.bbox, label,
                )
                if overlap_ok and clip_ok:
                    chosen_fi = top["fi"]
                    chosen_frame = top["frame"]
                    chosen_dets = [d]
                    break

        # 3) If still nothing, walk down the ranked frames and pick the
        #    first one whose heuristic bbox passes CLIP verification
        #    ("does this patch actually look like a wall defect, or is
        #    it a clock / sign / flag?").  If NONE pass, we'd rather
        #    show a sharp wall frame with NO localized box (and just
        #    the inspector's quote) than paste a misleading "peeling
        #    34%" sticker on the INDIA card.
        synthetic_box = False
        if chosen_fi is None:
            picked = None
            for cand in scored[:10]:
                ev_bbox = cand.get("ev_bbox")
                if ev_bbox is None:
                    continue
                if _box_passes_clip_for_class(
                    cand["frame"], ev_bbox, label,
                ):
                    picked = cand
                    break

            if picked is not None:
                # CLIP-verified evidence: use the precise bbox.
                chosen_fi = picked["fi"]
                chosen_frame = picked["frame"]
                ev_bbox = picked["ev_bbox"]
                ev_score = picked["ev_score"]
                conf = float(min(0.55, max(
                    VOICE_FALLBACK_CONFIDENCE, 0.30 + ev_score * 1.0,
                )))
                chosen_dets = [Detection(
                    label=label, confidence=conf, bbox=ev_bbox,
                )]
                synthetic_box = True
            else:
                # No CLIP-verified region -> do NOT paint a wrong box.
                # Pick the sharpest frame nearest the spoken time and
                # surface it with no detection box; the inspector's
                # quote is enough to convey what they meant.
                near_voice = sorted(
                    [c for c in scored
                     if abs(c["fi"] - f_center) <= f_after],
                    key=lambda r: (
                        abs(r["fi"] - f_center),
                        0 if r["fi"] <= f_center else 1,
                        -r["sharp"],
                    ),
                )
                fallback = (near_voice[:1] or scored[:1])[0]
                chosen_fi = fallback["fi"]
                chosen_frame = fallback["frame"]
                chosen_dets = []   # no annotated box
                synthetic_box = True

        ts = chosen_fi / fps if fps > 0 else 0.0
        kf_path: str | None = None
        if keyframes_dir is not None:
            stamp = (
                f"voice: \"{slot['quotes'][0][:50]}\""
                if slot["quotes"] else f"voice @ t={t_center:.1f}s"
            )
            annotated = _draw_detections(
                chosen_frame,
                chosen_dets,    # may be empty -> just stamps the quote
                timestamp_text=stamp,
            )
            kf_path = str(keyframes_dir / f"{label}_voice_t{ts:.2f}.jpg")
            cv2.imwrite(kf_path, annotated)

        max_conf = (
            max(d.confidence for d in chosen_dets)
            if chosen_dets else VOICE_FALLBACK_CONFIDENCE
        )
        voice_only.append({
            "label": label,
            "severity": SEVERITY.get(label, "Unknown"),
            "count": 1,
            "max_confidence": float(max_conf),
            "avg_confidence": float(max_conf),
            "first_time_sec": float(min(slot["all_t"])),
            "last_time_sec": float(max(slot["all_t"])),
            "recommendation": RECOMMENDATIONS.get(label, ""),
            "keyframe": {
                "image_path": kf_path,
                "time_sec": ts,
                "confidence": float(max_conf),
                "frame_idx": chosen_fi,
            },
            "source": "voice",
            "voice_quotes": slot["quotes"],
            "voice_synthetic_box": synthetic_box,
            "review_box_drawn": bool(chosen_dets),
        })

    cap.release()
    return voice_only


def _aggregate_image_defects(detections: list[Detection]) -> list[dict]:
    """Build the per-class summary dict that report_generator.py expects."""
    by_class: dict[str, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.label, []).append(det)

    rows: list[dict] = []
    for label, dets in by_class.items():
        confs = [d.confidence for d in dets]
        rows.append({
            "label": label,
            "severity": SEVERITY.get(label, "Unknown"),
            "count": len(dets),
            "max_confidence": float(max(confs)),
            "avg_confidence": float(sum(confs) / len(confs)),
            "recommendation": RECOMMENDATIONS.get(label, ""),
            "keyframe": {},   # filled in by save_defect_crops()
        })
    rows.sort(key=lambda r: r["max_confidence"], reverse=True)
    return rows


# ---------------------------------------------------------------- IMAGE
def analyze_image(
    image_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    grid: int = 3,
    imgsz: int = 640,
    annotated_out: str | Path | None = None,
    voice_hints: list[VoiceHint] | None = None,
) -> ImageReport:
    """Run YOLO on an image with a ``grid x grid`` tile pass.

    Tiling lets the detector catch multiple co-occurring defects (crack +
    stain, etc.) inside a high-resolution photo. All per-tile detections
    are translated back to image coords and merged with class-aware NMS.
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    started = time.time()

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    H, W = img.shape[:2]

    model = get_model(weights_path)
    grid = max(1, int(grid))

    # 1) Whole-image pass.
    all_dets: list[Detection] = _yolo_predict(
        model, img, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz,
    )

    # 2) Tile pass (only if grid > 1). Tiles overlap by ~10% so a defect on
    #    a tile boundary still gets a complete bbox on at least one tile.
    tiles: list[TilePrediction] = []
    if grid > 1:
        tw, th = W // grid, H // grid
        ox = max(8, tw // 10)
        oy = max(8, th // 10)
        for r in range(grid):
            for c in range(grid):
                x0 = max(0, c * tw - ox)
                y0 = max(0, r * th - oy)
                x1 = min(W, (c + 1) * tw + ox)
                y1 = min(H, (r + 1) * th + oy)
                crop = img[y0:y1, x0:x1]
                tile_dets = _yolo_predict(
                    model, crop, conf=conf_threshold,
                    iou=iou_threshold, imgsz=imgsz,
                )
                shifted = [
                    Detection(
                        label=d.label, confidence=d.confidence,
                        bbox=(d.bbox[0] + x0, d.bbox[1] + y0,
                              d.bbox[2] + x0, d.bbox[3] + y0),
                    )
                    for d in tile_dets
                ]
                tiles.append(TilePrediction(
                    tile_idx=r * grid + c,
                    bbox=(x0, y0, x1, y1),
                    detections=shifted,
                ))
                all_dets.extend(shifted)

    merged = _per_class_nms(all_dets, iou_thresh=0.5)

    voice_labels = {h.label for h in (voice_hints or [])}
    if voice_labels:
        merged.sort(
            key=lambda d: (d.label in voice_labels, d.confidence),
            reverse=True,
        )

    detected = _aggregate_image_defects(merged)

    annotated_path: str | None = None
    if annotated_out is not None:
        out_path = Path(annotated_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        annotated = _draw_detections(img, merged, timestamp_text=None)
        cv2.imwrite(str(out_path), annotated)
        annotated_path = str(out_path)

    return ImageReport(
        source_path=str(image_path),
        width=W,
        height=H,
        elapsed_sec=time.time() - started,
        detected_defects=detected,
        tiles=tiles,
        annotated_path=annotated_path,
        sampled_frames=1,
    )


def save_defect_crops(
    image_path: str | Path,
    image_report: ImageReport,
    out_dir: str | Path,
    conf_threshold: float = 0.25,   # noqa: ARG001 (kept for back-compat)
) -> list[dict]:
    """Save a per-defect annotated crop and update each defect dict's
    ``keyframe.image_path`` so the HTML/PDF report can embed them.
    """
    src = cv2.imread(str(image_path))
    if src is None:
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Best (highest-confidence) detection per class across all tiles.
    best_per_class: dict[str, Detection] = {}
    for tile in image_report.tiles:
        for d in tile.detections:
            if (d.label not in best_per_class
                    or d.confidence > best_per_class[d.label].confidence):
                best_per_class[d.label] = d

    h, w = src.shape[:2]
    crops_meta: list[dict] = []
    for d in image_report.detected_defects:
        label = d["label"]
        det = best_per_class.get(label)
        if det is None:
            continue
        x1, y1, x2, y2 = det.bbox
        pad = max(20, int(0.06 * min(h, w)))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        crop = src[cy1:cy2, cx1:cx2].copy()
        local = Detection(
            label=det.label,
            confidence=det.confidence,
            bbox=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
        )
        annotated = _draw_detections(crop, [local])
        out_path = out_dir / f"{label}_crop.jpg"
        cv2.imwrite(str(out_path), annotated)
        d["keyframe"] = {
            "image_path": str(out_path),
            "time_sec": None,
            "confidence": det.confidence,
        }
        crops_meta.append({"label": label, "image_path": str(out_path)})
    return crops_meta


# ---------------------------------------------------------------- VIDEO
def analyze_video(
    video_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    every_n_seconds: float = 0.5,
    imgsz: int = 640,
    annotated_out: str | Path | None = None,
    keyframes_dir: str | Path | None = None,
    progress_cb: Callable[[float], None] | None = None,
    voice_hints: list[VoiceHint] | None = None,
    # Accepted but unused by the YOLO pipeline (kept so app.py callers
    # don't have to change). The classifier-era "strict mode" was about
    # CAM quality gating; YOLO has its own confidence gate.
    strict_mode: bool = True,        # noqa: ARG001
) -> VideoReport:
    """Run YOLO on every Nth frame of a video; aggregate per-class stats.

    For each defect class we keep the highest-confidence frame as a
    keyframe, save it to ``keyframes_dir`` with bbox annotations, and
    record (count, first/last timestamp, max/avg confidence). Voice
    mentions from the transcript are used only as corroborating context
    for visual detections; any spoken defect that is not visually
    confirmed is returned separately in
    ``VideoReport.unconfirmed_voice_mentions`` and is not counted as a
    detected defect. If ``annotated_out`` is given, every sampled frame
    is written to a new MP4 with all detections drawn on it.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    started = time.time()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    step = max(1, int(round(every_n_seconds * fps)))

    writer = None
    if annotated_out is not None:
        annotated_out = Path(annotated_out).resolve()
        annotated_out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(annotated_out), fourcc, fps, (width, height),
        )

    if keyframes_dir is not None:
        keyframes_dir = Path(keyframes_dir).resolve()
        keyframes_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(weights_path)
    voice_labels_by_time = [(h.time_sec, h.label) for h in (voice_hints or [])]

    frames: list[FramePrediction] = []
    best_per_class: dict[str, dict] = {}
    sampled = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ts = frame_idx / fps if fps > 0 else 0.0
        detections = _yolo_predict(
            model, frame, conf=conf_threshold,
            iou=iou_threshold, imgsz=imgsz,
        )
        corroborated = False
        if voice_labels_by_time and detections:
            det_labels = {d.label for d in detections}
            for vt, vlbl in voice_labels_by_time:
                if abs(vt - ts) <= 4.0 and vlbl in det_labels:
                    corroborated = True
                    break

        fp = FramePrediction(
            frame_idx=frame_idx,
            time_sec=ts,
            detections=detections,
            voice_corroborated=corroborated,
        )
        frames.append(fp)
        sampled += 1

        for det in detections:
            label = det.label
            entry = best_per_class.get(label)
            if entry is None or det.confidence > entry["confidence"]:
                kf_path: str | None = None
                place_crops: list[dict] = []
                label_dets = [d for d in detections if d.label == label]
                if keyframes_dir is not None:
                    annotated = _draw_detections(
                        frame,
                        label_dets,
                        timestamp_text=f"frame {frame_idx}  t={ts:.1f}s",
                    )
                    kf_path = str(keyframes_dir / f"{label}_t{ts:.2f}.jpg")
                    cv2.imwrite(kf_path, annotated)
                    place_crops = _save_detection_place_crops(
                        frame,
                        label_dets,
                        label=label,
                        out_dir=keyframes_dir,
                        stem=f"{label}_t{ts:.2f}",
                    )
                best_per_class[label] = {
                    "confidence": det.confidence,
                    "time_sec": ts,
                    "frame_idx": frame_idx,
                    "image_path": kf_path,
                    "place_crops": place_crops,
                    "place_count": len(label_dets),
                }

        if writer is not None:
            annotated = _draw_detections(
                frame, detections,
                timestamp_text=f"frame {frame_idx}  t={ts:.1f}s",
            )
            writer.write(annotated)

        if progress_cb is not None and total_frames:
            progress_cb(min(1.0, frame_idx / total_frames))

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if progress_cb is not None:
        progress_cb(1.0)

    # Aggregate per-class video stats
    by_class: dict[str, list[FramePrediction]] = {}
    for fp in frames:
        for det in fp.detections:
            by_class.setdefault(det.label, []).append(fp)

    detected_defects: list[dict] = []
    keyframe_paths: list[dict] = []
    for label, fp_list in by_class.items():
        confs = [
            max(d.confidence for d in fp.detections if d.label == label)
            for fp in fp_list
        ]
        times = sorted(fp.time_sec for fp in fp_list)
        kf = best_per_class.get(label, {})
        place_tracks = _build_place_tracks(
            fp_list,
            label=label,
            width=width,
            height=height,
        )
        place_tracks = _filter_place_tracks(place_tracks)
        place_tracks = _save_video_place_tracks(
            video_path,
            label=label,
            tracks=place_tracks,
            keyframes_dir=keyframes_dir,
        )
        detected_defects.append({
            "label": label,
            "severity": SEVERITY.get(label, "Unknown"),
            "count": len(fp_list),
            "max_confidence": float(max(confs)),
            "avg_confidence": float(sum(confs) / len(confs)),
            "first_time_sec": times[0] if times else 0.0,
            "last_time_sec": times[-1] if times else 0.0,
            "recommendation": RECOMMENDATIONS.get(label, ""),
            "keyframe": kf,
            "place_crops": list(kf.get("place_crops", [])),
            "place_count": max(int(kf.get("place_count", 1)), len(place_tracks)),
            "places": place_tracks,
            "source": "visual",
        })
        if kf.get("image_path"):
            keyframe_paths.append({
                "label": label,
                "image_path": kf["image_path"],
                "time_sec": kf["time_sec"],
                "confidence": kf["confidence"],
            })
        for place in place_tracks:
            if place.get("image_path"):
                keyframe_paths.append({
                    "label": f"{label} place {place['index']}",
                    "image_path": place["image_path"],
                    "time_sec": place["time_sec"],
                    "confidence": place["confidence"],
                })

    # Voice-only fallback: every defect class the inspector named in the
    # transcript that the visual pipeline failed to find gets a "voice
    # evidence" entry built from the frame closest to the spoken
    # timestamp.  This way the report never silently drops a defect the
    # inspector saw but the model couldn't classify (e.g. faint hairline
    # cracks, off-white peeling that blends with the wall).
    voice_only_mentions = _voice_only_evidence_pass(
        video_path=video_path,
        voice_hints=voice_hints,
        already_found={d["label"] for d in detected_defects},
        keyframes_dir=keyframes_dir,
        model=model,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        fps=fps,
    )

    # Merge voice-only mentions into the main defect list so the
    # report UI surfaces them with a "voice-only" badge.  We keep a
    # separate copy in ``unconfirmed_voice_mentions`` for callers who
    # want to render them differently.
    for vm in voice_only_mentions:
        detected_defects.append(vm)
        kf = vm.get("keyframe") or {}
        if kf.get("image_path"):
            keyframe_paths.append({
                "label": vm["label"],
                "image_path": kf["image_path"],
                "time_sec": kf.get("time_sec", 0.0),
                "confidence": kf.get("confidence", 0.0),
            })

    detected_defects.sort(
        key=lambda r: (
            # Visual detections first (higher source weight), then by
            # confidence -- so the report table reads cleanly.
            0 if r.get("source", "visual") == "visual" else 1,
            -float(r.get("max_confidence", 0.0)),
        ),
    )
    voice_only_mentions.sort(
        key=lambda r: (
            r.get("first_time_sec", float("inf")),
            -r.get("max_confidence", 0.0),
        ),
    )

    return VideoReport(
        source_path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        sampled_frames=sampled,
        elapsed_sec=time.time() - started,
        detected_defects=detected_defects,
        frames=frames,
        keyframe_paths=keyframe_paths,
        unconfirmed_voice_mentions=voice_only_mentions,
        annotated_path=str(annotated_out) if annotated_out else None,
    )
