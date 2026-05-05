"""
YOLOv8 distractor mask.

The hand-tuned ``_build_artificial_mask`` in ``defect_analyzer.py`` is
heuristic (brightness, saturation, text clustering) and therefore
misses irregular objects -- a black-rimmed clock with a white face is
a textbook example.

This module wraps Ultralytics' stock COCO-trained YOLOv8n and exposes a
single function ``distractor_mask`` that returns a binary mask covering
every detected COCO object that's commonly mistaken for a wall defect:

  * ``clock``       (round wall clocks)
  * ``tv``          (mounted screens / monitors)
  * ``laptop``      (placed on furniture in office shots)
  * ``cell phone``  (held by inspector)
  * ``remote``      (visually similar to a small dark rectangle on a wall)
  * ``book``        (on a shelf in the background)
  * ``stop sign``   (orange/red sign content)
  * ``keyboard`` / ``mouse``  (desk clutter)
  * ``person``      (the inspector themselves)
  * ``frame`` / ``picture`` (when the model has them) -- COCO doesn't,
    but our hand-tuned mask already handles flat rectangles.

Bounding boxes are dilated slightly so the object's *edge* (which is
the high-contrast feature that fools the crack detector) is fully
covered.

The model + weights are loaded lazily, cached, and the predict() call
is silent (no console spam from ultralytics).  If ``ultralytics`` is
missing or the weights download fails, ``distractor_mask`` returns an
all-zeros mask and the rest of the pipeline keeps working.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np


_DISTRACTOR_NAMES: set[str] = {
    "clock", "tv", "laptop", "cell phone", "remote", "book",
    "keyboard", "mouse", "person", "stop sign",
    # Round things that often appear on inspection-room walls:
    "frisbee", "sports ball",
}

_lock = threading.Lock()
_state: dict = {
    "model": None,
    "load_error": None,
    "name_to_id": {},
    "distractor_ids": set(),
}

# YOLOv8m (medium) gives stronger detection of clocks / TVs / signs /
# books than yolov8n -- which directly improves the distractor mask
# and prevents the "watch detected as defect" / "INDIA card detected
# as peeling" false positives.  Falls back to yolov8n.pt if the
# medium weights aren't on disk.
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "yolov8m.pt"
_FALLBACK_WEIGHTS = Path(__file__).resolve().parent / "yolov8n.pt"


def _try_load(weights: str | Path = _DEFAULT_WEIGHTS) -> bool:
    """Lazily load the YOLOv8m COCO model.  Returns True on success.

    Tries the requested weights first, then falls back to the smaller
    yolov8n.pt if the medium file isn't available yet.  Both auto-
    download via Ultralytics on first use.
    """
    if _state["model"] is not None:
        return True
    if _state["load_error"] is not None:
        return False
    with _lock:
        if _state["model"] is not None:
            return True
        candidates = [str(weights)]
        if Path(weights).name != _FALLBACK_WEIGHTS.name:
            candidates.append(str(_FALLBACK_WEIGHTS))
        from ultralytics import YOLO  # type: ignore
        os.environ.setdefault("YOLO_VERBOSE", "False")
        last_err: Exception | None = None
        for w in candidates:
            try:
                model = YOLO(w)
                name_to_id = {v: k for k, v in model.names.items()}
                distractor_ids = {
                    name_to_id[n] for n in _DISTRACTOR_NAMES
                    if n in name_to_id
                }
                _state.update({
                    "model": model,
                    "name_to_id": name_to_id,
                    "distractor_ids": distractor_ids,
                    "weights_path": w,
                })
                return True
            except Exception as e:  # noqa: BLE001
                last_err = e
        _state["load_error"] = last_err
        return False


def is_available() -> bool:
    """True if the COCO YOLOv8n model can be used."""
    return _try_load()


def distractor_mask(
    frame_bgr: np.ndarray,
    conf: float = 0.20,
    dilate_px: int = 12,
) -> np.ndarray:
    """Return a binary mask (uint8, 0/255) covering YOLO-detected distractors.

    Returns an all-zeros mask when the model can't load or no
    distractor object is detected.  The mask is dilated so the
    high-contrast border of the object is also covered.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    H, W = frame_bgr.shape[:2]
    out = np.zeros((H, W), dtype=np.uint8)

    if not _try_load():
        return out

    model = _state["model"]
    distractor_ids = _state["distractor_ids"]
    if not distractor_ids:
        return out

    try:
        # `verbose=False` suppresses per-call logging.
        results = model.predict(
            frame_bgr, conf=conf, verbose=False, device="cpu",
        )
    except Exception:  # noqa: BLE001
        return out

    if not results:
        return out
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return out

    classes = res.boxes.cls.cpu().numpy().astype(int)
    xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
    for cls_id, (x0, y0, x1, y1) in zip(classes, xyxy):
        if cls_id not in distractor_ids:
            continue
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        out[y0:y1, x0:x1] = 255

    if dilate_px > 0 and out.any():
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1),
        )
        out = cv2.dilate(out, k, iterations=1)
    return out


def distractor_boxes(
    frame_bgr: np.ndarray,
    conf: float = 0.20,
    dilate_px: int = 12,
) -> list[tuple[int, int, int, int, str, float]]:
    """Return distractor objects as boxes ``(x0, y0, x1, y1, label, conf)``.

    Each box is dilated by ``dilate_px`` so the object's high-contrast
    *edge* (the feature that usually fools the crack detector) is fully
    covered.  Returns an empty list if the model can't load.
    """
    if frame_bgr is None or frame_bgr.size == 0 or not _try_load():
        return []
    H, W = frame_bgr.shape[:2]
    model = _state["model"]
    distractor_ids = _state["distractor_ids"]
    try:
        results = model.predict(
            frame_bgr, conf=conf, verbose=False, device="cpu",
        )
    except Exception:  # noqa: BLE001
        return []
    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    classes = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
    out: list[tuple[int, int, int, int, str, float]] = []
    for cls_id, c, (x0, y0, x1, y1) in zip(classes, confs, xyxy):
        if cls_id not in distractor_ids:
            continue
        # Dilate by dilate_px on each side, clamp to frame bounds.
        x0 = max(0, x0 - dilate_px); y0 = max(0, y0 - dilate_px)
        x1 = min(W, x1 + dilate_px); y1 = min(H, y1 + dilate_px)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append((int(x0), int(y0), int(x1), int(y1),
                    model.names[int(cls_id)], float(c)))
    return out


def detected_distractor_labels(
    frame_bgr: np.ndarray,
    conf: float = 0.20,
) -> list[tuple[str, float]]:
    """Return (label, conf) for every distractor object in the frame.

    Useful for debugging / surfacing in the UI ("clock hidden by mask").
    Empty list if the model can't load.
    """
    if frame_bgr is None or frame_bgr.size == 0 or not _try_load():
        return []
    model = _state["model"]
    distractor_ids = _state["distractor_ids"]
    try:
        results = model.predict(frame_bgr, conf=conf, verbose=False, device="cpu")
    except Exception:  # noqa: BLE001
        return []
    if not results:
        return []
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    classes = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    out: list[tuple[str, float]] = []
    for cls_id, c in zip(classes, confs):
        if cls_id in distractor_ids:
            out.append((model.names[int(cls_id)], float(c)))
    return out
