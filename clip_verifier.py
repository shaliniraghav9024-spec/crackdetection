"""
CLIP-based zero-shot defect verifier.

A small ResNet-18 classifier trained on a focused dataset is great at
saying "this looks like a crack" but easily fooled by visually similar
distractors (watches, clocks, signs, vehicles, framed pictures).
OpenAI's CLIP, on the other hand, was trained on hundreds of millions of
captioned images and "knows" what a clock vs a wall vs a hole actually
looks like.

This module wraps an OpenCLIP model (ViT-B/32 by default) and exposes a
single function ``verify_box`` that, given a frame and a candidate box,
returns:

* ``defect_score`` -- max similarity to a defect prompt.
* ``distractor_score`` -- max similarity to a "not a defect" prompt
  (clock, sign, watch, etc.).
* ``best_defect`` / ``best_distractor`` -- the prompt strings that
  matched best (useful for debugging / display).
* ``is_defect`` -- True iff a defect prompt beat the best distractor by
  a configurable margin.

The model + prompt embeddings are loaded lazily on first use and cached
for the rest of the process so the cost is paid once.

CLIP weights download from the open_clip cache (`pretrained="laion2b_s34b_b79k"`
for ViT-B-32) on first run; ~350 MB.  The cache is under
``~/.cache/clip/`` after that.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import torch

# Lazy imports so the rest of the project still works when open_clip is
# missing (the verifier just becomes a no-op).
_OPEN_CLIP = None
_LOAD_ERROR: Exception | None = None


# Prompt sets ---------------------------------------------------------------
#
# The defect prompts intentionally use plain "wall" language so CLIP
# matches close-up wall textures rather than building-specific jargon.
# The distractor prompts cover the most common real-world false
# positives observed in inspection videos.
DEFECT_PROMPTS: dict[str, list[str]] = {
    "hole": [
        "a photo of a hole in a painted wall",
        "a round puncture or void in a concrete wall",
        "a dark circular hole in a building wall",
        "a photo of a pinhole or small opening in a wall surface",
        "a hollow opening in a brick or plastered wall",
    ],
    "minor_crack": [
        "a photo of a hairline crack on a painted wall",
        "a close-up of a thin crack on a wall surface",
        "a small dark line crack on a painted concrete wall",
        "a fine fissure running across a wall surface",
        "a narrow crack in plaster or paint",
    ],
    "major_crack": [
        "a photo of a large crack on a wall",
        "a photo of a wide structural crack in a building",
        "a deep fissure on a painted wall",
        "a photo of a big crack running through concrete",
    ],
    "spalling": [
        "a photo of concrete spalling with exposed rebar or material",
        "a chipped corner of a painted wall exposing the underlying material",
        "a photo of broken concrete with a chunk missing",
        "a photo of concrete falling off a wall surface",
    ],
    "peeling": [
        "a photo of a wall with paint peeling off",
        "a photo of flaking paint on a wall",
        "a wall where paint has chipped off revealing a different colour underneath",
        "paint blistering and lifting off a building wall",
    ],
    "stain": [
        "a photo of a discoloured patch on a wall",
        "a water stain on a painted wall",
        "rust marks on a wall surface",
        "a brownish damp stain on a building wall",
    ],
    "algae": [
        "a photo of green algae growing on a wall",
        "moss or biological growth on an exterior wall",
        "green patches of biological growth on a building facade",
    ],
}

DISTRACTOR_PROMPTS: list[str] = [
    "a photo of a round wall clock",
    "a photo of a wristwatch",
    "a photo of a small printed sign or sticker on a wall",
    "a photo of a plastic name badge or label",
    "a photo of a national flag",
    "a photo of printed text on a card",
    "a photo of a poster on a wall",
    "a photo of a framed picture on a wall",
    "a photo of a door frame and door",
    "a photo of a wooden panel on a wall",
    "a photo of a switch plate or electrical outlet",
    "a photo of a piece of furniture",
    "a photo of a uniform painted wall with no defects",
    "a photo of a clean smooth wall surface",
]


@dataclass
class VerifyResult:
    is_defect: bool
    defect_score: float
    distractor_score: float
    best_defect: str
    best_defect_class: str
    best_distractor: str
    margin: float


_lock = threading.Lock()
_state: dict = {
    "model": None,
    "preprocess": None,
    "tokenizer": None,
    "device": None,
    "defect_emb": None,         # (P_defect, D)
    "defect_prompt_strs": [],   # length P_defect
    "defect_prompt_classes": [], # length P_defect
    "distractor_emb": None,     # (P_distract, D)
    "distractor_prompt_strs": [],
}


def _try_import() -> bool:
    """Lazy-import open_clip.  Returns True iff available."""
    global _OPEN_CLIP, _LOAD_ERROR
    if _OPEN_CLIP is not None:
        return True
    if _LOAD_ERROR is not None:
        return False
    try:
        import open_clip  # type: ignore
        _OPEN_CLIP = open_clip
        return True
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = e
        return False


def is_available() -> bool:
    """Return True if CLIP can be loaded on this machine."""
    return _try_import()


def _ensure_loaded(
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
) -> bool:
    """Load CLIP weights and pre-compute prompt embeddings.

    Returns False if open_clip isn't installed (the rest of the project
    will then skip CLIP verification gracefully).
    """
    if not _try_import():
        return False
    if _state["model"] is not None:
        return True

    with _lock:
        if _state["model"] is not None:
            return True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        oc = _OPEN_CLIP
        try:
            model, _, preprocess = oc.create_model_and_transforms(
                model_name, pretrained=pretrained,
            )
            tokenizer = oc.get_tokenizer(model_name)
        except Exception as e:  # noqa: BLE001
            global _LOAD_ERROR
            _LOAD_ERROR = e
            return False
        model.eval().to(device)

        # Build defect prompt embeddings (one row per prompt).
        defect_strs: list[str] = []
        defect_classes: list[str] = []
        for cls, prompts in DEFECT_PROMPTS.items():
            for p in prompts:
                defect_strs.append(p)
                defect_classes.append(cls)
        with torch.no_grad():
            tok = tokenizer(defect_strs).to(device)
            d_emb = model.encode_text(tok).float()
            d_emb = d_emb / d_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            tok2 = tokenizer(DISTRACTOR_PROMPTS).to(device)
            x_emb = model.encode_text(tok2).float()
            x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        _state.update({
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "device": device,
            "defect_emb": d_emb,
            "defect_prompt_strs": defect_strs,
            "defect_prompt_classes": defect_classes,
            "distractor_emb": x_emb,
            "distractor_prompt_strs": DISTRACTOR_PROMPTS,
        })
        return True


def _encode_image(bgr: np.ndarray) -> torch.Tensor:
    """Encode one BGR crop into a unit-norm CLIP image embedding."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil = Image.fromarray(rgb)
    tensor = _state["preprocess"](pil).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        emb = _state["model"].encode_image(tensor).float()
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return emb


def _crop_with_padding(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_ratio: float = 0.6,
) -> np.ndarray:
    """Crop ``bbox`` with proportional padding so CLIP sees enough context."""
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    H, W = frame_bgr.shape[:2]
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    X0 = max(0, x0 - pad_x)
    Y0 = max(0, y0 - pad_y)
    X1 = min(W, x1 + pad_x)
    Y1 = min(H, y1 + pad_y)
    if X1 <= X0 or Y1 <= Y0:
        return frame_bgr.copy()
    return frame_bgr[Y0:Y1, X0:X1].copy()


def verify_box(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin: float = 0.02,
) -> VerifyResult | None:
    """Decide whether the patch ``bbox`` is a real defect.

    Returns ``None`` when CLIP isn't available (caller should ignore).
    Otherwise returns a ``VerifyResult`` with ``is_defect`` reflecting
    whether the best defect prompt beat the best distractor by at
    least ``margin`` in cosine similarity space.
    """
    if not _ensure_loaded():
        return None
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    crop = _crop_with_padding(frame_bgr, bbox)
    if crop.size == 0:
        return None
    img_emb = _encode_image(crop)               # (1, D)

    d_sims = (img_emb @ _state["defect_emb"].T).squeeze(0).cpu().numpy()
    x_sims = (img_emb @ _state["distractor_emb"].T).squeeze(0).cpu().numpy()

    d_idx = int(d_sims.argmax())
    x_idx = int(x_sims.argmax())
    d_score = float(d_sims[d_idx])
    x_score = float(x_sims[x_idx])

    is_defect = (d_score - x_score) >= margin
    return VerifyResult(
        is_defect=is_defect,
        defect_score=d_score,
        distractor_score=x_score,
        best_defect=_state["defect_prompt_strs"][d_idx],
        best_defect_class=_state["defect_prompt_classes"][d_idx],
        best_distractor=_state["distractor_prompt_strs"][x_idx],
        margin=d_score - x_score,
    )


def verify_frame(
    frame_bgr: np.ndarray,
    margin: float = 0.02,
) -> VerifyResult | None:
    """Whole-frame variant of ``verify_box`` (no crop)."""
    return verify_box(
        frame_bgr,
        bbox=(0, 0, frame_bgr.shape[1], frame_bgr.shape[0]),
        margin=margin,
    )
