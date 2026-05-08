"""
Semantic mention matcher: spoken-transcript segments -> defect classes.

This sits next to ``audio_transcriber`` (which produces word- and
segment-level Whisper output) and provides a sentence-embedding-based
matcher in addition to the keyword list in ``audio_transcriber.py``.

The core idea: embed each defect class as a short rich description,
embed each transcript segment, and match by cosine similarity above
``SIMILARITY_THRESHOLD``. This catches paraphrases the keyword list
misses ("the wall is crumbling" -> spalling) while remaining robust to
Indian-English accents that the keyword approach already partially
handles via homophone normalisation.

The model (``all-MiniLM-L6-v2``, ~90 MB) is loaded lazily on first use
and cached for the rest of the process. If sentence-transformers is
unavailable for any reason (no internet on first run, environment
broken, ...) the matcher degrades to an empty-mentions list rather
than raising -- callers can keep going on the keyword path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


# --------------------------------------------------------------------- knobs
# All tunable parameters live at the top so callers can override them
# from the Streamlit UI or env vars without spelunking the code.
SIMILARITY_THRESHOLD: float = 0.45    # cosine sim cut-off; lower = more recall
WINDOW_SECONDS:       float = 2.5     # ± window around a mention for evidence
SAMPLE_FPS:           int   = 4       # frames-per-second to evaluate in window
DEDUP_BUCKET_SECONDS: float = 3.0     # collapse repeated mentions in this window


# --------------------------------------------------------------------- corpus
# One-line rich descriptions per defect class. Each is a bag of
# inspection-relevant phrases that the encoder will average over.
# The "normal" class is intentionally absent so it never produces a
# mention. Keep these in lower-case, no punctuation, single line --
# the encoder works at the sentence/phrase level.
DEFECT_DESCRIPTIONS: dict[str, str] = {
    "algae":
        "algae green growth moss fungus mould mildew biofilm "
        "wet damp patch on wall biological discolouration",
    "hole":
        "hole opening puncture missing chunk gap void cavity "
        "in wall broken hollow opening damage",
    "major_crack":
        "major crack large wide deep fracture structural fissure "
        "dangerous severe crack badly cracked wall has broken",
    "minor_crack":
        "minor crack small thin hairline fine surface fissure "
        "slight cracking visible crack wall crack",
    "peeling":
        "peeling paint flaking blistering coming off chipped "
        "delaminated finish paint not proper paint removed paint missing",
    "spalling":
        "spalling concrete breaking off exposed rebar deterioration "
        "falling chunk plaster coming off chipped concrete material loss",
    "stain":
        "stain discoloration watermark efflorescence rust mark "
        "damp stain brown patch yellow patch salt deposit white deposit",
}


# --------------------------------------------------------------------- model cache
_MODEL = None
_CLASS_VECS = None  # torch.Tensor of shape (n_classes, dim), pre-normalised
_CLASS_LABELS: list[str] = []
_LOAD_ERROR: Exception | None = None


def _ensure_model():
    """Load (and cache) the sentence-transformer.

    Returns the model on success, or ``None`` if loading fails. The
    failure is captured in ``_LOAD_ERROR`` so callers can surface it
    if they want.
    """
    global _MODEL, _CLASS_VECS, _CLASS_LABELS, _LOAD_ERROR
    if _MODEL is not None:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = e
        return None
    _CLASS_LABELS = list(DEFECT_DESCRIPTIONS.keys())
    descriptions = [DEFECT_DESCRIPTIONS[c] for c in _CLASS_LABELS]
    _CLASS_VECS = _MODEL.encode(
        descriptions, convert_to_tensor=True, normalize_embeddings=True,
    )
    return _MODEL


def matcher_load_error() -> Exception | None:
    """Expose the most-recent load failure for UI error reporting."""
    return _LOAD_ERROR


# --------------------------------------------------------------------- types
@dataclass
class DefectMention:
    """A defect class semantically inferred from a transcript segment."""
    label: str            # canonical defect class
    time_sec: float       # midpoint of the matched segment (for keyframe seek)
    start_sec: float      # transcript segment start
    end_sec: float        # transcript segment end
    confidence: float     # cosine similarity in [-1, 1] (typically 0.45-0.85)
    text: str             # the transcript text that triggered the mention


# --------------------------------------------------------------------- core
def find_defect_mentions(
    segments: Sequence,                # iterable of audio_transcriber.Segment
    threshold: float = SIMILARITY_THRESHOLD,
    dedup_bucket: float = DEDUP_BUCKET_SECONDS,
) -> list[DefectMention]:
    """Map transcript segments to defect classes via cosine similarity.

    Parameters
    ----------
    segments
        Whisper segments. Each must expose ``start``, ``end``, ``text``
        attributes (``audio_transcriber.Segment`` does). Empty / blank
        segments are skipped.
    threshold
        Minimum cosine similarity for a mention to be recorded.
    dedup_bucket
        Mentions of the same class within this many seconds are
        collapsed to the highest-confidence one. Set to 0 to disable.

    Returns
    -------
    list[DefectMention]
        Sorted by ``time_sec`` ascending. Empty if the encoder cannot
        be loaded or there are no usable segments.
    """
    model = _ensure_model()
    if model is None:
        return []
    if not segments:
        return []

    from sentence_transformers import util

    # Filter out empty segments BEFORE encoding so indices stay aligned.
    usable: list = [s for s in segments if (getattr(s, "text", "") or "").strip()]
    if not usable:
        return []

    seg_vecs = model.encode(
        [s.text.strip() for s in usable],
        convert_to_tensor=True, normalize_embeddings=True,
    )
    sims = util.cos_sim(seg_vecs, _CLASS_VECS)  # (n_segments, n_classes)

    # First pass: collect every (label, segment) pair above threshold.
    candidates: list[DefectMention] = []
    for si, seg in enumerate(usable):
        row = sims[si]
        for ci, label in enumerate(_CLASS_LABELS):
            score = float(row[ci])
            if score < threshold:
                continue
            t_mid = 0.5 * (float(seg.start) + float(seg.end))
            candidates.append(DefectMention(
                label=label,
                time_sec=t_mid,
                start_sec=float(seg.start),
                end_sec=float(seg.end),
                confidence=score,
                text=seg.text.strip(),
            ))

    if not candidates:
        return []

    # Second pass: dedup by (label, time-bucket) keeping highest conf.
    if dedup_bucket > 0:
        bucket_size = max(0.1, float(dedup_bucket))
        keyed: dict[tuple[str, int], DefectMention] = {}
        for m in candidates:
            key = (m.label, int(m.time_sec // bucket_size))
            prev = keyed.get(key)
            if prev is None or m.confidence > prev.confidence:
                keyed[key] = m
        deduped = list(keyed.values())
    else:
        deduped = candidates

    deduped.sort(key=lambda m: m.time_sec)
    return deduped


# --------------------------------------------------------------------- usage
if __name__ == "__main__":
    # Minimal self-test using a fake transcript so this file is
    # runnable on its own (smoke test for the encoder + threshold).
    @dataclass
    class _S:
        start: float
        end: float
        text: str

    transcript = [
        _S(0.0,  3.0,  "Here on the north wall there is major cracking near the window"),
        _S(3.0,  6.5,  "And the paint is peeling off above the door"),
        _S(6.5,  9.0,  "I can also see some green algae growing in the corner"),
        _S(9.0, 12.0,  "Nothing unusual in this section, looks clean"),
        _S(12.0, 16.0, "There is a small hole below the window frame too"),
    ]
    for m in find_defect_mentions(transcript):
        print(f"  {m.time_sec:6.2f}s  {m.label:<12}  conf={m.confidence:.3f}  "
              f"\"{m.text[:60]}\"")
