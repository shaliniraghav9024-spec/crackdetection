"""
Voice-commentary transcription for inspection videos.

Pipeline
--------
1. ``ffmpeg`` extracts a 16-kHz mono WAV track from the video.
2. ``openai-whisper`` (loaded lazily, cached) transcribes the audio
   into time-stamped segments.
3. Segments are tagged with whichever defect keywords they mention,
   so the final report can correlate "what the inspector said" with
   "what the model saw" at the same timestamp.

Whisper is optional.  If the model fails to load (no internet, missing
package, etc.) the app falls back gracefully and produces a stub
transcript explaining that the audio could not be processed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


# Keyword -> defect class.  Loose pattern matching keeps it forgiving
# of mispronunciations / accents.  When the inspector says generic words
# ("hole", "paint not proper", "wall is broken") we still want the
# right defect class to fire so the report includes a frame.
DEFECT_KEYWORDS: dict[str, list[str]] = {
    "major_crack": [
        "major crack", "big crack", "large crack", "wide crack",
        "structural crack", "deep crack",
    ],
    "minor_crack": [
        "minor crack", "small crack", "hairline", "fine crack",
        "thin crack",
    ],
    "hole": [
        "hole", "holes", "puncture", "perforation", "hollow",
        "small hole", "tiny hole", "pin hole", "pinhole",
    ],
    "spalling": [
        "spalling", "spalled", "spall", "concrete falling",
        "concrete coming off", "rebar exposed", "exposed rebar",
        "broken concrete", "chipped",
    ],
    "peeling": [
        "peeling", "peel", "flaking", "flaked", "blistering",
        "delamination", "paint coming off",
        # Common phrasings used by inspectors:
        "paint is not proper", "paint not proper", "paint is removed",
        "paint removed", "paint is gone", "paint chipped",
        "paint is peeling", "no paint", "paint missing",
        "paint is bad", "paint is broken",
    ],
    "algae": ["algae", "moss", "fungus", "biological growth",
              "green growth", "green patch"],
    "stain": ["stain", "stained", "discolour", "discolor", "discoloration",
              "discolouration", "watermark", "rust mark", "efflorescence"],
    "water": ["water", "moisture", "damp", "leak", "leakage", "seepage",
              "wet patch"],
    "general_crack": ["crack", "cracks", "cracking", "cracked", "fissure"],
}


_WHISPER_MODEL = None
_WHISPER_LOAD_ERROR: Exception | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    detected_defects: list[str] = field(default_factory=list)


@dataclass
class TranscriptResult:
    has_audio: bool
    has_speech: bool
    language: str | None
    text: str
    segments: list[Segment]
    audio_path: str | None = None
    error: str | None = None
    elapsed_sec: float = 0.0
    defect_mentions: dict[str, list[float]] = field(default_factory=dict)


# --------------------------------------------------------------------- audio

def _has_audio_stream(video_path: Path) -> bool:
    """Return True if ffprobe reports at least one audio stream."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, check=False, timeout=20,
        )
        return "audio" in out.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_audio(video_path: str | Path,
                  out_path: str | Path | None = None) -> Path | None:
    """Extract mono 16-kHz WAV from video. Returns ``None`` if no audio track."""
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with `sudo apt install ffmpeg`."
        )
    if not _has_audio_stream(video_path):
        return None

    out_path = Path(out_path) if out_path else \
        Path(tempfile.mkstemp(suffix=".wav")[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    if out_path.stat().st_size < 4096:
        # Likely a silent/empty track
        out_path.unlink(missing_ok=True)
        return None
    return out_path


# ------------------------------------------------------------------- whisper

_WHISPER_LOADED_SIZE: str | None = None


def _load_whisper(model_size: str = "medium"):
    """Load (and cache) the whisper model. Returns ``None`` on failure.

    The cache is keyed on ``model_size`` so callers can switch between
    "tiny" and "medium" at runtime (e.g. via the Streamlit sidebar) and
    each size loads at most once per process.
    """
    global _WHISPER_MODEL, _WHISPER_LOAD_ERROR, _WHISPER_LOADED_SIZE
    if _WHISPER_MODEL is not None and _WHISPER_LOADED_SIZE == model_size:
        return _WHISPER_MODEL
    # Different size requested -> drop the previous one and reload.
    _WHISPER_MODEL = None
    _WHISPER_LOAD_ERROR = None
    try:
        import whisper  # openai-whisper
        _WHISPER_MODEL = whisper.load_model(model_size)
        _WHISPER_LOADED_SIZE = model_size
        return _WHISPER_MODEL
    except Exception as e:  # noqa: BLE001
        _WHISPER_LOAD_ERROR = e
        return None


def _dedup_repeated_phrases(text: str) -> str:
    """Collapse Whisper's stuttered repetitions inside a single segment.

    Whisper's smaller models often emit "X, X on Y" or "X. X on Y" where
    the same opening clause is repeated. This walks comma/semicolon-
    separated clauses and drops any clause whose words are a strict
    prefix of the next clause, so

        "There is some hole, there is some hole on wall."

    becomes

        "There is some hole on wall."
    """
    if not text:
        return text
    parts = re.split(r'(\s*[,;]\s*)', text)
    if len(parts) < 3:
        return text

    def _norm_words(s: str) -> list[str]:
        return re.sub(r'[^\w\s]', '', s).lower().split()

    keep = [True] * len(parts)
    for i in range(0, len(parts) - 2, 2):
        cur_words = _norm_words(parts[i])
        nxt_words = _norm_words(parts[i + 2])
        if (len(cur_words) >= 2
                and len(nxt_words) >= len(cur_words)
                and nxt_words[:len(cur_words)] == cur_words):
            keep[i] = False
            keep[i + 1] = False  # drop the trailing separator too

    rebuilt = "".join(p for p, k in zip(parts, keep) if k).lstrip()
    if rebuilt:
        rebuilt = rebuilt[0].upper() + rebuilt[1:]
    return rebuilt


# Whisper-tiny / -base often hear common inspection words as homophones
# (especially with non-native accents).  We normalise the most frequent
# substitutions BEFORE keyword matching so the report still picks the
# right defect class even when the raw transcript says "tent" or "whole".
# The substitutions are intentionally conservative -- they only fire on
# whole words inside a building-inspection context (followed by "wall",
# "is not proper", "removed", etc.) so they don't corrupt regular text.
_HOMOPHONE_RULES: list[tuple[str, str]] = [
    # "tent is not proper" / "tent is removed" -> "paint ..."
    (r"\btent\b(?=\s+(is\s+not\s+proper|is\s+removed|is\s+gone|"
     r"not\s+proper|is\s+bad|is\s+chipped|is\s+peeling|missing))", "paint"),
    # "whole on wall" / "whole in wall" -> "hole on wall"
    (r"\bwhole\b(?=\s+(on|in|of)\s+(the\s+)?wall)", "hole"),
    # "pant is not proper" -> "paint is not proper"
    (r"\bpant\b(?=\s+(is\s+not\s+proper|is\s+removed|is\s+gone|"
     r"not\s+proper))", "paint"),
    # "ploor" / "blore" -> "floor"  (used in voice locations)
    # (no defect mapping; just helps "stain on floor" match cleanly)
]


def _normalize_transcript(text: str) -> str:
    """Lower-case and apply Whisper-mishearing fixes."""
    if not text:
        return ""
    out = text.lower()
    for pat, repl in _HOMOPHONE_RULES:
        out = re.sub(pat, repl, out)
    return out


def _tag_defects(text: str) -> list[str]:
    """Return the canonical defect classes mentioned in ``text``."""
    if not text:
        return []
    norm = _normalize_transcript(text)
    found: list[str] = []
    for canonical, keywords in DEFECT_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"s?\b"
            if re.search(pattern, norm):
                found.append(canonical)
                break
    # Deduplicate, preserve order
    seen: set[str] = set()
    return [c for c in found if not (c in seen or seen.add(c))]


def transcribe_video(
    video_path: str | Path,
    model_size: str = "medium",
    keep_audio: bool = False,
    audio_out: str | Path | None = None,
    language: str | None = "en",
) -> TranscriptResult:
    """Extract audio from ``video_path`` and transcribe it with Whisper.

    On any failure (no audio, whisper unavailable, runtime error) a
    populated ``TranscriptResult`` with an ``error`` message is returned
    rather than raising, so the calling UI can keep going.
    """
    started = time.time()
    audio_path: Path | None = None
    try:
        audio_path = extract_audio(video_path, out_path=audio_out)
    except Exception as e:  # noqa: BLE001
        return TranscriptResult(
            has_audio=False, has_speech=False, language=None, text="",
            segments=[], audio_path=None,
            error=f"Audio extraction failed: {e}",
            elapsed_sec=time.time() - started,
        )

    if audio_path is None:
        return TranscriptResult(
            has_audio=False, has_speech=False, language=None, text="",
            segments=[], audio_path=None,
            error="No audio track / track too short.",
            elapsed_sec=time.time() - started,
        )

    model = _load_whisper(model_size)
    if model is None:
        if not keep_audio and audio_path.exists():
            audio_path.unlink(missing_ok=True)
            audio_path = None
        return TranscriptResult(
            has_audio=True, has_speech=False, language=None, text="",
            segments=[], audio_path=str(audio_path) if audio_path else None,
            error=("Whisper model could not be loaded "
                   f"({_WHISPER_LOAD_ERROR})."),
            elapsed_sec=time.time() - started,
        )

    try:
        # Force English by default.  Larger Whisper models (medium /
        # large) tend to misclassify Indian-English inspection audio
        # as Marathi / Hindi and emit garbled results -- pinning the
        # language fixes that.  Pass ``language=None`` to let Whisper
        # auto-detect (matches its old behaviour).
        result = model.transcribe(
            str(audio_path), fp16=False, verbose=False,
            language=language,
        )
    except Exception as e:  # noqa: BLE001
        if not keep_audio and audio_path.exists():
            audio_path.unlink(missing_ok=True)
            audio_path = None
        return TranscriptResult(
            has_audio=True, has_speech=False, language=None, text="",
            segments=[], audio_path=str(audio_path) if audio_path else None,
            error=f"Whisper transcription failed: {e}",
            elapsed_sec=time.time() - started,
        )

    raw_segments = result.get("segments") or []
    segments = []
    defect_mentions: dict[str, list[float]] = {}
    for s in raw_segments:
        text = _dedup_repeated_phrases((s.get("text") or "").strip())
        defects = _tag_defects(text)
        seg = Segment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=text,
            detected_defects=defects,
        )
        segments.append(seg)
        for d in defects:
            defect_mentions.setdefault(d, []).append(seg.start)

    full_text = _dedup_repeated_phrases((result.get("text") or "").strip())
    language = result.get("language")
    has_speech = bool(full_text)

    if not keep_audio and audio_path and audio_path.exists():
        audio_path.unlink(missing_ok=True)
        audio_path = None

    return TranscriptResult(
        has_audio=True,
        has_speech=has_speech,
        language=language,
        text=full_text,
        segments=segments,
        audio_path=str(audio_path) if audio_path else None,
        error=None,
        elapsed_sec=time.time() - started,
        defect_mentions=defect_mentions,
    )
