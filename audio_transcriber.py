"""
Voice-commentary transcription for inspection videos.

Pipeline
--------
1. ``ffmpeg`` extracts a 16-kHz mono WAV track from the video.
2. ``faster-whisper`` (CTranslate2 backend, ~4x faster than openai-whisper
   on CPU) transcribes the audio into time-stamped segments. Falls back
   to ``openai-whisper`` if the faster-whisper package is unavailable.
3. Built-in Silero VAD strips silent / non-speech regions to suppress
   the hallucinations that openai-whisper produced on long quiet clips.
4. Segments are tagged with whichever defect keywords they mention,
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
        # Indian-English inspection variants
        "very big crack", "major damage", "big damage", "huge crack",
        "severe crack", "serious crack", "dangerous crack",
        "wall is cracked badly", "wall has broken",
    ],
    "minor_crack": [
        "minor crack", "small crack", "hairline", "fine crack",
        "thin crack", "slight crack", "little crack",
        # Whisper often expands "crack" as "crick" or "quack" -- handled below
        "wall crack", "surface crack", "visible crack",
    ],
    "hole": [
        "hole", "holes", "puncture", "perforation", "hollow",
        "small hole", "tiny hole", "pin hole", "pinhole",
        # Indian-English
        "there is hole", "there is a hole", "hole in wall",
        "hole on wall", "damage hole", "broken hole",
        "opening in wall", "gap in wall",
    ],
    "spalling": [
        "spalling", "spalled", "spall", "concrete falling",
        "concrete coming off", "rebar exposed", "exposed rebar",
        "broken concrete", "chipped", "spalling is there",
        "concrete is broken", "plaster is falling", "plaster falling",
        "chunk missing", "material falling", "plaster is coming off",
    ],
    "peeling": [
        "peeling", "peel", "flaking", "flaked", "blistering",
        "delamination", "paint coming off",
        # Common Indian-English inspector phrasings:
        "paint is not proper", "paint not proper", "paint is removed",
        "paint removed", "paint is gone", "paint chipped",
        "paint is peeling", "no paint", "paint missing",
        "paint is bad", "paint is broken", "paint is damaged",
        "paint is worn", "paint has worn", "paint worn off",
        "paint is not there", "no paint on wall",
        "paint is falling", "Wall has no paint",
    ],
    "algae": [
        "algae", "moss", "fungus", "biological growth",
        "green growth", "green patch", "black spot", "black spots",
        "green stain", "green deposit", "green on wall",
        "mould", "mold", "damp growth",
    ],
    "stain": [
        "stain", "stained", "discolour", "discolor", "discoloration",
        "discolouration", "watermark", "rust mark", "efflorescence",
        "water stain", "damp stain", "brown stain", "dark stain",
        "yellow stain", "salt deposit", "white deposit",
    ],
    "water": [
        "water", "moisture", "damp", "leak", "leakage", "seepage",
        "wet patch", "water damage", "water seepage", "water leakage",
        "water is coming", "seepage is there",
    ],
    "general_crack": [
        "crack", "cracks", "cracking", "cracked", "fissure",
        # Whisper mishearings of "crack" with Indian accents
        "crick", "krack",
    ],
}


_WHISPER_MODEL = None
_WHISPER_BACKEND: str | None = None  # "faster" or "openai"
_WHISPER_LOAD_ERROR: Exception | None = None


@dataclass
class Word:
    """A single word with its timestamp (faster-whisper word_timestamps)."""
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    detected_defects: list[str] = field(default_factory=list)
    # Word-level timestamps when ``word_timestamps=True`` was passed
    # (faster-whisper backend only). Empty list on the openai-whisper
    # fallback or when VAD strips a segment to a single token.
    words: list[Word] = field(default_factory=list)


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

    Prefers ``faster-whisper`` (CTranslate2 + int8 on CPU is ~4x faster
    than openai-whisper at the same model size and uses ~half the RAM).
    Falls back to ``openai-whisper`` if faster-whisper isn't installed.

    The cache is keyed on ``model_size`` so callers can switch between
    "tiny" and "medium" at runtime and each size loads at most once
    per process.
    """
    global _WHISPER_MODEL, _WHISPER_LOAD_ERROR, _WHISPER_LOADED_SIZE
    global _WHISPER_BACKEND
    if _WHISPER_MODEL is not None and _WHISPER_LOADED_SIZE == model_size:
        return _WHISPER_MODEL
    # Different size requested -> drop the previous one and reload.
    _WHISPER_MODEL = None
    _WHISPER_LOAD_ERROR = None
    _WHISPER_BACKEND = None

    # Try faster-whisper first (preferred backend on CPU).
    try:
        from faster_whisper import WhisperModel
        # int8 keeps RAM low on CPU; cpu_threads=0 -> use all available.
        _WHISPER_MODEL = WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=0,
        )
        _WHISPER_LOADED_SIZE = model_size
        _WHISPER_BACKEND = "faster"
        return _WHISPER_MODEL
    except Exception as fw_err:  # noqa: BLE001
        _WHISPER_LOAD_ERROR = fw_err  # tentative; may be replaced below

    # Fallback to openai-whisper.
    try:
        import whisper  # openai-whisper
        _WHISPER_MODEL = whisper.load_model(model_size)
        _WHISPER_LOADED_SIZE = model_size
        _WHISPER_BACKEND = "openai"
        _WHISPER_LOAD_ERROR = None
        return _WHISPER_MODEL
    except Exception as e:  # noqa: BLE001
        _WHISPER_LOAD_ERROR = e
        _WHISPER_BACKEND = None
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
    # "paid is not proper" -> "paint is not proper"
    (r"\bpaid\b(?=\s+(is\s+not\s+proper|is\s+not\s+there|is\s+removed|"
     r"not\s+proper|is\s+bad|is\s+missing))", "paint"),
    # "pant" -> "paint" in inspection context
    (r"\bpant\b(?=\s+(is\s+not\s+proper|is\s+removed|is\s+gone|"
     r"not\s+proper|is\s+not\s+there))", "paint"),
    # "whole on wall" / "whole in wall" / just "whole wall" -> "hole ..."
    (r"\bwhole\b(?=\s+(on|in|of|in\s+the)\s+(the\s+)?wall)", "hole"),
    # "whole" followed immediately by "in" or "on" -> hole
    (r"\bwhole\b(?=\s+(is\s+there|is\s+seen|here))", "hole"),
    # "crick" -> "crack" (common mishearing)
    (r"\bcrick\b(?=\s*(s|ing|ed)?(\s|$))", "crack"),
    # "spallings" / "spalling is" are fine; catch "sparring"
    (r"\bsparring\b(?=\s+(on|in|of)\s+(the\s+)?wall)", "spalling"),
    # "peeping" -> "peeling" (Whisper mishearing)
    (r"\bpeeping\b", "peeling"),
    # "feeling" -> "peeling" when preceded by paint / wall context
    (r"(?<=paint\s)\bfeeling\b", "peeling"),
]


def _normalize_transcript(text: str) -> str:
    """Lower-case and apply Whisper-mishearing fixes."""
    if not text:
        return ""
    out = text.lower()
    for pat, repl in _HOMOPHONE_RULES:
        out = re.sub(pat, repl, out)
    return out


_KEYWORD_TO_CANONICAL: dict[str, str] = {
    # Two of the DEFECT_KEYWORDS top-level keys are convenience
    # *keyword buckets*, not canonical defect classes. We collapse them
    # onto the closest real class so the UI never surfaces non-defect
    # tokens like "general_crack" / "water" in the "defects: ..." tag
    # (they used to leak through unchanged). Kept in sync with
    # defect_analyzer.CLASS_ALIASES.
    "general_crack": "minor_crack",
    "water":         "stain",
}


def _tag_defects(text: str) -> list[str]:
    """Return the canonical defect classes mentioned in ``text``.

    Buckets that aren't real defect classes (``general_crack`` is just
    a catch-all crack keyword, ``water`` is a moisture cue) are mapped
    to their canonical equivalents via ``_KEYWORD_TO_CANONICAL`` so the
    report only ever lists real defect classes.
    """
    if not text:
        return []
    norm = _normalize_transcript(text)
    found: list[str] = []
    for canonical, keywords in DEFECT_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"s?\b"
            if re.search(pattern, norm):
                found.append(_KEYWORD_TO_CANONICAL.get(canonical, canonical))
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
        if _WHISPER_BACKEND == "faster":
            # faster-whisper streams a generator; we materialise it
            # below so the rest of the code path is identical to the
            # openai-whisper one.  vad_filter=True strips silence and
            # kills the long-quiet hallucinations the old backend
            # produced on inspection-pause clips.  word_timestamps=True
            # gives us per-word start/end which downstream uses to seek
            # straight to the moment a defect was mentioned (rather
            # than just the segment midpoint).
            seg_iter, info = model.transcribe(
                str(audio_path),
                language=language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                beam_size=5,
                word_timestamps=True,
            )
            raw_segments = [
                {
                    "start": float(s.start),
                    "end": float(s.end),
                    "text": s.text,
                    "words": [
                        {
                            "start": float(w.start),
                            "end": float(w.end),
                            "text": w.word,
                        }
                        for w in (s.words or [])
                    ],
                }
                for s in seg_iter
            ]
            detected_lang = info.language
        else:
            result = model.transcribe(
                str(audio_path), fp16=False, verbose=False,
                language=language,
            )
            raw_segments = result.get("segments") or []
            detected_lang = result.get("language")
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

    segments = []
    defect_mentions: dict[str, list[float]] = {}
    full_text_parts: list[str] = []
    for s in raw_segments:
        text = _dedup_repeated_phrases((s.get("text") or "").strip())
        defects = _tag_defects(text)
        words = [
            Word(
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                text=str(w.get("text") or ""),
            )
            for w in (s.get("words") or [])
        ]
        seg = Segment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", 0.0)),
            text=text,
            detected_defects=defects,
            words=words,
        )
        segments.append(seg)
        full_text_parts.append(text)
        for d in defects:
            defect_mentions.setdefault(d, []).append(seg.start)

    full_text = _dedup_repeated_phrases(" ".join(p for p in full_text_parts if p))
    language = detected_lang
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
