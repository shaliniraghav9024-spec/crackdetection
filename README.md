# Building Defect Inspector — YOLOv8m + Whisper-medium

A Streamlit web application that detects building defects in inspection
**videos** and **photos**, transcribes any voice commentary with
**OpenAI Whisper**, and generates a downloadable **PDF inspection
report** with marked-up keyframes, severity-ranked findings, and the
inspector's quotes synchronised to each defect.

The pipeline is voice-first and class-aware: the inspector's words
guide the detector and, when YOLO can't visually localise a defect the
inspector named (e.g. faint hairline cracks, peeling paint that blends
with the wall), a class-specific OpenCV evidence scorer + CLIP
zero-shot verifier picks the most representative frame from the whole
video and surfaces it with a "voice-only" badge.

> **Detector ensemble:** local fine-tuned YOLOv8 + 2 pretrained crack
> detectors from Hugging Face (Levanell, OpenSistemas).
> **Distractor filter:** YOLOv8m on COCO (clocks / TVs / signs /
> books / phones) + OpenCLIP zero-shot verifier.
> **Voice:** OpenAI Whisper (`tiny` / `base` / `small` / **`medium`**
> (default) / `large`), language pinned to English by default.
> **Audio extraction:** `ffmpeg`.
> **Report:** ReportLab PDF with annotated keyframes per defect.

---

## Pipeline at a glance

```
                                 ┌── voice hints ──┐
            ffmpeg ─► Whisper ───┤                 │
                       │         └── transcript ───┤
video ─►   ┌───────────┴──────────────────────────┐│
           │                                      ▼▼
           │   ┌──────────────────────────────────────────────────┐
           │   │  YOLO ensemble (primary + 2 crack-detector aux)  │
           │   │       │                                          │
           │   │       ▼                                          │
           │   │  YOLOv8m-COCO distractor mask                    │
           │   │  (clock / sign / book / phone → drop)            │
           │   │       │                                          │
           │   │       ▼                                          │
           │   │  OpenCLIP zero-shot verifier                     │
           │   │  (defect prompts vs. distractor prompts)         │
           │   └──────────────────────────────────────────────────┘
           │              │
           │              ▼
           │   ┌──────────────────────────────────────────────────┐
           │   │  Voice-only evidence pass (for hints YOLO missed)│
           │   │  • class-aware visual evidence scoring           │
           │   │    (peeling / hole / crack / stain / algae)      │
           │   │  • global rescue scan if local evidence is weak  │
           │   │  • CLIP-verified bbox or quote-only fallback     │
           │   └──────────────────────────────────────────────────┘
           ▼
        PDF report  ·  annotated MP4  ·  per-defect keyframes
```

Defect classes ([`classes.txt`](classes.txt) / [`data.yaml`](data.yaml)):

| Class         | Severity | Source(s) |
|---------------|----------|-----------|
| `major_crack` | High     | YOLO ensemble |
| `spalling`    | High     | YOLO ensemble |
| `hole`        | High     | YOLO ensemble + heuristic dark-blob scorer |
| `minor_crack` | Medium   | YOLO ensemble + heuristic crack scorer |
| `peeling`     | Medium   | YOLO ensemble + heuristic LAB-distance scorer |
| `stain`       | Low      | YOLO ensemble + heuristic LAB-distance scorer |
| `algae`       | Low      | YOLO ensemble + heuristic green-mask scorer |
| `normal`      | None     | Negative training class — never reported |

Generic synonyms are folded in automatically via `CLASS_ALIASES` in
`defect_analyzer.py` (`crack` → `minor_crack`, `flaking` /
`blistering` → `peeling`, `concrete_falling` → `spalling`, etc.) so
inspectors can use natural language and the model's prompt pool
overlap stays small.

---

## Setup

```bash
# 1. Activate the environment
source venv/bin/activate

# 2. Install dependencies (Streamlit, Ultralytics, Whisper, OpenCLIP, ReportLab)
pip install -r app_requirements.txt

# 3. Make sure ffmpeg is installed (audio extraction)
sudo apt install ffmpeg          # Ubuntu / Debian
# or:    brew install ffmpeg     # macOS
```

On first run the app auto-downloads:

| Asset | Size | Purpose |
|-------|------|---------|
| `yolov8m.pt`                                                    | ~50 MB  | YOLOv8m COCO weights — distractor mask + fallback |
| Whisper `medium` (`~/.cache/whisper/medium.pt`)                 | ~1.4 GB | Voice transcription |
| OpenCLIP ViT-B/32 (`~/.cache/huggingface/`)                     | ~600 MB | Zero-shot defect verifier |
| `models/levanell/yolov8n-seg-cracks-joints.pt`                  | ~7 MB   | Auxiliary crack/joint segmenter |
| `models/opensistemas_n/yolov8n/weights/best.pt`                 | ~6 MB   | Auxiliary general crack detector |

The two auxiliary models live under `models/` — drop the `.pt` files
into the listed paths to enable them. The pipeline silently skips
any auxiliary whose weights are missing, so the project still works
without them.

---

## Train your own primary detector

```bash
# Default recipe: yolov8m.pt, 150 epochs, strong augmentation,
# cosine LR, early stop on mAP50 with patience 25.
python train_yolo.py

# Tweaks
python train_yolo.py --epochs 200 --imgsz 640 --batch 16
python train_yolo.py --model yolov8s.pt   # smaller / CPU-friendlier
python train_yolo.py --hsv-v 0.6 --erasing 0.5   # more brightness/erasing
```

Pre-flight checks run before training:
- `data.yaml` exists and is valid
- `dataset/{train,val}/images` contains image files
- at least one label `.txt` is non-empty (the classic "I forgot to draw
  any boxes" mistake)

Best weights land at `runs/detect/<name>/weights/best.pt`. The app
picks the **most recent** `best.pt` automatically; if none exists it
falls back to the auto-downloaded `yolov8m.pt` (which has no defect
classes, so the report relies on the auxiliary crack detectors and
the voice-only evidence pass until you fine-tune).

### Augmentation recipe (defaults)

| Augmentation | Default | Why |
|--------------|---------|-----|
| HSV jitter (h/s/v) | 0.015 / 0.70 / 0.40 | Different lighting, paint colours |
| Mosaic | 1.0 | Multiple defects per training image |
| Mixup | 0.10 | Robustness on small datasets |
| Erasing | 0.40 | Forces multi-cue learning |
| Affine (degrees / scale / shear) | 10° / 0.5 / 2 | Camera-angle robustness |
| Flip LR | 0.5 | Horizontal symmetry of walls |
| Albumentations Blur / MedianBlur / ToGray / CLAHE | auto | Blurry / low-light frames |
| Cosine LR + 3-epoch warmup | on | Smoother convergence on small data |
| Early stop on mAP50 | patience 25 | Stop when val plateaus |

### Honest note on accuracy

I **don't promise 95 %+ mAP on BD3 val.** Final accuracy depends on:

- how many annotated images you have per class,
- class balance (rare classes like `hole` need more examples),
- image resolution and motion blur,
- whether you can afford `yolov8m.pt` over `yolov8s.pt`.

The training script prints the realised metrics each epoch. If
`mAP50` plateaus below your target on a given class, the next move is
**more annotated examples of that class**, not more epochs. Use
`runs/detect/<name>/results.png` and the per-class confusion matrix to
see exactly which class is dragging the average.

---

## Run the app

```bash
source venv/bin/activate
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. Sidebar controls:

| Setting | Effect |
|---------|--------|
| **Input type** (Video / Image) | Switches the upload type and the relevant controls. |
| **Confidence threshold** | YOLO box must score ≥ this to be kept. |
| **NMS IoU threshold** | Non-max-suppression cut-off. |
| **Image tile grid (N×N)** | Image-only. Tile a high-res photo so multiple co-occurring defects each get their own box. |
| **Sample one frame every N s** | Video-only. Lower = more recall, slower. |
| **Transcribe voice commentary** | Video-only. ffmpeg + Whisper. |
| **Whisper model size** | `tiny` / `base` / `small` / **`medium`** / `large`. Default `medium` — drastically fewer mishearings than `tiny`. |
| **Voice commentary language** | **English (forced)** by default. Whisper-medium / large tend to misclassify Indian-English audio as Hindi / Marathi unless pinned. Switch to *Auto-detect* for non-English commentary. |
| **Save annotated video** | Video-only. Writes an MP4 with boxes drawn on every sampled frame. |
| **Suppress clocks / signs / watches / phones** | Toggles the YOLOv8m-COCO distractor mask. |
| **Zero-shot CLIP verifier** | Toggles the OpenCLIP verification step that rejects boxes whose patch matches a "clock / sign / flag / book" prompt better than a defect prompt. |

The page shows: an annotated preview, a defect summary table
(severity-ranked), per-defect findings with the marked-up keyframe and
matching voice lines, the full transcript, and a download button for
the **PDF report**.

### Voice-only fallback (what the badge means)

When the inspector says *"there is a hole on the wall"* but YOLO can't
visually localise it, the report shows the entry with a 🎙️ **voice-only**
badge. The keyframe is chosen by:

1. Sampling frames in a wide window around the spoken time
   (-12 s / +6 s — inspectors often pan past a defect *before* describing it).
2. Scoring each candidate by a class-specific OpenCV evidence scorer
   (LAB-distance for peeling / stain, dark-blob compactness for holes,
   long-thin contour ratio for cracks, green-mask area for algae).
3. If local evidence is weak, scanning the **whole video** at 0.3 s
   spacing for the strongest evidence frame for that class.
4. Verifying the candidate bbox with OpenCLIP — if the patch reads as
   "a clock / a sign / a flag", the bbox is dropped. With no
   CLIP-verified bbox we still surface the sharpest frame near the
   voice mention with the inspector's quote stamped on it, but **no
   misleading box** is drawn.

This is what the user feedback boiled down to: *"keep analysis both
but balancing with proper frames and voice transcription"*. Voice
hints inform detection, but visual evidence is what ends up in the
report keyframe.

---

## CLI

```bash
# image
python predict_yolo.py path/to/photo.jpg --conf 0.30 --grid 3

# folder of images
python predict_yolo.py path/to/folder/

# video
python predict_yolo.py path/to/inspection.mp4 \
    --conf 0.30 --every-n-seconds 0.5 --save-video

# explicit weights + larger imgsz
python predict_yolo.py --weights runs/detect/bd3_yolov8m/weights/best.pt \
    --imgsz 960 path/to/file
```

Output goes to `output/yolo/<run_name>/`:
- `*_annotated.jpg` / `*_annotated.mp4` — boxes drawn on the input
- `keyframes/<class>_t<seconds>.jpg` — per-class best frames (videos)
- `<stem>.csv` — flat detection log
- `summary.json` — index of everything written

---

## Output layout (Streamlit app)

Every analysis run writes a fresh subfolder under `output/app_runs/`:

```
output/app_runs/run_XXXX/
├── input.{mp4,jpg,...}
├── annotated.jpg | annotated.mp4   # image mode | video mode (if enabled)
├── keyframes/                      # one image per detected defect (videos)
│   ├── hole_t1.01.jpg              # visual detection
│   ├── peeling_voice_t14.55.jpg    # voice-only fallback
│   └── ...
├── crops/                          # one annotated crop per defect (images)
└── report.pdf
```

---

## Project layout

```
app.py                  # Streamlit front-end (image + video)
defect_analyzer.py      # YOLO ensemble, distractor mask, CLIP verifier,
                        # voice-only evidence pass, image tiling, video sampling
audio_transcriber.py    # ffmpeg + Whisper voice transcription with
                        # homophone normalisation (paint↔tent, hole↔whole)
clip_verifier.py        # OpenCLIP zero-shot defect / distractor scorer
yolo_distractors.py     # YOLOv8m-COCO distractor detection + mask
report_generator.py     # ReportLab PDF rendering
predict_yolo.py         # CLI tester (images / folders / videos)
train_yolo.py           # YOLOv8 fine-tune script (default: yolov8m.pt)
prepare_yolo_dataset.py # BD3 → YOLO dataset prep
data.yaml               # YOLO dataset config (classes + paths)
classes.txt             # Canonical 8-class list
yolo_classes.txt        # Same 8 classes, used by labelImg
runs/detect/            # Ultralytics training output (weights, plots, logs)
models/                 # Drop-in auxiliary detectors (Levanell, OpenSistemas)
output/app_runs/        # Per-session app outputs (PDF reports, keyframes)
```

---

## Why an ensemble, not a single model?

A single fine-tuned YOLOv8 on a small in-house defect set will:

- **miss** subtle hairline cracks (not enough training examples), and
- **hallucinate** "peeling" or "minor_crack" boxes on watches, signs,
  the INDIA-flag card, etc., because the distribution of negatives
  during training is too narrow.

The ensemble trades a bit of latency for far fewer of those failure
modes:

| Layer | Catches |
|-------|---------|
| Local fine-tuned YOLO | The 8 defect classes specific to *your* footage. |
| Levanell crack/joint segmentation | Real crack patterns missed by the local model; explicitly differentiates `joint` (architectural seam) from `crack`. |
| OpenSistemas crack-seg | Generic crack patterns from a 4 k-image training set. |
| YOLOv8m COCO distractor mask | Clocks, TVs, signs, books, cell phones, scissors — masked out before any defect heuristic runs. |
| OpenCLIP zero-shot verifier | Catches the long-tail distractors COCO doesn't know about (national flags, badges, decals, tiny posters) by comparing patch embeddings against `"a clock / a sign / a flag / a smartwatch"` prompts. |
| Voice-only evidence pass | Reports defects the inspector named but YOLO missed, using class-specific OpenCV scoring + a global frame-rescue scan + CLIP gating. |

---

## Whisper accuracy notes

Whisper-medium is the new default because it was the smallest model
that **stopped mishearing inspection vocabulary** on real Indian-English
commentary. With `tiny` / `base` we routinely got:

- "paint is not proper" → "tent is not proper"
- "hole on the wall" → "whole on the wall"

Both are still corrected by the homophone rules in
`audio_transcriber.py`, but they're far less common with `medium`
(Whisper transcribes *"There is some hole on wall."* and *"There is a
crack in the wall."* cleanly). The English-language pin is essential —
on the same audio, auto-detect Whisper-medium classifies the speaker as
**Marathi** and emits garbage like *"linkage"*, *"savage"*. Pinning
the language costs nothing on English audio.

---

## Further reading

- [data.yaml](data.yaml) — dataset configuration consumed by training.
- [classes.txt](classes.txt) — canonical defect class list.
