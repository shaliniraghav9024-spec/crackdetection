# Building Defect Inspector — YOLO + faster-whisper + sentence-transformers

A Streamlit web app that inspects building **videos** with spoken
commentary, detects defects with a YOLO ensemble, transcribes the
inspector's voice with **faster-whisper**, semantically matches
spoken phrases to defect classes via **sentence-transformers**, and
generates a downloadable **PDF inspection report** with annotated
keyframes, severity-ranked findings, and per-defect transcript quotes.

The pipeline is **fully automatic** — there are no manual sliders to
tweak. Just upload a video; everything below is decided per-clip from
the data itself:

| Auto-tuned | How it's chosen |
|---|---|
| Per-class confidence threshold | Adaptive: starts at `0.10` recall floor, raised to the per-class median when a class has many low-conf detections (`> 0.5/frame` and `median < 0.30`); voice-mentioned classes always use the floor. |
| NMS IoU | Fixed at `0.45`; downstream containment-dedup catches overlapping nested boxes that pure-IoU NMS misses. |
| Frame stride | `0.5 s` per sample; per-frame summary cards then collapse adjacent samples via time-bucket (1.5 s) + spatial-IoU dedup. |
| Whisper backend / size / language | `faster-whisper` `medium` int8, English forced; falls back to `openai-whisper` if the faster backend isn't installed. Override with `WHISPER_SIZE=large-v3` only if you have a GPU. |
| FP-filter pipeline | All 6 stages on by default (distractor mask → CLIP → paint-chip → implausible-hole → containment dedup). Set `DISABLE_FP_FILTERS=1` only when comparing un-filtered recall during a re-training cycle. |
| Frame-instance cap | 3 cards per video — picked by `(num_boxes desc, max_confidence desc)` after dedup, then re-sorted chronologically for display. |

The Streamlit sidebar surfaces only the defect class list and a status
line ("FP filters ON / OFF"). No knobs.

> **Detector ensemble:** local fine-tuned YOLO + 2 local crack
> detectors (Levanell, OpenSistemas) + auto-downloaded YOLOv8m
> pothole detector (`keremberke/yolov8m-pothole-segmentation`).
> **False-positive suppression:** YOLOv8m-COCO distractor mask + OpenCLIP
> zero-shot verifier + paint-chip reclassifier + implausible-hole filter
> (aspect ratio, edge proximity, continuous-dark-strip) + containment-
> based box consolidation.
> **Voice:** **faster-whisper** (CTranslate2 int8) — `medium` by default;
> falls back to openai-whisper if unavailable. English forced.
> **Mention matching:** sentence-transformers `all-MiniLM-L6-v2` cosine
> similarity (semantic) plus the existing keyword list.
> **Audio extraction:** `ffmpeg`.
> **Report:** ReportLab PDF.

---

## Pipeline at a glance

```
                          ffmpeg
                            │
                            ▼
video ──────────►   16 kHz mono WAV
            │               │
            │               ▼
            │       faster-whisper (word_timestamps=True, vad_filter=True)
            │               │
            │               ▼  segments + words
            │       ┌──────────────────────────────────┐
            │       │  defect_matcher                  │
            │       │  • all-MiniLM-L6-v2 embeddings   │
            │       │  • cosine sim ≥ 0.45             │
            │       │  • dedup buckets (3 s)           │
            │       └──────────────────────────────────┘
            │               │ DefectMention[]
            ▼               ▼
   ┌──────────────────────────────────────────────────────────┐
   │  YOLO ensemble (4 models)                                │
   │     primary best.pt  +  Levanell  +  OpenSistemas        │
   │     +  keremberke pothole (auto-downloaded from HF)      │
   │                                                          │
   │  ▼ per-class NMS  +  containment dedup                   │
   │  ▼ COCO distractor mask  (clock/book/phone/tie/...)      │
   │  ▼ OpenCLIP zero-shot verifier  (watch / ID card / ...)  │
   │  ▼ paint-chip reclassifier  (hole → peeling on bright)   │
   │  ▼ implausible-hole filter                               │
   │       AR band, edge margin, continuous-dark-strip        │
   └──────────────────────────────────────────────────────────┘
                            │
                            ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Per-frame summary cards (Streamlit)                     │
   │  • time-bucket dedup (1.5 s)                             │
   │  • spatial-IoU dedup (same scene → one card)             │
   │  • cap at 3 frames; full frame, all boxes drawn          │
   │                                                          │
   │  Live mention scan panel: streams each spoken mention    │
   │  with its evidence frame as it is found (st.status).     │
   └──────────────────────────────────────────────────────────┘
                            │
                            ▼
                 PDF report  ·  keyframes  ·  optional annotated MP4
```

Defect classes:

The **trained YOLO model** uses **6 classes** ([`classes.txt`](classes.txt) /
[`data.yaml`](data.yaml)). The runtime ensemble keeps the broader
schema below: `hole` is contributed by the auxiliary pothole detector
(no source data exists in the current training set), and `normal` is
the negative class.

| Class         | Severity | Source |
|---------------|----------|--------|
| `major_crack` | High     | trained YOLO (BD3 cracks) |
| `spalling`    | High     | trained YOLO |
| `minor_crack` | Medium   | trained YOLO (also catches `crack` via aliases) |
| `peeling`     | Medium   | trained YOLO + paint-chip reclassifier |
| `stain`       | Low      | trained YOLO |
| `algae`       | Low      | trained YOLO |
| `hole`        | High     | **auxiliary only** — keremberke pothole detector + 4-stage hole sanity filter; the trained YOLO no longer outputs this class |
| `normal`      | None     | Negative class — never reported |

[`yolo_classes.txt`](yolo_classes.txt) still lists the broader 8-class
set used by labelImg, so future hand-annotations of `hole` data can
drop straight back into the trained model.

`CLASS_ALIASES` in `defect_analyzer.py` folds in synonyms (`crack` →
`minor_crack`, `flaking` / `blistering` → `peeling`, etc.).

---

## Setup

```bash
# 1. Activate the environment
source venv/bin/activate

# 2. Install Python dependencies
pip install -r app_requirements.txt

# 3. Make sure ffmpeg is on PATH (required for audio extraction)
sudo apt install ffmpeg          # Ubuntu / Debian
# or:    brew install ffmpeg     # macOS
```

On first run the app downloads the following one-off:

| Asset | Size | Purpose |
|-------|------|---------|
| Whisper `medium` (CTranslate2)               | ~1.5 GB | faster-whisper voice transcription |
| `sentence-transformers/all-MiniLM-L6-v2`     | ~90 MB  | Semantic transcript→class matcher |
| `keremberke/yolov8m-pothole-segmentation`    | ~52 MB  | Pothole / hole-class auxiliary YOLO |
| OpenCLIP ViT-B/32                            | ~600 MB | Zero-shot defect / distractor verifier |
| `yolov8m.pt` (Ultralytics, on demand)        | ~50 MB  | YOLOv8m COCO weights — distractor mask |
| `models/levanell/yolov8n-seg-cracks-joints.pt` | ~7 MB | Local crack/joint segmenter (commit-tracked) |
| `models/opensistemas_n/yolov8n/weights/best.pt`| ~6 MB | Local generic crack detector (commit-tracked) |

Every model is loaded **lazily**: missing weights or download
failures degrade gracefully (the matcher returns no semantic mentions,
the pothole aux is silently skipped, etc.).

### Optional environment variables

| Variable | Default | Effect |
|---|---|---|
| `WHISPER_SIZE` | `medium` | `tiny` / `base` / `small` / `medium` / `large-v3`. Use `large-v3` only with a GPU — on CPU it is ~10× slower than `medium`. |
| `DISABLE_FP_FILTERS` | unset | Set to `1` to turn off the distractor mask + CLIP verifier (useful while re-training to compare unfiltered detection recall). |

---

## Data preparation pipeline

The training data lives at `~/building-defect-detection/dataset/` in YOLO
format (`{train,val}/{images,labels}`). The scripts below build it from
two upstream Roboflow exports — `merged_dataset` (6 source classes:
crack, leakage, abscission, corrosion, bulge, algae) and `bd3_dataset`
(single-class crack annotations) — into the 6-class project taxonomy.

```
~/property_inspector/merged_dataset    (6 raw classes)
~/property_inspector/bd3_dataset       (1 raw class)
            │
            ▼  merge_dataset.py     (MD5 dedup against bd3_dataset hashes)
            │
~/building-defect-detection/dataset_old/   (raw IDs preserved)
            │
            ▼  remap_labels.py      (6-class mapping, bbox clamping, intra-dedup)
            │
~/building-defect-detection/dataset_remapped/
            │
            ▼  merge_bd3.py             (BD3 → major_crack, dedup against existing)
            │  or, if BD3 was already in dataset_old:
            ▼  relabel_bd3_as_major_crack.py
            │
~/building-defect-detection/dataset/        (final, training-ready)
```

| Script | Purpose |
|---|---|
| [`merge_dataset.py`](merge_dataset.py) | Copy `merged_dataset` into the project, MD5-dedup against `bd3_dataset` |
| [`remap_labels.py`](remap_labels.py) | Map raw 6-class IDs to project taxonomy, clamp bboxes that extend past `[0,1]`, intra-dataset dedup |
| [`merge_bd3.py`](merge_bd3.py) | Copy BD3 cracks into the dataset as `major_crack`, hash-dedup against existing |
| [`relabel_bd3_as_major_crack.py`](relabel_bd3_as_major_crack.py) | One-shot in-place fix when BD3 was already present but mis-labelled (idempotent + atomic backup) |
| [`validate_dataset.py`](validate_dataset.py) | Read-only parallel validator: structure, MD5 duplicates, decode errors, missing/orphan labels, invalid YOLO coords, per-class distribution. Always run after every pipeline step. |

The 6-class taxonomy in [`data.yaml`](data.yaml) /
[`classes.txt`](classes.txt) is:

```
0=algae   1=major_crack   2=minor_crack   3=peeling   4=spalling   5=stain
```

```bash
# Validate the source
python validate_dataset.py --root ~/building-defect-detection/dataset_old \
    --num-classes 6 --workers 10 --report old_report.json

# Apply the 6-class mapping with bbox clamping + dedup
python remap_labels.py        # writes dataset_remapped/

# Merge BD3 cracks as major_crack
python merge_bd3.py           # appends to dataset_remapped/

# Re-validate the new dataset before training
python validate_dataset.py --root ~/building-defect-detection/dataset_remapped \
    --num-classes 6 --workers 10 --report new_report.json

# Atomic, reversible swap
mv ~/building-defect-detection/dataset            ~/building-defect-detection/dataset_legacy
mv ~/building-defect-detection/dataset_remapped   ~/building-defect-detection/dataset
```

Current train-set distribution after the pipeline (53,422 boxes total,
86.7% empty / negative images):

| ID | class | train | val | total | % of boxes |
|---:|---|---:|---:|---:|---:|
| 0 | algae       |    415 |    52 |    467 |  0.9% |
| 1 | major_crack |  1,268 |   133 |  1,401 |  2.6% |
| 2 | minor_crack | 14,400 | 1,656 | 16,056 | 30.1% |
| 3 | peeling     | 18,671 | 2,157 | 20,828 | 39.0% |
| 4 | spalling    |  7,383 |   965 |  8,348 | 15.6% |
| 5 | stain       |  5,584 |   738 |  6,322 | 11.8% |

`algae` and `major_crack` are severely under-represented — class-balanced
sampling, copy-paste augmentation (already on at `0.25` in the
hyperparameter profile), or focal-loss are required for usable per-class
recall on those two.

---

## Train your own primary detector

`train_yolo.py` defaults to `--model yolo11n.pt` (auto-downloaded from
Ultralytics on first use). Pre-flight checks run before training and
will refuse to start if `dataset/` has no annotations.

```bash
# CPU smoke test (verifies the pipeline; ~15-30 min on 12 cores)
yolo train model=yolo11n.pt data=data.yaml epochs=1 imgsz=320 batch=8 \
    device=cpu workers=4 fraction=0.02 \
    project=runs/detect name=smoketest_local

# GPU baseline — recommended first real run
python train_yolo.py --model yolo11s.pt --epochs 200 --imgsz 640 \
    --batch 32 --device 0 --name bd3_yolo11s_v1

# Custom 4-scale architecture (P2-P5 head; better for small cracks)
python train_yolo.py --cfg yolo11_emc.yaml --model yolo11n.pt \
    --imgsz 960 --batch 16 --device 0 --name bd3_yolo11_emc
```

Best weights land at `runs/detect/<name>/weights/best.pt`. The app
picks the **best-trained** `best.pt` automatically — ranked by
`epochs × imgsz`. The UI surfaces the resolved architecture
(YOLOv8 / YOLO11 / etc.) so you always know which model is loaded.

If no fine-tuned run exists the app falls back to `yolov8m.pt` (stock
COCO weights — has no defect classes), and the report is carried
mostly by the auxiliary crack + pothole models until you train.

### Cloud-GPU workflow

CPU-only training on the full 117 k-image dataset is impractical
(days–weeks). Two cloud paths are scaffolded; both reuse the same tar
package and run the same `train_yolo.py`.

```bash
# Pack project + dataset (~3.5 GB tar, no compression — JPEGs already shrink-resistant)
./pack_dataset.sh                  # → ../bd3_cloud_pkg.tar  + sha256 checksum
```

| Path | When | Files |
|---|---|---|
| **Google Colab** | First training run; free T4 (12-hr session) or paid L4/A100 | [`bd3_train_colab.ipynb`](bd3_train_colab.ipynb) — mounts Drive, extracts tarball, patches `data.yaml`, re-validates, trains, copies `runs/<name>/` back to Drive, optional ONNX export |
| **vast.ai** | Cheaper sustained runs ($0.20–0.40 / hr for RTX 3090/4090/A10) | [`deploy_vastai.sh`](deploy_vastai.sh) — rsyncs tarball + run script to remote, launches training inside `tmux`. [`vastai_remote_train.sh`](vastai_remote_train.sh) runs on the instance; configurable via env vars (`MODEL=yolo11m.pt EPOCHS=300 ...`) |

```bash
# vast.ai end-to-end
./pack_dataset.sh
./deploy_vastai.sh root@ssh4.vast.ai 12345
ssh -p 12345 root@ssh4.vast.ai 'tail -f /workspace/train.log'
rsync -avh --progress -e "ssh -p 12345" \
    root@ssh4.vast.ai:/workspace/bd3/runs/ ./runs_vastai/
```

Both cloud scripts re-run [`validate_dataset.py`](validate_dataset.py)
after extraction so corrupt uploads are caught before any GPU time is
spent.

### Honest note on accuracy

Final accuracy depends on:

- how many **annotated** images you have per class,
- class balance — see the table above; algae @ 0.9% and major_crack
  @ 2.6% are the bottlenecks,
- image resolution and motion blur,
- whether you can afford `yolo11m.pt` over `yolo11n.pt`.

`runs/detect/<name>/results.png` and the per-class confusion matrix
show which class is dragging the average. **More annotated examples**
of the weak class beats more epochs almost every time.

---

## Run the app

```bash
source venv/bin/activate
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. The sidebar shows only:

- The defect class list with severities.
- Whether false-positive filters are on.

Everything else — confidence floor, NMS IoU, frame-sampling rate,
Whisper backend / size / language, audio toggle — is **auto-tuned per
video**. The defaults live as constants at the top of [`app.py`](app.py)
under `# auto-tuned defaults`. To override one without editing code:

```bash
WHISPER_SIZE=large-v3 streamlit run app.py        # GPU-only; CPU is too slow
DISABLE_FP_FILTERS=1 streamlit run app.py         # un-filtered comparison run
TRANSFORMERS_VERBOSITY=error streamlit run app.py # silence transformers spam
```

The page produces, in order:

1. **🎙️ Live mention scan** — `st.status` panel that streams each
   semantic mention found in the transcript and the evidence frame
   YOLO localised for it (red box drawn within ±2.5 s window at 4 fps).
2. **📊 Defect Summary** — flat list of unique frames containing any
   defect (after time + spatial dedup), capped at 3, each with all
   boxes drawn together and the nearest transcript line beneath.
3. **🔍 Defect Findings (per-defect detail)** — keyframe + severity +
   confidence + recommendation + matched transcript per defect class.
4. **🎙️ Voice Mentions Not Visually Confirmed** — items the inspector
   named but the visual detector did not back up. Shown for review,
   never counted as detected defects.
5. **📝 Voice Transcript** — full Whisper output with per-segment
   defect tags.
6. **📄 Inspection Report** — download button for the PDF.

---

## False-positive filter chain

These run inside [`defect_analyzer._yolo_predict`](defect_analyzer.py)
on every frame's detections, in order:

| Stage | Catches | Action |
|---|---|---|
| 1. Per-class NMS  | Standard duplicate boxes | merge |
| 2. **Containment dedup** | One big covering box stacked with smaller boxes inside it (the "5 boxes on one crack" pattern) — IoU is low so NMS misses | drop the inner ones |
| 3. COCO distractor mask | Wall clocks, books, phones, ties, handbags, suitcases, bottles, persons, signs, keyboards | drop overlapping defect boxes |
| 4. OpenCLIP zero-shot | Wristwatches, ID cards, logos, framed pictures (text-prompt mismatch) | drop |
| 5. Paint-chip reclassifier | Box interior is brighter than ring **or** highly saturated → it's exposed primer, not a void | relabel `hole` → `peeling` |
| 6. Implausible-hole filter | Hole boxes that fail any of: aspect-ratio in `[0.35, 2.80]`, ≥15 px from frame edge, **not** part of a continuous dark strip extending above and below | drop |

All thresholds are constants near the top of `defect_analyzer.py` —
search for `_HOLE_AR_MIN`, `_HOLE_EDGE_MARGIN_PX`, `containment_thresh`.

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
python predict_yolo.py --weights runs/detect/<run>/weights/best.pt \
    --imgsz 960 path/to/file
```

Output goes to `output/yolo/<run_name>/`:

- `*_annotated.{jpg,mp4}` — boxes drawn on the input
- `keyframes/<class>_t<seconds>.jpg` — per-class best frames (videos)
- `<stem>.csv` — flat detection log
- `summary.json` — index of everything written

---

## Output layout (Streamlit app)

Every analysis run writes a fresh subfolder under `output/app_runs/`:

```
output/app_runs/run_XXXX/
├── input.{mp4,...}
├── annotated.mp4                  # only if save-annotated-video is on
├── keyframes/                     # one image per detected defect
│   ├── hole_t1.01.jpg             # visual detection
│   ├── peeling_voice_t14.55.jpg   # voice-only fallback
│   └── ...
├── summary_instances/             # full-frame cards for the summary section
│   └── any_inst_t<sec>.jpg
├── mention_NN_<class>.jpg         # per-mention evidence frames
└── report.pdf
```

---

## Project layout

```
app.py                  # Streamlit front-end (auto-tuned, video-only)
defect_analyzer.py      # YOLO ensemble + post-filters + frame-instance renderers
                        #  - render_class_frame_instances
                        #  - render_any_defect_frame_instances
                        #  - find_evidence_frame_around_time
                        #  - 6-stage post-filter pipeline
audio_transcriber.py    # ffmpeg + faster-whisper (with openai-whisper fallback)
                        #  - word_timestamps, vad_filter, homophone normalisation
defect_matcher.py       # sentence-transformers semantic transcript matcher
                        #  - DefectMention dataclass, DEFECT_DESCRIPTIONS
clip_verifier.py        # OpenCLIP zero-shot defect / distractor scorer
yolo_distractors.py     # YOLOv8m-COCO distractor detection + mask
report_generator.py     # ReportLab PDF rendering
predict_yolo.py         # CLI tester (images / folders / videos)
train_yolo.py           # YOLO fine-tune script (defaults to yolo11n.pt)
tune_hyperparams.py     # Ray Tune hyperparameter search (mAP-driven)
yolo11_emc.yaml         # 4-scale (P2-P5) custom architecture for small defects
augment_config.py       # Albumentations pipeline (CLAHE / blur / weather)
                        #   auto-picked up by Ultralytics if installed

# --- data pipeline (raw datasets → training-ready 6-class) ---
merge_dataset.py                   # merged_dataset → dataset/   (MD5 dedup)
remap_labels.py                    # 6-class mapping + bbox clamp + dedup
merge_bd3.py                       # BD3 cracks → dataset_remapped as major_crack
relabel_bd3_as_major_crack.py      # one-shot in-place fix (idempotent)
validate_dataset.py                # parallel read-only validator
prepare_yolo_dataset.py            # legacy BD3 → YOLO dataset prep

# --- evaluation ---
validate_and_compare.py # side-by-side mAP comparison across checkpoints

# --- cloud training ---
pack_dataset.sh           # tar dataset + scripts for upload (~3.5 GB)
bd3_train_colab.ipynb     # Colab notebook (Drive mount + train + save back)
deploy_vastai.sh          # rsync tar + launch training over SSH inside tmux
vastai_remote_train.sh    # remote-side train script (env-var configurable)

# --- config ---
data.yaml               # YOLO dataset config (6 training classes + paths)
classes.txt             # Canonical 6-class training list
yolo_classes.txt        # Broader 8-class list used by labelImg (annotation-side)

runs/detect/            # Ultralytics training output (weights, plots, logs)
models/                 # Auxiliary detectors
  ├── levanell/                       (commit-tracked)
  ├── opensistemas_n/                 (commit-tracked)
  └── keremberke_pothole_m/           (auto-downloaded on first run)
output/app_runs/        # Per-session app outputs (PDFs, keyframes)
dataset/                # Active training set (6 classes)
dataset_old/            # Untouched copy with raw merged_dataset class IDs
                        #   — kept as a safety net for re-running remap
```

---

## Why an ensemble, not a single model?

A single fine-tuned YOLO on a small in-house defect set will:

- **miss** subtle hairline cracks (not enough training examples), and
- **hallucinate** "peeling" or "minor_crack" boxes on watches, signs,
  the INDIA-flag card, doorframes, etc., because the negative
  distribution during training is too narrow.

The ensemble + filter chain trades latency for far fewer of those
failure modes:

| Layer | Catches |
|-------|---------|
| Local fine-tuned YOLO | The 6 defect classes in *your* footage (algae, major_crack, minor_crack, peeling, spalling, stain). |
| Levanell crack/joint segmentation | Real crack patterns missed by the local model; explicitly differentiates `joint` (architectural seam) from `crack`. |
| OpenSistemas crack-seg | Generic crack patterns from a 4 k-image training set. |
| **keremberke YOLOv8m pothole** | Sole vote for the `hole` class. The trained YOLO has no source data for `hole`, so this aux model's output is the only way `hole` ever appears in a report. |
| YOLOv8m COCO distractor mask | Clocks, TVs, signs, books, cell phones, ties, handbags, persons. |
| OpenCLIP zero-shot verifier | Long-tail distractors COCO doesn't know about (national flags, badges, decals). |
| Paint-chip reclassifier | Bright/saturated "hole" boxes that are really exposed primer → `peeling`. |
| Implausible-hole filter | Wall seams, doorframes, baseboards mis-detected as holes. |
| Containment dedup | Stacks of boxes around a single crack collapse to one. |

---

## Whisper accuracy notes

`faster-whisper` (CTranslate2 with int8 on CPU) is the primary backend:
about 4× faster than `openai-whisper` at the same model size and roughly
half the RAM. That makes the `medium` model affordable on CPU — the
smallest size that **stops mishearing inspection vocabulary** on real
Indian-English commentary. With `tiny` / `base` we routinely got:

- "paint is not proper" → "tent is not proper"
- "hole on the wall" → "whole on the wall"

Both are still corrected by the homophone rules in
`audio_transcriber.py`, but they are far less common with `medium`.
The English-language pin (`language="en"`) is essential — on the same
audio, auto-detect Whisper-medium classifies the speaker as **Marathi**
and emits garbled output. Pinning English costs nothing on English
audio.

`openai-whisper` is kept as a fallback so the project still runs if
faster-whisper fails to import or download.

---

## Further reading

- [`data.yaml`](data.yaml) — dataset configuration consumed by training.
- [`classes.txt`](classes.txt) — canonical defect class list.
- [`defect_matcher.py`](defect_matcher.py) — tunable knobs for the
  semantic mention matcher (`SIMILARITY_THRESHOLD`, `WINDOW_SECONDS`,
  `SAMPLE_FPS`, `DEDUP_BUCKET_SECONDS`).
