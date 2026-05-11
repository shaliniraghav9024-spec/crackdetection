# User Journey Map — Building Defect Inspector

This journey map describes how a building inspector moves through the
**Building Defect Inspector** app, from arriving on site with a phone
camera to handing a stakeholder a finished PDF report.

---

## Personas

| Persona | Role | Primary goal |
|---------|------|--------------|
| **Asha — Field Inspector** | Walks the building, films defects, narrates as she goes. | Capture every visible defect on video without stopping to write notes. |
| **Ravi — Site Engineer** | Reviews inspection footage at his desk and signs off repair plans. | Get an objective, ranked list of defects with evidence images. |
| **Meera — Project Manager** | Shares status with the client and scopes repair work. | A clean PDF/HTML report she can forward without editing. |

---

## End-to-end journey

| Stage | What the user does | What the app does | User feels | Pain points the app removes |
|------|--------------------|-------------------|------------|------------------------------|
| 1. **Capture** | Asha walks the site filming a video on her phone, narrating defects out loud (e.g. *"major crack on the left column"*). | — (offline, on device) | Focused; hands-free | No clipboard, no per-photo annotation. |
| 2. **Launch** | `streamlit run app.py` → opens `http://localhost:8501`. | Loads the fine-tuned YOLO11 weights (auto-resolved from `runs/detect/*/weights/best.pt`, fallback `yolov8m.pt`) plus the aux-model ensemble. | Confident the model is ready | One command, no notebook setup. |
| 3. **Configure** | In the sidebar she picks confidence threshold, frame-sampling interval, and Whisper model size (`tiny` / `base` / `small`). | Stores settings; warms the classifier. | In control of speed-vs-accuracy | Sensible defaults — no tuning needed for a first pass. |
| 4. **Upload video** | Drag-and-drops the `.mp4` from her phone. | Saves to `output/app_runs/run_XXXX/input.mp4` and shows the video player. | Reassured (sees her own footage) | No format wrangling — `ffmpeg` handles the rest. |
| 5. **Frame analysis** | Watches the progress bar. | Samples frames every `0.5s`, runs the YOLO ensemble (fine-tuned YOLO11 + Levanell + OpenSistemas + keremberke pothole), suppresses false positives via the distractor mask + CLIP verifier + paint-chip / implausible-hole filters, then keeps per-class keyframes. | Patient — feedback is visible | No silent waiting; per-frame progress. |
| 6. **Voice transcription** | (optional) Leaves Whisper enabled. | `ffmpeg` strips audio → 16 kHz mono WAV → Whisper → time-stamped segments → keyword scan for defect terms. | Surprised it picked up her commentary | Her narration becomes searchable evidence. |
| 7. **Review summary** | Scans the **Defect Summary** dashboard: counts per class, severity pills, max-confidence values. | Aggregates per-class stats: frames seen, max confidence, first/last appearance, representative keyframe. | Oriented in seconds | No frame-by-frame scrubbing. |
| 8. **Drill into findings** | Opens **Defect Findings** to see the marked-up keyframe and the voice lines that overlap that defect's timestamp. | Cross-references CAM-marked frames with Whisper segments; highlights defect mentions. | Trusts the result (image + her own words back it up) | Evidence is co-located with audio context. |
| 9. **Export report** | Clicks **Download PDF** (or HTML / JSON). | `report_generator.py` renders `report.html`, `report.pdf`, `report.json` with embedded keyframes into `output/app_runs/run_XXXX/`. | Done. | One artifact she can email; no manual write-up. |
| 10. **Hand-off** | Meera attaches the PDF to the client email; Ravi opens the JSON in his repair-planning tool. | — | Professional output | Same run feeds humans (PDF) and tools (JSON). |

---

## Journey diagram

```
   Capture            Configure & Upload         Analyze                Report
 ┌────────────┐     ┌────────────────────┐    ┌────────────────────┐  ┌────────────┐
 │ phone film │     │ sidebar settings   │    │ frame sampling     │  │ HTML       │
 │ + voice    │ ──► │ + drag-drop .mp4   │──► │ YOLOv8 + aux       │─►│ PDF        │
 │ narration  │     │                    │    │ ffmpeg + Whisper   │  │ JSON       │
 └────────────┘     └────────────────────┘    └────────────────────┘  └────────────┘
                                                       │
                                                       ▼
                                          per-defect keyframes,
                                          severity, timestamps,
                                          and matching voice lines
```

---

## Emotional curve

```
  high │                                           ●  hand-off
       │                                       ●     report ready
       │                              ●     drill-in matches her voice
  ok   │   ●  launch          ●  upload
       │       ●  configure
  low  │                          ●  waiting on frames (mitigated by progress bar)
       └──────────────────────────────────────────────────────────────────────
         t1     t2          t3       t4       t5         t6         t7
```

The two risk points are **stage 5** (long compute on CPU) and **stage 6**
(Whisper download / transcription on first run). Both are mitigated by
visible progress, cached models, and a configurable Whisper size.

---

## Where each artifact comes from

| Artifact | Produced by | Code |
|----------|-------------|------|
| Per-frame defect bounding boxes + labels | YOLOv8 ensemble (fine-tuned + Levanell + OpenSistemas + keremberke pothole) | [defect_analyzer.py](defect_analyzer.py) |
| Time-stamped voice segments + defect keyword tags | `ffmpeg` + OpenAI Whisper | [audio_transcriber.py](audio_transcriber.py) |
| HTML / PDF / JSON inspection report | Jinja-rendered templates | [report_generator.py](report_generator.py) |
| The whole interactive flow above | Streamlit UI | [app.py](app.py) |

## Models behind each stage

| Stage | Model | Why this model |
|-------|-------|----------------|
| 5. Frame analysis (label + region) | **YOLOv8** fine-tuned on BD3 (`yolov8n/s/m.pt`, COCO pretrained) + 3 aux detectors (Levanell, OpenSistemas, keremberke pothole) | True bounding boxes per defect; ensemble fills gaps where the in-house weights are weak. |
| 5b. False-positive suppression | YOLOv8m COCO distractor mask + OpenCLIP zero-shot verifier + paint-chip / implausible-hole filters | Drops boxes that fall on watches, clocks, signs, paint chips, etc. |
| 6. Voice transcription | **OpenAI Whisper** (`tiny` / `base` / `small`) | Robust on noisy on-site audio; size is user-selectable for speed vs. accuracy. |
| 6. Audio extraction | **ffmpeg** | Standardizes any phone-camera audio to 16 kHz mono WAV before Whisper. |

Full per-model details: see [README.md](README.md).

---

## Success criteria per stage

- **Capture:** Asha never has to pause filming to take notes.
- **Configure → Upload:** ≤ 30 s from launching the app to a running analysis.
- **Analyze:** Every defect class visible in the footage shows up with at least one keyframe above the confidence threshold.
- **Review:** Ravi can identify the top three defects without scrubbing the video.
- **Report:** Meera can forward the PDF to a client without re-formatting it.
