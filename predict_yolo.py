"""
Run a trained YOLOv8 detector on images, image folders, or videos and
save annotated outputs (boxes + class label + confidence) plus a JSON
defect summary to ``output/yolo/<run_name>/``.

Usage:
    python predict_yolo.py path/to/image.jpg
    python predict_yolo.py path/to/folder/
    python predict_yolo.py path/to/video.mp4
    python predict_yolo.py --weights runs/detect/bd3_yolov8s/weights/best.pt \
                           path/to/video.mp4 --conf 0.30 --imgsz 640
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from defect_analyzer import (
    analyze_image, analyze_video, get_model,
    Detection, ImageReport, VideoReport,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="Image file, folder of images, or video file.")
    p.add_argument("--weights", default=None,
                   help="Path to trained .pt weights. Defaults to the most "
                        "recent runs/detect/*/weights/best.pt (or yolov8n.pt "
                        "if no fine-tuned weights exist yet).")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold (0..1).")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--grid", type=int, default=3,
                   help="Image-mode tiling grid (NxN). 1 disables tiling.")
    p.add_argument("--every-n-seconds", type=float, default=0.5,
                   help="Video-mode frame sampling interval.")
    p.add_argument("--save-video", action="store_true",
                   help="Write an annotated MP4 alongside the JSON / CSV.")
    p.add_argument("--device", default="",
                   help="'' = auto, 'cpu' = force CPU, '0' = first CUDA GPU.")
    p.add_argument("--output-dir", default="output/yolo",
                   help="Where to save annotated outputs.")
    p.add_argument("--name", default=None,
                   help="Subfolder inside --output-dir. Defaults to source basename.")
    return p.parse_args()


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VID_EXTS


def _detection_to_dict(det: Detection) -> dict[str, Any]:
    return {
        "label": det.label,
        "confidence": round(det.confidence, 4),
        "bbox": list(det.bbox),
    }


def _report_to_dict(r: ImageReport | VideoReport) -> dict[str, Any]:
    if not is_dataclass(r):
        return {}
    out = asdict(r)
    # asdict() recurses into Detection dataclasses inside tiles/frames,
    # which is fine; just round confidences for readability.
    return out


def _write_csv(report: ImageReport | VideoReport, csv_path: Path) -> int:
    rows = 0
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        if isinstance(report, VideoReport):
            w.writerow(["frame_idx", "time_sec", "label", "confidence",
                        "x1", "y1", "x2", "y2"])
            for fp in report.frames:
                for d in fp.detections:
                    w.writerow([fp.frame_idx, f"{fp.time_sec:.3f}",
                                d.label, f"{d.confidence:.4f}",
                                *d.bbox])
                    rows += 1
        else:
            w.writerow(["tile_idx", "label", "confidence",
                        "x1", "y1", "x2", "y2"])
            for tile in report.tiles:
                for d in tile.detections:
                    w.writerow([tile.tile_idx, d.label,
                                f"{d.confidence:.4f}", *d.bbox])
                    rows += 1
    return rows


def main() -> None:
    args = parse_args()
    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.name or src.stem
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Warm the model up front so a clear error fires before we start
    # iterating over a folder of images.
    get_model(args.weights)

    sources: list[Path] = []
    if src.is_dir():
        sources = sorted(p for p in src.iterdir() if _is_image(p))
        if not sources:
            raise SystemExit(f"No image files in folder: {src}")
    else:
        sources = [src]

    summary: list[dict[str, Any]] = []
    for s in sources:
        if _is_video(s):
            print(f"[video] {s}")
            r = analyze_video(
                s,
                weights_path=args.weights,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                every_n_seconds=args.every_n_seconds,
                imgsz=args.imgsz,
                annotated_out=(run_dir / f"{s.stem}_annotated.mp4")
                              if args.save_video else None,
                keyframes_dir=run_dir / "keyframes",
            )
        elif _is_image(s):
            print(f"[image] {s}")
            r = analyze_image(
                s,
                weights_path=args.weights,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                grid=args.grid,
                imgsz=args.imgsz,
                annotated_out=run_dir / f"{s.stem}_annotated.jpg",
            )
        else:
            print(f"[skip ] {s} (unsupported extension)")
            continue

        json_path = run_dir / f"{s.stem}.json"
        csv_path = run_dir / f"{s.stem}.csv"
        with json_path.open("w") as fh:
            json.dump(_report_to_dict(r), fh, indent=2, default=str)
        n_rows = _write_csv(r, csv_path)
        summary.append({
            "source": str(s),
            "json": str(json_path),
            "csv":  str(csv_path),
            "detections": n_rows,
            "defect_classes": [d["label"] for d in r.detected_defects],
        })

    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(f"Processed {len(summary)} source(s).")
    print(f"Outputs : {run_dir}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
