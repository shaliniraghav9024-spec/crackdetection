"""
Run the trained ResNet-18 classifier on a video, frame by frame.

For each processed frame the script:
  - predicts the most likely defect class out of the 7 BD3 classes
  - overlays the class name + confidence on the frame
  - writes the annotated frame to an output video
  - logs the prediction to a CSV file

NOTE: This is *classification*, not object detection.  The model labels the
whole frame; it does not draw bounding boxes.  For real bounding-box
detection you need a YOLO-style detector trained with box annotations.

Usage:
    python predict_video.py path/to/input.mp4
    python predict_video.py input.mp4 --output annotated.mp4 --every 5 --topk 3
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def load_class_names(path: Path) -> list[str]:
    """Read class names from a text file (one class per line)."""
    if not path.exists():
        raise SystemExit(
            f"Classes file not found: {path}\n"
            f"Make sure classes.txt exists at the project root."
        )
    names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not names:
        raise SystemExit(f"Classes file is empty: {path}")
    return sorted(names)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", help="Path to the input video file.")
    p.add_argument("--weights", default="output/resnet18/resnet18_best.pth",
                   help="Path to the trained .pth weights file.")
    p.add_argument("--classes", default="classes.txt",
                   help="Path to classes.txt (one class name per line).")
    p.add_argument("--output", default=None,
                   help="Output annotated video path. "
                        "Default: output/<video_stem>_annotated.mp4")
    p.add_argument("--csv", default=None,
                   help="Output predictions CSV path. "
                        "Default: output/<video_stem>_predictions.csv")
    p.add_argument("--every", type=int, default=1,
                   help="Run inference on every Nth frame (default 1 = every frame). "
                        "Use 5 or 10 to speed up long videos on CPU.")
    p.add_argument("--topk", type=int, default=3,
                   help="Show top-K predictions on screen (default 3).")
    p.add_argument("--conf-threshold", type=float, default=0.0,
                   help="Hide the overlay when top prediction is below this "
                        "probability (0.0 - 1.0). Default 0 = always show.")
    return p.parse_args()


def load_model(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict_frame(model: nn.Module, frame_bgr, tf, device: torch.device,
                  k: int, num_classes: int):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    top = torch.topk(probs, k=min(k, num_classes))
    return list(zip(top.indices.tolist(), top.values.tolist()))


def annotate(frame, predictions, conf_threshold: float, frame_idx: int,
             ts_sec: float, class_names: list[str]):
    h, w = frame.shape[:2]
    pad = 10
    line_h = 28
    box_w = max(360, int(w * 0.45))
    box_h = pad * 2 + line_h * (len(predictions) + 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    top_idx, top_p = predictions[0]
    color_top = (0, 255, 0) if top_p >= max(conf_threshold, 0.5) else (0, 215, 255)

    header = f"frame {frame_idx}  t={ts_sec:.2f}s"
    cv2.putText(frame, header, (pad + 10, pad + line_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

    for i, (idx, p) in enumerate(predictions):
        label = f"{class_names[idx]:<12}  {p * 100:5.1f}%"
        y = pad + line_h * (i + 2) - 6
        color = color_top if i == 0 else (255, 255, 255)
        thickness = 2 if i == 0 else 1
        cv2.putText(frame, label, (pad + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise SystemExit(
            f"Weights not found: {weights_path}\n"
            f"Train the model first: `python train_resnet18.py`"
        )

    out_dir = Path("output").resolve()
    out_dir.mkdir(exist_ok=True)
    output_video = Path(args.output).resolve() if args.output else \
        out_dir / f"{video_path.stem}_annotated.mp4"
    output_csv = Path(args.csv).resolve() if args.csv else \
        out_dir / f"{video_path.stem}_predictions.csv"

    class_names = load_class_names(Path(args.classes).resolve())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device:      {device}")
    print(f"Model:       {weights_path}")
    print(f"Classes:     {class_names}")
    print(f"Input video: {video_path}")
    print(f"Output mp4:  {output_video}")
    print(f"Output csv:  {output_csv}")
    print(f"Inferring every {args.every} frame(s); top-{args.topk} predictions\n")

    model = load_model(weights_path, len(class_names), device)
    tf = make_transform()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open the video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"OpenCV could not open the output writer: {output_video}")

    csv_file = open(output_csv, "w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["frame", "time_sec", "predicted_class", "confidence"]
                   + [f"top{i + 1}" for i in range(args.topk)]
                   + [f"top{i + 1}_prob" for i in range(args.topk)])

    last_predictions: list[tuple[int, float]] = []
    frame_idx = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts_sec = frame_idx / fps if fps else 0.0

            if frame_idx % max(1, args.every) == 0:
                last_predictions = predict_frame(
                    model, frame, tf, device, args.topk, len(class_names)
                )
                top_idx, top_p = last_predictions[0]
                row = [frame_idx, f"{ts_sec:.3f}",
                       class_names[top_idx], f"{top_p:.4f}"]
                row += [class_names[i] for i, _ in last_predictions]
                row += [f"{p:.4f}" for _, p in last_predictions]
                csv_w.writerow(row)

            if last_predictions:
                annotate(frame, last_predictions, args.conf_threshold,
                         frame_idx, ts_sec, class_names)
            writer.write(frame)

            frame_idx += 1
            if total > 0 and frame_idx % max(1, total // 20 or 1) == 0:
                pct = 100.0 * frame_idx / total
                print(f"  ... {frame_idx}/{total} frames ({pct:.0f}%)")
    finally:
        cap.release()
        writer.release()
        csv_file.close()

    elapsed = time.time() - started
    print(f"\nDone. Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-9):.1f} fps).")
    print(f"Annotated video: {output_video}")
    print(f"Predictions CSV: {output_csv}")


if __name__ == "__main__":
    main()
