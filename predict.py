"""
Run the trained ResNet-18 model on one or more images and print the predicted
defect class.

Usage:
    python predict.py path/to/image.jpg [more images...]
    python predict.py --weights output/resnet18/resnet18_best.pth path/to/image.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    p.add_argument("images", nargs="+", help="Image file(s) to classify.")
    p.add_argument("--weights", default="output/resnet18/resnet18_best.pth",
                   help="Path to the trained .pth weights file.")
    p.add_argument("--classes", default="classes.txt",
                   help="Path to classes.txt (one class name per line).")
    p.add_argument("--topk", type=int, default=3, help="Show top-K predictions.")
    return p.parse_args()


def load_model(weights_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        raise SystemExit(
            f"Weights not found: {weights_path}\n"
            f"Train the model first: `python train_resnet18.py`"
        )

    class_names = load_class_names(Path(args.classes).resolve())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(weights_path, len(class_names), device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"Model:   {weights_path}")
    print(f"Classes: {class_names}")
    print(f"Device:  {device}\n")

    for raw_path in args.images:
        path = Path(raw_path)
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] {path}: cannot open ({e})")
            continue

        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        topk = torch.topk(probs, k=min(args.topk, len(class_names)))
        print(f"{path}")
        for rank, (p, idx) in enumerate(zip(topk.values.tolist(),
                                            topk.indices.tolist()), start=1):
            marker = "  ->" if rank == 1 else "    "
            print(f"{marker} {class_names[idx]:<12} {p * 100:5.1f}%")
        print()


if __name__ == "__main__":
    main()
