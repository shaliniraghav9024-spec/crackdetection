"""
ResNet-18 training driver for the BD3 (Building Defects Detection) dataset.

This is a runnable, parameterized version of
    BD3-Dataset/code/model-train/resnet18/nn-ResNet-18.ipynb

Defaults are tuned to run end-to-end on the 37 sample images shipped in the
repo, but every knob is overridable from the CLI so you can point it at the
full 3,965-image BD3 dataset later.

Usage (with the venv activated):
    python train_resnet18.py
    python train_resnet18.py --data-dir BD3-Dataset/dataset/train-curat-dataset \
                             --epochs 200 --batch-size 16
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="BD3-Dataset/dataset/train-curat-dataset",
                   help="Root containing train/, val/, test/ subfolders (ImageFolder layout).")
    p.add_argument("--output-dir", default="output/resnet18",
                   help="Where to save weights, plots, and reports.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--feature-extract", action="store_true", default=True,
                   help="Freeze backbone, train only the classifier head.")
    p.add_argument("--full-finetune", dest="feature_extract", action="store_false",
                   help="Fine-tune all layers instead of feature-extracting.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transforms_map = {"train": train_tf, "val": eval_tf, "test": eval_tf}

    image_datasets = {
        split: datasets.ImageFolder(data_dir / split, transforms_map[split])
        for split in ("train", "val", "test")
    }
    dataloaders = {
        split: torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, ds in image_datasets.items()
    }
    sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names = image_datasets["train"].classes
    return dataloaders, sizes, class_names


def build_model(num_classes: int, feature_extract: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, dataloaders, sizes, criterion, optimizer, device, epochs: int):
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    started = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)
        for phase in ("train", "val"):
            if sizes[phase] == 0:
                continue
            model.train(phase == "train")
            running_loss = 0.0
            running_correct = 0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += int(torch.sum(preds == labels.data))

            epoch_loss = running_loss / sizes[phase]
            epoch_acc = running_correct / sizes[phase]
            print(f"  {phase:5s} loss={epoch_loss:.4f} acc={epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

    elapsed = time.time() - started
    print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f"Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history


def evaluate(model, dataloader, device, class_names):
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    correct = sum(int(p == l) for p, l in zip(all_preds, all_labels))
    acc = correct / max(1, len(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    return acc, cm, report, all_labels, all_preds


def save_history_plot(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_acc"], label="train")
    if history["val_acc"]:
        axes[0].plot(epochs, history["val_acc"], label="val")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_loss"], label="train")
    if history["val_loss"]:
        axes[1].plot(epochs, history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise SystemExit(
            f"Dataset not found: {data_dir}\n"
            f"Run `python prepare_dataset.py` first to build the split."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data:   {data_dir}")
    print(f"Output: {output_dir}\n")

    dataloaders, sizes, class_names = build_dataloaders(
        data_dir, args.batch_size, args.num_workers
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Sizes: {sizes}\n")

    model = build_model(len(class_names), args.feature_extract).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()

    model, history = train_model(
        model, dataloaders, sizes, criterion, optimizer, device, args.epochs
    )

    weights_path = output_dir / "resnet18_best.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"\nSaved weights -> {weights_path}")

    save_history_plot(history, output_dir / "training_history.png")
    print(f"Saved plot    -> {output_dir / 'training_history.png'}")

    if sizes["test"] > 0:
        test_acc, cm, report, y_true, y_pred = evaluate(
            model, dataloaders["test"], device, class_names
        )
        print(f"\nTest accuracy: {test_acc:.4f}")
        print("Classification report:\n" + report)
        print("Confusion matrix:\n", cm)

        save_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
        (output_dir / "classification_report.txt").write_text(report)
        pd.DataFrame({
            "actual": [class_names[i] for i in y_true],
            "predicted": [class_names[i] for i in y_pred],
        }).to_csv(output_dir / "test_predictions.csv", index=False)
        print(f"Saved CM       -> {output_dir / 'confusion_matrix.png'}")
        print(f"Saved report   -> {output_dir / 'classification_report.txt'}")
        print(f"Saved preds    -> {output_dir / 'test_predictions.csv'}")
    else:
        print("No test set available; skipping evaluation.")


if __name__ == "__main__":
    main()
