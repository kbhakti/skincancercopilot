"""
Training for the skin lesion CNN classifier.

Two ways to use this module:

1. CLI, single architecture:
    python -m src.training.train --arch resnet18 --epochs 20 --batch-size 32

2. Imported as a library (used by src/training/compare_models.py to train
   several candidate architectures back-to-back and compare them):
    from src.training.train import train_model
    result = train_model(architecture="resnet18", epochs=20)

`train_model()` always saves the *best* checkpoint (by validation
macro-F1) it sees during training to `save_path` (defaults to
saved_models/best_model.pth), plus a matching label map and history file.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src import config
from src.models.cnn_model import build_model, count_params
from src.preprocessing.data_loader import get_dataloaders


def macro_f1_from_confusion(y_true, y_pred, num_classes):
    """Compute macro-F1 without requiring scikit-learn (kept as an optional dep)."""
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_labels = [], []

    torch.set_grad_enabled(train)
    desc = "train" if train else "val"
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    torch.set_grad_enabled(True)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    f1 = macro_f1_from_confusion(all_labels, all_preds, config.NUM_CLASSES)
    return avg_loss, accuracy, f1


def plot_history(history, out_path, title_suffix=""):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"Loss{title_suffix}"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"Accuracy{title_suffix}"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(epochs, history["train_f1"], label="train")
    axes[2].plot(epochs, history["val_f1"], label="val")
    axes[2].set_title(f"Macro F1{title_suffix}"); axes[2].set_xlabel("epoch"); axes[2].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def train_model(
    architecture: str,
    epochs: int = config.NUM_EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    freeze_backbone: bool = False,
    weighted_sampler: bool = False,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    train_fraction: float = 1.0,
    save_path: str = None,
    history_path: str = None,
    label_map_path: str = None,
    curves_path: str = None,
    verbose: bool = True,
):
    """Train a single architecture end-to-end and checkpoint the best epoch.

    This is the reusable training routine shared by the single-model CLI
    (`python -m src.training.train`) and the multi-model comparison script
    (`src/training/compare_models.py`).

    Returns a dict summarizing the run: architecture, param count, best
    validation accuracy/F1/epoch, full per-epoch history, and the paths
    the checkpoint/history/label-map/curves were written to.
    """
    save_path = save_path or config.BEST_MODEL_PATH
    history_path = history_path or config.TRAINING_HISTORY_PATH
    label_map_path = label_map_path or config.LABEL_MAP_PATH
    curves_path = curves_path or os.path.join(config.OUTPUTS_DIR, "training_curves.png")

    if verbose:
        print(f"\n{'='*70}\nTraining architecture: {architecture}\nDevice: {config.DEVICE}\n{'='*70}")

    train_loader, val_loader, test_loader, classes, class_weights = get_dataloaders(
        batch_size=batch_size, use_weighted_sampler=weighted_sampler, train_fraction=train_fraction,
    )
    if verbose:
        print(f"Classes ({len(classes)}): {classes}")
        print(f"Class weights (inverse frequency): {class_weights.tolist()}")
        if train_fraction < 1.0:
            print(f"Training on a {train_fraction*100:.0f}% stratified subset "
                  f"({len(train_loader.dataset)} images) for faster comparison.")

    model = build_model(architecture=architecture, pretrained=True, freeze_backbone=freeze_backbone)
    n_params = count_params(model)
    if verbose:
        print(f"Total parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": []}
    best_val_f1, best_val_acc, best_epoch = -1.0, -1.0, -1
    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, train=True)
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, optimizer, config.DEVICE, train=False)
        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        dt = time.time() - t0
        if verbose:
            print(f"Epoch {epoch:02d}/{epochs} [{dt:.1f}s] "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1, best_val_acc, best_epoch = val_f1, val_acc, epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": architecture,
                "classes": classes,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "epoch": epoch,
                "n_params": n_params,
            }, save_path)
            if verbose:
                print(f"  -> saved new best checkpoint (val_f1={val_f1:.4f}) to {save_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping: no improvement for {patience} epochs.")
                break

    total_train_time_s = time.time() - t_start

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(label_map_path, "w") as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)
    plot_history(history, curves_path, title_suffix=f" ({architecture})")

    if verbose:
        print(f"\nBest val macro-F1: {best_val_f1:.4f} (epoch {best_epoch})")
        print(f"Total training time: {total_train_time_s:.1f}s")
        print(f"Model saved to: {save_path}")

    return {
        "architecture": architecture,
        "n_params": n_params,
        "best_val_f1": best_val_f1,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "total_train_time_s": total_train_time_s,
        "history": history,
        "checkpoint_path": save_path,
        "classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the skin lesion CNN classifier")
    parser.add_argument("--arch", type=str, default=config.DEFAULT_ARCHITECTURE,
                         choices=config.CANDIDATE_ARCHITECTURES)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Freeze pretrained conv layers, only train the head")
    parser.add_argument("--weighted-sampler", action="store_true",
                         help="Use a WeightedRandomSampler in addition to class-weighted loss")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--train-fraction", type=float, default=1.0,
                         help="Train on a class-stratified random subset of this fraction "
                              "of the training set (e.g. 0.25), for faster CPU-only runs")
    args = parser.parse_args()

    result = train_model(
        architecture=args.arch, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, freeze_backbone=args.freeze_backbone,
        weighted_sampler=args.weighted_sampler, patience=args.patience,
        train_fraction=args.train_fraction,
    )
    print(f"\nTip: to compare {args.arch} against the other candidate architectures "
          f"({', '.join(c for c in config.CANDIDATE_ARCHITECTURES if c != args.arch)}) "
          f"and automatically pick the best one, run:\n"
          f"  python -m src.training.compare_models\n")
    print("Run `python -m src.evaluation.evaluate` next to score it on the held-out test set.")


if __name__ == "__main__":
    main()
