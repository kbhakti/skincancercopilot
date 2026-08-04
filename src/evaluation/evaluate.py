"""
Evaluate a trained checkpoint on the held-out test split.

CLI usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --checkpoint saved_models/best_model.pth

Library usage (used by src/training/compare_models.py to score each
candidate architecture):
    from src.evaluation.evaluate import run_evaluation
    metrics = run_evaluation(checkpoint_path, suffix="resnet18")

Produces (suffix="" -> canonical filenames used by the Streamlit app):
    outputs/test_metrics<_suffix>.json
    outputs/confusion_matrix/confusion_matrix<_suffix>.png
    outputs/roc_curves/roc_curves<_suffix>.png
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from tqdm import tqdm

from src import config
from src.models.cnn_model import build_model
from src.preprocessing.data_loader import get_dataloaders


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_probs, all_preds, all_labels = [], [], []
    for images, labels in tqdm(loader, desc="evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return (np.concatenate(all_probs), np.concatenate(all_preds),
            np.concatenate(all_labels))


@torch.no_grad()
def measure_inference_latency(model, loader, device, n_batches: int = 5) -> float:
    """Average single-image inference latency in milliseconds, measured
    over a few batches. Used as a deployment-relevance criterion in the
    model comparison report alongside accuracy/F1."""
    import time
    model.eval()
    times_ms = []
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        times_ms.append(1000 * dt / images.size(0))
    return float(np.mean(times_ms)) if times_ms else float("nan")


def plot_confusion_matrix(cm, class_names, out_path, title_suffix=""):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (row-normalized){title_suffix}")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_roc_curves(y_true, y_probs, class_names, out_path, title_suffix=""):
    n_classes = len(class_names)
    y_true_onehot = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = {}
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true_onehot[:, i], y_probs[:, i])
        aucs[cls] = float(auc)
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"One-vs-Rest ROC Curves per Class{title_suffix}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    return aucs


def run_evaluation(checkpoint_path: str = config.BEST_MODEL_PATH, suffix: str = "",
                    verbose: bool = True):
    """Load `checkpoint_path`, score it on the test split, save plots +
    metrics JSON, and return the metrics dict.

    `suffix` (e.g. "resnet18") is appended to output filenames so multiple
    architectures can be evaluated side by side without overwriting each
    other — see src/training/compare_models.py. Leave it empty ("") for
    the canonical, unsuffixed filenames the Streamlit app expects.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Run `python -m src.training.train` "
            f"or `python -m src.training.compare_models` first."
        )

    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    architecture = ckpt["architecture"]
    class_names = ckpt["classes"]
    if verbose:
        print(f"Loaded checkpoint: arch={architecture} epoch={ckpt.get('epoch')} "
              f"val_f1={ckpt.get('val_f1'):.4f}")

    model = build_model(architecture=architecture, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader, loader_classes, _ = get_dataloaders()
    assert loader_classes == class_names, "Class order mismatch between checkpoint and dataset"

    y_probs, y_pred, y_true = collect_predictions(model, test_loader, config.DEVICE)
    latency_ms = measure_inference_latency(model, test_loader, config.DEVICE)

    report = classification_report(y_true, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
    accuracy = float(np.mean(y_pred == y_true))
    cm = confusion_matrix(y_true, y_pred)

    tag = f"_{suffix}" if suffix else ""
    title_suffix = f" ({suffix})" if suffix else ""
    cm_path = os.path.join(config.CONFUSION_MATRIX_DIR, f"confusion_matrix{tag}.png")
    roc_path = os.path.join(config.ROC_DIR, f"roc_curves{tag}.png")
    metrics_path = os.path.join(config.OUTPUTS_DIR, f"test_metrics{tag}.json")

    plot_confusion_matrix(cm, class_names, cm_path, title_suffix=title_suffix)
    aucs = plot_roc_curves(y_true, y_probs, class_names, roc_path, title_suffix=title_suffix)

    metrics = {
        "checkpoint": checkpoint_path,
        "architecture": architecture,
        "n_params": ckpt.get("n_params"),
        "test_accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "macro_auc": float(np.mean(list(aucs.values()))),
        "avg_inference_latency_ms": latency_ms,
        "per_class": {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
                "support": report[cls]["support"],
                "roc_auc": aucs[cls],
            } for cls in class_names
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": cm_path,
        "roc_curve_plot": roc_path,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"\nTest accuracy: {accuracy:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"Macro AUC: {metrics['macro_auc']:.4f}")
        print(f"Avg inference latency: {latency_ms:.2f} ms/image")
        print(f"\nPer-class report:")
        for cls in class_names:
            pc = metrics["per_class"][cls]
            print(f"  {cls:6s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                  f"F1={pc['f1']:.3f} AUC={pc['roc_auc']:.3f} n={pc['support']}")
        print(f"\nSaved: {metrics_path}")
        print(f"Saved: {cm_path}")
        print(f"Saved: {roc_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test split")
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument("--suffix", type=str, default="",
                         help="Optional filename suffix, e.g. --suffix resnet18")
    args = parser.parse_args()
    run_evaluation(checkpoint_path=args.checkpoint, suffix=args.suffix)


if __name__ == "__main__":
    main()
