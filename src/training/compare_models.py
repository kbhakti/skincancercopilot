"""
Comparative model selection for the skin lesion classifier.

Trains all candidate architectures (config.CANDIDATE_ARCHITECTURES —
by default a from-scratch CNN, a ResNet18 transfer-learning model, and
an EfficientNet-B0 transfer-learning model), evaluates each on the
held-out test set, ranks them, and promotes the winner to
saved_models/best_model.pth — the canonical checkpoint loaded by
src/evaluation/evaluate.py, src/explainability/gradcam.py, and the
Streamlit app (src/app/app.py). No other code needs to change to "use"
the chosen model: everything downstream already reads best_model.pth
and reads its own architecture out of the checkpoint.

Usage:
    python -m src.training.compare_models
    python -m src.training.compare_models --epochs 10
    python -m src.training.compare_models --architectures resnet18 efficientnet_b0

Produces:
    saved_models/<arch>_model.pth           per-architecture checkpoints
    saved_models/best_model.pth             winning checkpoint (copied)
    outputs/<arch>_training_curves.png      per-architecture loss/acc/F1 curves
    outputs/confusion_matrix/confusion_matrix_<arch>.png
    outputs/roc_curves/roc_curves_<arch>.png
    outputs/test_metrics_<arch>.json
    outputs/model_comparison/model_comparison.json   full comparison table (machine-readable)
    outputs/model_comparison/model_comparison.md      comparison table (human-readable)
    outputs/model_comparison/model_comparison.png     comparison bar charts
    outputs/confusion_matrix/confusion_matrix.png     regenerated for the winning model
    outputs/roc_curves/roc_curves.png                 regenerated for the winning model
    outputs/test_metrics.json                         regenerated for the winning model
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.training.train import train_model
from src.evaluation.evaluate import run_evaluation


def plot_comparison(results, out_path):
    archs = [r["architecture"] for r in results]
    accuracy = [r["test_accuracy"] for r in results]
    macro_f1 = [r["macro_f1"] for r in results]
    macro_auc = [r["macro_auc"] for r in results]
    params_m = [r["n_params"] / 1e6 for r in results]
    latency = [r["avg_inference_latency_ms"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(archs))
    width = 0.25

    ax = axes[0]
    ax.bar(x - width, accuracy, width, label="Test Accuracy")
    ax.bar(x, macro_f1, width, label="Macro F1")
    ax.bar(x + width, macro_auc, width, label="Macro AUC")
    ax.set_xticks(x); ax.set_xticklabels(archs)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set performance")
    ax.legend(fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, params_m, color="#8172B2")
    ax.set_xticks(x); ax.set_xticklabels(archs)
    ax.set_title("Model size (millions of parameters)")
    for b, v in zip(bars, params_m):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}M", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    bars = ax.bar(x, latency, color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(archs)
    ax.set_title("Avg CPU/GPU inference latency (ms/image)")
    for b, v in zip(bars, latency):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def write_markdown_table(results, winner_arch, out_path):
    lines = [
        "# Model Comparison Report",
        "",
        "Candidate architectures trained and evaluated on the held-out test split.",
        "Ranking metric: **macro-F1** (appropriate given the ~60x class imbalance "
        "found in the EDA — accuracy alone would favor models that just predict "
        "the majority class `nv`).",
        "",
        "| Architecture | Params | Test Acc | Macro F1 | Weighted F1 | Macro AUC | "
        "Latency (ms/img) | Best val F1 | Epochs run | Train time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        marker = " **<- selected**" if r["architecture"] == winner_arch else ""
        lines.append(
            f"| {r['architecture']}{marker} | {r['n_params']/1e6:.2f}M | "
            f"{r['test_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
            f"{r['macro_auc']:.4f} | {r['avg_inference_latency_ms']:.2f} | "
            f"{r['best_val_f1']:.4f} | {r['epochs_run']} | {r['total_train_time_s']:.1f} |"
        )
    lines += [
        "",
        f"**Selected model: `{winner_arch}`** — promoted to `saved_models/best_model.pth`, "
        "which is what `src/evaluation/evaluate.py`, `src/explainability/gradcam.py`, and "
        "the Streamlit app (`src/app/app.py`) load by default.",
        "",
        "See `outputs/model_comparison/model_comparison.png` for a visual comparison and "
        "`outputs/confusion_matrix/` / `outputs/roc_curves/` for per-architecture diagnostic plots.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare candidate architectures, then promote the best one."
    )
    parser.add_argument("--architectures", nargs="+", default=config.CANDIDATE_ARCHITECTURES,
                         choices=config.CANDIDATE_ARCHITECTURES,
                         help="Which architectures to compare (default: all candidates)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                         help="Max epochs per architecture (early stopping still applies)")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                         help="Train each candidate on a class-stratified random subset of "
                              "this fraction of the training set (e.g. 0.25), for faster "
                              "CPU-only comparison runs. Validation/test stay full-size.")
    args = parser.parse_args()

    print(f"Comparing {len(args.architectures)} architectures: {args.architectures}")
    print(f"(max {args.epochs} epochs each, patience={args.patience}, "
          f"train_fraction={args.train_fraction})\n")

    results = []
    for arch in args.architectures:
        ckpt_path = config.arch_checkpoint_path(arch)
        history_path = os.path.join(config.SAVED_MODELS_DIR, f"{arch}_training_history.json")
        label_map_path = os.path.join(config.SAVED_MODELS_DIR, f"{arch}_label_map.json")
        curves_path = os.path.join(config.OUTPUTS_DIR, f"{arch}_training_curves.png")

        train_result = train_model(
            architecture=arch, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weighted_sampler=args.weighted_sampler, patience=args.patience,
            train_fraction=args.train_fraction,
            save_path=ckpt_path, history_path=history_path,
            label_map_path=label_map_path, curves_path=curves_path,
        )

        eval_metrics = run_evaluation(checkpoint_path=ckpt_path, suffix=arch)

        results.append({
            "architecture": arch,
            "n_params": train_result["n_params"],
            "best_val_f1": train_result["best_val_f1"],
            "best_val_acc": train_result["best_val_acc"],
            "epochs_run": train_result["epochs_run"],
            "total_train_time_s": train_result["total_train_time_s"],
            "test_accuracy": eval_metrics["test_accuracy"],
            "macro_f1": eval_metrics["macro_f1"],
            "weighted_f1": eval_metrics["weighted_f1"],
            "macro_auc": eval_metrics["macro_auc"],
            "avg_inference_latency_ms": eval_metrics["avg_inference_latency_ms"],
            "checkpoint_path": ckpt_path,
        })

    # Rank by macro-F1 (robust to class imbalance), tie-break on test accuracy.
    results_sorted = sorted(results, key=lambda r: (r["macro_f1"], r["test_accuracy"]), reverse=True)
    winner = results_sorted[0]

    print(f"\n{'='*70}\nCOMPARISON SUMMARY (ranked by macro-F1)\n{'='*70}")
    header = f"{'Architecture':16s} {'Params':>10s} {'TestAcc':>8s} {'MacroF1':>8s} {'MacroAUC':>9s} {'Latency(ms)':>12s}"
    print(header)
    for r in results_sorted:
        flag = "  <-- SELECTED" if r["architecture"] == winner["architecture"] else ""
        print(f"{r['architecture']:16s} {r['n_params']/1e6:8.2f}M {r['test_accuracy']:8.4f} "
              f"{r['macro_f1']:8.4f} {r['macro_auc']:9.4f} {r['avg_inference_latency_ms']:12.2f}{flag}")

    os.makedirs(config.MODEL_COMPARISON_DIR, exist_ok=True)
    with open(config.MODEL_COMPARISON_JSON_PATH, "w") as f:
        json.dump({"results": results_sorted, "selected_architecture": winner["architecture"]},
                   f, indent=2)
    plot_comparison(results_sorted, config.MODEL_COMPARISON_PLOT_PATH)
    write_markdown_table(results_sorted, winner["architecture"], config.MODEL_COMPARISON_MD_PATH)

    print(f"\nSaved comparison report:")
    print(f"  {config.MODEL_COMPARISON_JSON_PATH}")
    print(f"  {config.MODEL_COMPARISON_MD_PATH}")
    print(f"  {config.MODEL_COMPARISON_PLOT_PATH}")

    # Promote the winner to the canonical checkpoint used by evaluate.py,
    # gradcam.py, and the Streamlit app.
    shutil.copy(winner["checkpoint_path"], config.BEST_MODEL_PATH)
    print(f"\nPromoted '{winner['architecture']}' to canonical checkpoint: {config.BEST_MODEL_PATH}")

    # Regenerate the canonical (unsuffixed) label map / history / curves / eval
    # artifacts from the winner so downstream tools that expect the default
    # filenames (saved_models/label_map.json, outputs/test_metrics.json, ...)
    # reflect the selected model.
    winner_history = os.path.join(config.SAVED_MODELS_DIR, f"{winner['architecture']}_training_history.json")
    winner_label_map = os.path.join(config.SAVED_MODELS_DIR, f"{winner['architecture']}_label_map.json")
    winner_curves = os.path.join(config.OUTPUTS_DIR, f"{winner['architecture']}_training_curves.png")
    shutil.copy(winner_history, config.TRAINING_HISTORY_PATH)
    shutil.copy(winner_label_map, config.LABEL_MAP_PATH)
    shutil.copy(winner_curves, os.path.join(config.OUTPUTS_DIR, "training_curves.png"))

    print("Regenerating canonical evaluation artifacts (confusion matrix, ROC curves, metrics)...")
    run_evaluation(checkpoint_path=config.BEST_MODEL_PATH, suffix="", verbose=False)

    print(f"\nDone. The app / evaluate / gradcam scripts will now use '{winner['architecture']}' "
          f"via {config.BEST_MODEL_PATH}.")


if __name__ == "__main__":
    main()
