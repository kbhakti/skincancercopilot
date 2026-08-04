# Model Comparison Report

Candidate architectures trained and evaluated on the held-out test split.
Ranking metric: **macro-F1** (appropriate given the ~60x class imbalance found in the EDA — accuracy alone would favor models that just predict the majority class `nv`).

| Architecture | Params | Test Acc | Macro F1 | Weighted F1 | Macro AUC | Latency (ms/img) | Best val F1 | Epochs run | Train time (s) |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 **<- selected** | 11.18M | 0.6165 | 0.4038 | 0.6599 | 0.8725 | 30.49 | 0.4114 | 2 | 793.0 |
| efficientnet_b0 | 4.02M | 0.5086 | 0.3676 | 0.5607 | 0.8569 | 26.44 | 0.3831 | 2 | 702.4 |
| simplecnn | 1.21M | 0.4894 | 0.1929 | 0.5221 | 0.7239 | 77.74 | 0.2171 | 2 | 1630.0 |

**Selected model: `resnet18`** — promoted to `saved_models/best_model.pth`, which is what `src/evaluation/evaluate.py`, `src/explainability/gradcam.py`, and the Streamlit app (`src/app/app.py`) load by default.

See `outputs/model_comparison/model_comparison.png` for a visual comparison and `outputs/confusion_matrix/` / `outputs/roc_curves/` for per-architecture diagnostic plots.