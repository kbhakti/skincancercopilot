# skincancercopilot(AI Dermatology Copilot)
An end-to-end AI healthcare decision-support prototype for skin lesion
images, combining Computer Vision, Explainable AI (Grad-CAM),
Retrieval-Augmented Generation (RAG), and a local LLM.

```
User uploads skin image
        |
        v
 CNN model predicts lesion class          (src/models, src/training)
        |
        v
 Grad-CAM explains the prediction         (src/explainability)
        |
        v
 Retrieve trusted medical knowledge       (src/rag)
        |
        v
 Local LLM generates explanations         (src/rag/llm_explainer.py)
        |
        v
 Results displayed in a Streamlit app     (src/app/app.py)
```

## Project layout

```
dataset/                  train/validation/test images, 7 classes (already provided)
notebooks/EDA.ipynb       full exploratory data analysis (already run, results embedded)
src/
  config.py                central paths, class list, hyperparameters
  preprocessing/
    data_loader.py          PyTorch Dataset/DataLoader, transforms, class weights
  models/
    cnn_model.py             SimpleCNN + ResNet18/EfficientNet-B0 transfer learning
  training/
    train.py                 reusable training loop for a single architecture
    compare_models.py         trains & compares all candidate architectures, picks the best
  evaluation/
    evaluate.py               test-set metrics, confusion matrix, ROC curves
  explainability/
    gradcam.py                 Grad-CAM heatmaps + overlay + region description
  rag/
    knowledge/*.md               curated dermatology reference docs (7 classes)
    build_index.py               builds the local FAISS vector index
    retriever.py                  semantic search over the knowledge base
    llm_explainer.py              prompts a local Ollama LLM (+ template fallback)
  app/
    app.py                         Streamlit web app tying everything together
saved_models/              trained checkpoints + label map (created by training)
outputs/                   training curves, confusion matrix, ROC curves, Grad-CAMs,
                            model_comparison/ (comparison report across architectures)
requirements.txt
```

The dataset (7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`)
was already present in `dataset/train|validation|test/<class>/*.jpg`
when this project was built. `notebooks/EDA.ipynb` analyzes it in
detail — notably it found the images are 224x224 and **effectively
grayscale** (identical R/G/B channels), and heavily class-imbalanced
(~60x between the largest and smallest class). Both findings directly
shaped the normalization constants in `src/config.py` and the
class-weighted loss in `src/training/train.py`.

## 1. Prerequisites

- Python 3.10+ (a virtual environment is strongly recommended)
- ~4 GB free disk space for model weights + embedding model + Ollama model
- [Ollama](https://ollama.com) installed, for the local LLM step
  (optional — the app still works without it, using a template-based
  explanation generator as a fallback)
- A GPU is optional; training/inference both run on CPU, just slower

## 2. Set up the environment

```bash
cd skincancercopilot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the EDA (optional — already executed)

`notebooks/EDA.ipynb` already contains the executed outputs. To
re-run it yourself:

```bash
jupyter notebook notebooks/EDA.ipynb
```

## 4. Train and compare candidate models, then auto-select the best one

This project evaluates **three appropriate architectures** for skin
lesion classification and picks the best one automatically instead of
committing to a single model up front:

| Architecture | Type | Why it's a candidate |
|---|---|---|
| `simplecnn` | From-scratch CNN (4 conv blocks) | Cheap baseline, no pretrained weights, shows what a plain CNN can do on this data |
| `resnet18` | Transfer learning (ImageNet-pretrained) | Classic, well-understood backbone; strong accuracy/speed balance on small-to-medium medical image sets |
| `efficientnet_b0` | Transfer learning (ImageNet-pretrained) | Modern, parameter-efficient backbone; often matches/beats ResNet18 accuracy with fewer parameters |

Run the comparison (recommended path):

```bash
python -m src.training.compare_models --epochs 20
```

**A note on CPU-only training time.** This dataset has ~14k training
images at 224x224; on a CPU-only machine (no CUDA), one epoch takes
roughly 15-30 minutes per architecture depending on how compute-heavy
it is (measured on a 12-core machine: SimpleCNN ~28 min/epoch,
ResNet18 ~15 min/epoch, EfficientNet-B0 ~25 min/epoch after the fix
below — SimpleCNN is slower than ResNet18 despite having far fewer
parameters because it doesn't aggressively downsample in its first
conv block the way ResNet18's stem does). Running the full recommended
`--epochs 20` budget across all three architectures can therefore take
several hours on CPU. Two things to know:

- `src/config.py` caps `torch.set_num_threads()` to
  `cpu_count - NUM_WORKERS` (both defined in `src/config.py`). Without
  this, PyTorch's intra-op thread pool defaults to *all* logical cores
  while the `NUM_WORKERS` DataLoader worker processes are also
  competing for CPU, causing ~5x oversubscription slowdown (measured:
  10.4s/batch → 2.1s/batch on ResNet18 after capping threads). If you
  change `NUM_WORKERS`, this adjusts automatically.
- For a first pass, use `--train-fraction` to train on a class-stratified
  random subset of the training set (validation/test stay full-size, so
  the reported comparison metrics are still measured on the complete
  held-out test set):
  ```bash
  python -m src.training.compare_models --epochs 2 --patience 2 --train-fraction 0.2
  ```
  This trains each of the 3 architectures on ~20% of the training images
  (stratified per class, so the rare classes like `df` aren't wiped out)
  for 2 epochs — roughly 45-70 minutes total across all three
  architectures on a 12-core CPU, versus several hours at full data/epochs.
  Once you've confirmed the pipeline works end-to-end, re-run with
  `--train-fraction 1.0` (the default) and a bigger `--epochs` for a
  stronger final model — it simply overwrites `saved_models/best_model.pth`.
  If you have a CUDA GPU, none of this applies: `config.DEVICE` picks it
  up automatically and training is dramatically faster.

This trains all three (each with class-weighted loss + early stopping,
best checkpoint kept by validation macro-F1), evaluates every one of
them on the held-out **test** split, and ranks them by **macro-F1**
(the right metric here given the ~60x class imbalance found in the
EDA — plain accuracy would favor a model that just predicts the
majority class `nv`). It then:

- copies the winning checkpoint to `saved_models/best_model.pth` — the
  single canonical model that `evaluate.py`, `gradcam.py`, and the
  Streamlit app all load by default (no further code changes needed to
  "use" the chosen model);
- writes a comparison report to `outputs/model_comparison/`:
  `model_comparison.json` (full metrics), `model_comparison.md`
  (readable table), `model_comparison.png` (bar charts of accuracy /
  macro-F1 / macro-AUC / parameter count / inference latency per
  architecture);
- writes per-architecture confusion matrices and ROC curves to
  `outputs/confusion_matrix/confusion_matrix_<arch>.png` and
  `outputs/roc_curves/roc_curves_<arch>.png`, plus a canonical
  (unsuffixed) copy for the selected model.

To compare only a subset, or change the epoch budget per model:

```bash
python -m src.training.compare_models --architectures resnet18 efficientnet_b0 --epochs 10
```

**Alternative:** train a single architecture without comparison:

```bash
python -m src.training.train --arch resnet18 --epochs 20 --batch-size 32
```

This saves directly to `saved_models/best_model.pth` and skips the
comparison report. It also accepts `--train-fraction` for the same
faster-CPU-run tradeoff described above.

## 5. Evaluate the selected model on the held-out test set

`compare_models.py` already runs this for the winning model, but you
can re-run it anytime:

```bash
python -m src.evaluation.evaluate
```

Produces `outputs/test_metrics.json`,
`outputs/confusion_matrix/confusion_matrix.png`, and
`outputs/roc_curves/roc_curves.png` for whatever checkpoint is
currently at `saved_models/best_model.pth`.

## 6. (Optional) Generate standalone Grad-CAM samples

```bash
python -m src.explainability.gradcam
```

Saves one Grad-CAM overlay per class to `outputs/gradcam/`. (The
Streamlit app also generates these live for any uploaded image.)

## 7. Build the local RAG knowledge index

```bash
python -m src.rag.build_index
```

Reads `src/rag/knowledge/*.md`, chunks and embeds them with a small
local `sentence-transformers` model, and writes a FAISS index to
`src/rag/index/`. This is fully offline — no API key needed. The first
run downloads the ~90 MB embedding model from Hugging Face.

## 8. Set up the local LLM (Ollama)

```bash
# Install Ollama: https://ollama.com/download
ollama serve                 # starts the local server on :11434
ollama pull llama3.2         # or any other model you prefer
```

To use a different model or host, set environment variables before
running the app:

```bash
export OLLAMA_MODEL=llama3.2      # Windows (PowerShell): $env:OLLAMA_MODEL="llama3.2"
export OLLAMA_BASE_URL=http://localhost:11434
```

This project was verified end-to-end against a locally running Ollama
server with `llama3.1` pulled (`OLLAMA_MODEL=llama3.1`) — any
reasonably capable instruction-following model works, since the prompt
(`src/rag/llm_explainer.py::_build_prompt`) is generic. Larger models
give more coherent, better-structured explanations; smaller/quantized
models respond faster but may occasionally drift from the requested
4-section format (the parser degrades gracefully — see the note below).

If Ollama isn't running, times out, or isn't reachable, the app
automatically falls back to a deterministic, template-based explanation
built directly from the retrieved knowledge chunks, so the full
pipeline still works without it. This was verified directly: running
the CNN comparison training in the background (CPU-bound, saturating
most cores) caused a real Ollama call to time out, and the app cleanly
fell back to the template path instead of erroring.

## 9. Run the web app

```bash
streamlit run src/app/app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`),
upload a skin lesion image, and the app will run the full pipeline:
CNN prediction -> Grad-CAM -> RAG retrieval -> LLM explanation.

## Quick start (all steps in order)

```bash
cd skincancercopilot
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Trains & compares simplecnn/resnet18/efficientnet_b0, picks the best, saves it.
# Full run (best accuracy, several hours on CPU-only machines):
python -m src.training.compare_models --epochs 20
# Faster first pass (~45-70 min on a 12-core CPU-only machine):
python -m src.training.compare_models --epochs 2 --patience 2 --train-fraction 0.2

python -m src.rag.build_index

ollama serve &
ollama pull llama3.2      # or use a model you already have pulled, e.g. llama3.1,
                           # via: set OLLAMA_MODEL=llama3.1 (Windows) / export OLLAMA_MODEL=llama3.1

streamlit run src/app/app.py
```

## Important disclaimer

This is an educational decision-support prototype, not a medical
device. It does not provide a diagnosis. Always consult a qualified
dermatologist or physician for evaluation of any skin lesion. See
`src/config.py::DISCLAIMER`, which is also surfaced in the app UI.
