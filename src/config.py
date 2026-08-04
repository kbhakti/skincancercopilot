"""
Central configuration for the AI Dermatology Copilot project.

All paths, class definitions, and hyperparameters used across the
preprocessing / training / evaluation / explainability / RAG / app
modules live here so there is a single source of truth.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")

SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "best_model.pth")
TRAINING_HISTORY_PATH = os.path.join(SAVED_MODELS_DIR, "training_history.json")
LABEL_MAP_PATH = os.path.join(SAVED_MODELS_DIR, "label_map.json")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
GRADCAM_DIR = os.path.join(OUTPUTS_DIR, "gradcam")
CONFUSION_MATRIX_DIR = os.path.join(OUTPUTS_DIR, "confusion_matrix")
ROC_DIR = os.path.join(OUTPUTS_DIR, "roc_curves")
METRICS_JSON_PATH = os.path.join(OUTPUTS_DIR, "test_metrics.json")

MODEL_COMPARISON_DIR = os.path.join(OUTPUTS_DIR, "model_comparison")
MODEL_COMPARISON_JSON_PATH = os.path.join(MODEL_COMPARISON_DIR, "model_comparison.json")
MODEL_COMPARISON_PLOT_PATH = os.path.join(MODEL_COMPARISON_DIR, "model_comparison.png")
MODEL_COMPARISON_MD_PATH = os.path.join(MODEL_COMPARISON_DIR, "model_comparison.md")

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
RAG_DIR = os.path.join(SRC_DIR, "rag")
KNOWLEDGE_DIR = os.path.join(RAG_DIR, "knowledge")
INDEX_DIR = os.path.join(RAG_DIR, "index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "knowledge.faiss")
CHUNK_STORE_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

for d in [SAVED_MODELS_DIR, OUTPUTS_DIR, GRADCAM_DIR, CONFUSION_MATRIX_DIR,
          ROC_DIR, INDEX_DIR, MODEL_COMPARISON_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes (HAM10000 diagnostic categories)
# ---------------------------------------------------------------------------
# NOTE: order matters only insofar as it must match torchvision.datasets
# ImageFolder's alphabetical class ordering, which is what these already are.
CLASS_NAMES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma (Bowen's disease)",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
CLASS_CODES = sorted(CLASS_NAMES.keys())  # akiec, bcc, bkl, df, mel, nv, vasc
NUM_CLASSES = len(CLASS_CODES)

# Rough clinical risk tier used only to color-code / prioritize the UI and
# to steer the LLM's "recommended action" — NOT a substitute for medical
# judgement. akiec/bcc/mel are malignant or pre-malignant; the rest are
# generally benign but can still warrant a check if changing.
RISK_TIER = {
    "akiec": "pre-malignant",
    "bcc": "malignant",
    "bkl": "benign",
    "df": "benign",
    "mel": "malignant",
    "nv": "benign",
    "vasc": "benign",
}

# ---------------------------------------------------------------------------
# Image / training hyperparameters
# ---------------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 6
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 1e-5

# Leave headroom for the NUM_WORKERS DataLoader worker processes: on
# CPU-only machines, letting torch's intra-op thread pool default to
# *all* logical cores while worker processes are also active causes heavy
# oversubscription (measured ~5x slowdown per training batch on this
# machine). Capping it here fixes every entry point (train.py,
# compare_models.py, evaluate.py, the Streamlit app) at once.
_cpu_count = os.cpu_count() or 4
torch.set_num_threads(max(1, _cpu_count - NUM_WORKERS))

# The EDA (notebooks/EDA.ipynb) found this dataset's images, although
# stored as 3-channel JPEGs, are effectively grayscale (R == G == B for
# every sampled pixel). We therefore normalize with the dataset's own
# measured per-channel mean/std (computed in the EDA) rather than blindly
# assuming ImageNet statistics, even though the three channels are equal.
DATASET_MEAN = [0.5271, 0.5271, 0.5271]
DATASET_STD = [0.2208, 0.2208, 0.2208]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
DEFAULT_ARCHITECTURE = "resnet18"  # one of: simplecnn, resnet18, efficientnet_b0
GRADCAM_TARGET_LAYER = {
    "resnet18": "layer4",
    "efficientnet_b0": "features",
    "simplecnn": "conv_block4",
}

# Candidate architectures evaluated by src/training/compare_models.py:
#   - simplecnn:       from-scratch CNN baseline (no pretraining, cheapest)
#   - resnet18:        classic transfer-learning backbone, strong accuracy/speed balance
#   - efficientnet_b0: modern, parameter-efficient transfer-learning backbone
# The comparison script trains all three, evaluates them on the held-out
# test set, and promotes the best one (by macro-F1) to saved_models/best_model.pth,
# which is what evaluate.py / gradcam.py / the Streamlit app load by default.
CANDIDATE_ARCHITECTURES = ["simplecnn", "resnet18", "efficientnet_b0"]


def arch_checkpoint_path(architecture: str) -> str:
    """Per-architecture checkpoint path used during model comparison,
    e.g. saved_models/resnet18_model.pth."""
    return os.path.join(SAVED_MODELS_DIR, f"{architecture}_model.pth")

# ---------------------------------------------------------------------------
# RAG / LLM
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 4
CHUNK_MAX_CHARS = 700
CHUNK_OVERLAP_CHARS = 100

# Local, open-source LLM served via Ollama (https://ollama.com) — no API
# key / internet call needed once the model is pulled.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_S = 60

DISCLAIMER = (
    "This tool is an educational decision-support prototype. It is NOT a "
    "medical device and does not provide a diagnosis. Always consult a "
    "qualified dermatologist or physician for evaluation of any skin lesion."
)
