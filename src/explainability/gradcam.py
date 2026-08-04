"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the PyTorch
skin lesion classifier.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).

This is the "Explainable AI" step of the pipeline: given an image and a
trained model, it highlights which regions of the lesion most influenced
the predicted class, giving both patients and clinicians a visual reason
for the prediction (complementing the RAG/LLM textual explanation).
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src import config
from src.models.cnn_model import build_model, get_target_layer


class GradCAM:
    """Computes Grad-CAM heatmaps for a given model / target conv layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        # register_full_backward_hook is preferred over the deprecated
        # register_backward_hook for correct gradient semantics.
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """Compute the Grad-CAM heatmap for `input_tensor` (1, C, H, W).

        Returns:
            cam: np.ndarray of shape (H, W) normalized to [0, 1]
            class_idx: the class index the CAM was generated for
            probs: softmax probabilities (numpy array) for all classes
        """
        self.model.eval()
        input_tensor = input_tensor.to(config.DEVICE)
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(probs.argmax(dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]        # (C, h, w)
        activations = self.activations[0]    # (C, h, w)

        weights = gradients.mean(dim=(1, 2))  # global-average-pool the gradients -> (C,)
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[c]

        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx, probs.detach().cpu().numpy()[0]


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize `cam` to the image size and blend it as a jet-colormap overlay.

    Returns an RGB uint8 numpy array ready for display/saving.
    """
    img = np.array(pil_image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE)))
    cam_resized = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(img * (1 - alpha) + heatmap * alpha)
    return overlay


def describe_activation_region(cam: np.ndarray, threshold: float = 0.6) -> str:
    """Turn the CAM into a short natural-language description of *where*
    the model focused, used to enrich the LLM prompt (src/rag/llm_explainer.py)
    with a human-readable spatial cue instead of raw pixel data.
    """
    h, w = cam.shape
    ys, xs = np.where(cam >= threshold)
    if len(xs) == 0:
        return "a diffuse region without a single sharply localized focus"

    cx, cy = xs.mean() / w, ys.mean() / h
    vert = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "central")
    horiz = "left" if cx < 0.33 else ("right" if cx > 0.66 else "central")
    coverage = len(xs) / (h * w)
    extent = "a small, localized area" if coverage < 0.15 else (
        "a broad area" if coverage > 0.45 else "a moderately sized area")
    if vert == "central" and horiz == "central":
        location = "the central part of the lesion"
    else:
        location = f"the {vert}-{horiz} part of the lesion".replace("central-central", "center")
    return f"{extent} in {location} (~{coverage*100:.0f}% of the image above threshold)"


def load_model_for_inference(checkpoint_path: str = config.BEST_MODEL_PATH):
    """Convenience loader shared by the Streamlit app and CLI demo below."""
    ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
    model = build_model(architecture=ckpt["architecture"], pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    target_layer = get_target_layer(model, ckpt["architecture"])
    return model, target_layer, ckpt["classes"]


if __name__ == "__main__":
    # Quick CLI demo: generate Grad-CAM overlays for a handful of test images
    # and save them under outputs/gradcam/.
    import glob
    import random
    from src.preprocessing.data_loader import get_transforms

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise SystemExit("No trained model found. Run `python -m src.training.train` first.")

    model, target_layer, classes = load_model_for_inference()
    cam_engine = GradCAM(model, target_layer)
    transform = get_transforms(train=False)

    random.seed(0)
    sample_files = []
    for cls in classes:
        files = glob.glob(os.path.join(config.TEST_DIR, cls, "*.jpg"))
        if files:
            sample_files.append((cls, random.choice(files)))

    for true_cls, path in sample_files:
        pil_img = Image.open(path).convert("RGB")
        tensor = transform(pil_img).unsqueeze(0)
        cam, pred_idx, probs = cam_engine.generate(tensor)
        overlay = overlay_heatmap(pil_img, cam)
        pred_cls = classes[pred_idx]
        region = describe_activation_region(cam)

        out_name = f"gradcam_true-{true_cls}_pred-{pred_cls}.png"
        out_path = os.path.join(config.GRADCAM_DIR, out_name)
        Image.fromarray(overlay).save(out_path)
        print(f"true={true_cls:6s} pred={pred_cls:6s} conf={probs[pred_idx]:.3f} "
              f"region='{region}' -> {out_path}")
