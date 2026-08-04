"""
AI Dermatology Copilot — Streamlit web app.

Pipeline implemented here:
  User uploads image
        -> CNN model predicts lesion class (src/models, saved_models/best_model.pth)
        -> Grad-CAM explains the prediction (src/explainability/gradcam.py)
        -> Retrieve trusted medical knowledge (src/rag/retriever.py)
        -> Local LLM generates plain-language + clinical explanations (src/rag/llm_explainer.py)
        -> Results displayed below

Run:
    streamlit run src/app/app.py
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src import config
from src.explainability.gradcam import GradCAM, overlay_heatmap, describe_activation_region, load_model_for_inference
from src.preprocessing.data_loader import get_transforms
from src.rag.retriever import KnowledgeRetriever
from src.rag.llm_explainer import LLMExplainer

st.set_page_config(page_title="AI Dermatology Copilot", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading CNN classifier...")
def _load_model():
    if not os.path.exists(config.BEST_MODEL_PATH):
        return None
    model, target_layer, classes = load_model_for_inference()
    # Pull a bit of checkpoint metadata (which of the 3 candidate
    # architectures was selected by src/training/compare_models.py, and
    # how it scored) purely for display in the sidebar.
    ckpt_meta = torch.load(config.BEST_MODEL_PATH, map_location="cpu")
    meta = {
        "architecture": ckpt_meta.get("architecture", "unknown"),
        "val_f1": ckpt_meta.get("val_f1"),
        "val_acc": ckpt_meta.get("val_acc"),
        "n_params": ckpt_meta.get("n_params"),
    }
    return model, target_layer, classes, meta


@st.cache_resource(show_spinner="Loading knowledge base + retriever...")
def _load_retriever():
    try:
        return KnowledgeRetriever()
    except FileNotFoundError:
        return None


def run_pipeline(pil_image: Image.Image, model, target_layer, classes, retriever):
    transform = get_transforms(train=False)
    tensor = transform(pil_image).unsqueeze(0)

    cam_engine = GradCAM(model, target_layer)
    cam, pred_idx, probs = cam_engine.generate(tensor)
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    overlay = overlay_heatmap(pil_image, cam)
    region_description = describe_activation_region(cam)

    explanation = None
    if retriever is not None:
        explainer = LLMExplainer(retriever)
        explanation = explainer.explain(pred_class, confidence, region_description)

    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": probs,
        "overlay": overlay,
        "region_description": region_description,
        "explanation": explanation,
    }


def main():
    st.title("🩺 AI Dermatology Copilot")
    st.caption(
        "Computer Vision + Explainable AI + Retrieval-Augmented Generation, combined into "
        "an end-to-end decision-support prototype for skin lesion images."
    )
    st.warning(f"⚠️ {config.DISCLAIMER}", icon="⚠️")

    with st.sidebar:
        st.header("About")
        st.markdown(
            "**Pipeline:**\n\n"
            "1. CNN classifies the lesion (7 HAM10000 classes)\n"
            "2. Grad-CAM highlights the influential region\n"
            "3. Local RAG retrieves trusted dermatology knowledge\n"
            "4. A local LLM (via Ollama) writes patient + clinician explanations\n"
        )
        st.divider()
        st.subheader("Local LLM status")
        st.markdown(
            f"- Ollama URL: `{config.OLLAMA_BASE_URL}`\n"
            f"- Model: `{config.OLLAMA_MODEL}`\n\n"
            "If Ollama isn't running, explanations fall back to a template "
            "built directly from the retrieved knowledge base."
        )
        st.divider()
        st.subheader("Class legend")
        for code, name in config.CLASS_NAMES.items():
            st.markdown(f"- **{code}** — {name} _({config.RISK_TIER[code]})_")

    loaded = _load_model()
    retriever = _load_retriever()

    if loaded is None:
        st.error(
            "No trained model found at `saved_models/best_model.pth`. "
            "Run `python -m src.training.compare_models` (recommended) or "
            "`python -m src.training.train` first, then reload this page."
        )
        return
    model, target_layer, classes, model_meta = loaded

    with st.sidebar:
        st.divider()
        st.subheader("Active model")
        params_str = f"{model_meta['n_params']/1e6:.2f}M" if model_meta["n_params"] else "n/a"
        val_f1_str = f"{model_meta['val_f1']:.3f}" if model_meta["val_f1"] is not None else "n/a"
        st.markdown(
            f"- Architecture: **{model_meta['architecture']}**\n"
            f"- Parameters: {params_str}\n"
            f"- Validation macro-F1: {val_f1_str}\n\n"
            "Selected automatically by `src/training/compare_models.py` out of "
            f"{len(config.CANDIDATE_ARCHITECTURES)} candidate architectures "
            f"({', '.join(config.CANDIDATE_ARCHITECTURES)}) — see "
            "`outputs/model_comparison/model_comparison.md` for the full comparison."
        )

    if retriever is None:
        st.info(
            "RAG index not found. Run `python -m src.rag.build_index` to enable retrieval "
            "and explanation generation. Predictions + Grad-CAM will still work without it.",
            icon="ℹ️",
        )

    uploaded_file = st.file_uploader(
        "Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.markdown("Upload an image above to run the full pipeline.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_column_width=True)

    with st.spinner("Running CNN inference, Grad-CAM, retrieval, and LLM generation..."):
        result = run_pipeline(pil_image, model, target_layer, classes, retriever)

    with col2:
        st.subheader("Grad-CAM Explanation")
        st.image(result["overlay"], use_column_width=True,
                  caption=f"Model focused on {result['region_description']}")

    st.divider()
    pred_class = result["pred_class"]
    confidence = result["confidence"]
    risk_tier = config.RISK_TIER[pred_class]
    risk_color = {"malignant": "red", "pre-malignant": "orange", "benign": "green"}[risk_tier]

    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted class", f"{pred_class.upper()}")
    m2.metric("Confidence", f"{confidence*100:.1f}%")
    m3.markdown(f"**Risk tier:** :{risk_color}[{risk_tier.upper()}]")
    st.markdown(f"**{config.CLASS_NAMES[pred_class]}**")

    prob_df = pd.DataFrame({
        "class": [f"{c} — {config.CLASS_NAMES[c]}" for c in classes],
        "probability": result["probs"],
    }).sort_values("probability", ascending=False)
    st.bar_chart(prob_df.set_index("class"))

    st.divider()
    st.subheader("AI-Generated Explanation")
    explanation = result["explanation"]
    if explanation is None:
        st.info("Build the RAG index (`python -m src.rag.build_index`) to see patient/clinician explanations.")
    else:
        badge = "🤖 local LLM" if explanation.used_llm else "📋 template (LLM unavailable)"
        st.caption(f"Generated by: {badge}")

        tab1, tab2, tab3 = st.tabs(["Patient Explanation", "Clinician Summary", "Next Steps"])
        with tab1:
            st.write(explanation.patient_explanation)
        with tab2:
            st.write(explanation.clinician_summary)
        with tab3:
            st.write(explanation.recommended_action)

        st.caption(f"⚠️ {explanation.disclaimer}")
        if explanation.sources:
            with st.expander("Retrieved knowledge sources"):
                for s in explanation.sources:
                    st.markdown(f"- {s}")


if __name__ == "__main__":
    main()
