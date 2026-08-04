"""
LLM explanation generator.

Takes the CNN prediction + Grad-CAM description + retrieved knowledge
chunks and produces a structured, plain-language explanation with four
parts: Patient Explanation, Clinician Summary, Recommended Next Steps,
and Disclaimer.

Primary path: a local, open-source LLM served by Ollama
(https://ollama.com) — fully offline once the model is pulled, no API
key required.

Fallback path: if Ollama isn't running/reachable, a deterministic
template-based generator builds the same four sections directly from
the retrieved knowledge-base chunks, so the app still works end-to-end
without any LLM installed (useful for grading/demo environments).
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import re
from dataclasses import dataclass, field

import requests

from src import config


@dataclass
class Explanation:
    patient_explanation: str
    clinician_summary: str
    recommended_action: str
    disclaimer: str = config.DISCLAIMER
    sources: list = field(default_factory=list)
    used_llm: bool = False


def build_query(predicted_class: str, confidence: float, region_description: str) -> str:
    """Compose a rich retrieval query, as recommended for better RAG results:
    disease + confidence + Grad-CAM spatial cue + the generation task."""
    class_name = config.CLASS_NAMES[predicted_class]
    return (
        f"Disease: {class_name} ({predicted_class}). "
        f"Model confidence: {confidence*100:.1f}%. "
        f"Grad-CAM highlighted {region_description}. "
        f"Question: definition, symptoms, risk factors, diagnosis, treatment, "
        f"and recommended next steps for a patient and a clinician."
    )


def _format_context(chunks) -> str:
    lines = []
    for c in chunks:
        lines.append(f"### {c['doc_title']} — {c['heading']}\n{c['text']}")
    return "\n\n".join(lines)


def _build_prompt(predicted_class, confidence, region_description, chunks) -> str:
    class_name = config.CLASS_NAMES[predicted_class]
    risk_tier = config.RISK_TIER[predicted_class]
    context = _format_context(chunks)
    return f"""You are a dermatology decision-support assistant. A convolutional \
neural network analyzed a skin lesion image and produced the following result. \
Use ONLY the retrieved knowledge context below plus the prediction details to \
answer — do not invent medical facts that aren't supported by the context.

PREDICTION
- Predicted class: {class_name} ({predicted_class})
- Model confidence: {confidence*100:.1f}%
- Risk tier: {risk_tier}
- Grad-CAM explainability: the model's attention was concentrated on {region_description}.

RETRIEVED KNOWLEDGE CONTEXT
{context}

Write your answer as four clearly labeled sections:
1. PATIENT EXPLANATION — 3-5 plain-language sentences a non-medical patient can \
understand: what the prediction means, in general terms, and why (referencing the \
Grad-CAM region qualitatively, e.g. "the AI focused on the irregular border").
2. CLINICIAN SUMMARY — 2-4 concise, clinically-toned sentences for a physician \
(prediction, confidence, salient supporting features from the context, differential \
considerations if relevant).
3. RECOMMENDED NEXT STEPS — 2-3 bullet-style sentences on what the patient should do \
next, calibrated to the risk tier ({risk_tier}).
4. DISCLAIMER — one sentence making clear this is an AI decision-support prototype, \
not a diagnosis, and a licensed clinician must confirm any finding.
"""


def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=config.OLLAMA_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# Each pattern greedily consumes any markdown bold markers ("**"), "#"
# heading markers, and "N." numbering surrounding the heading text, so
# the *whole* heading (not just the bare keyword) is what delimits
# sections below. Matching only the bare keyword (e.g. "CLINICIAN
# SUMMARY") would leave the preceding "**2. " leaking into the tail of
# the previous section, which is exactly what real Ollama output does
# since it formats headings as markdown.
_SECTION_PATTERNS = {
    "patient_explanation": r"[#*\s]*1?\.?\s*PATIENT EXPLANATION[#*\s:]*",
    "clinician_summary": r"[#*\s]*2?\.?\s*CLINICIAN SUMMARY[#*\s:]*",
    "recommended_action": r"[#*\s]*3?\.?\s*RECOMMENDED NEXT STEPS[#*\s:]*",
    "disclaimer": r"[#*\s]*4?\.?\s*DISCLAIMER[#*\s:]*",
}


def _parse_llm_sections(raw_text: str) -> dict:
    """Best-effort split of the LLM's free-text response into the four
    sections. Falls back to putting everything in patient_explanation if
    the model didn't follow the requested structure."""
    positions = []  # (match_start, key, match_end)
    for key, pattern in _SECTION_PATTERNS.items():
        m = re.search(pattern, raw_text, re.IGNORECASE)
        if m:
            positions.append((m.start(), key, m.end()))
    positions.sort()

    if not positions:
        return {"patient_explanation": raw_text.strip(), "clinician_summary": "",
                "recommended_action": "", "disclaimer": config.DISCLAIMER}

    sections = {}
    for i, (start, key, match_end) in enumerate(positions):
        section_end = positions[i + 1][0] if i + 1 < len(positions) else len(raw_text)
        text = raw_text[match_end:section_end].strip(" \n:-*")
        sections[key] = text
    for key in _SECTION_PATTERNS:
        sections.setdefault(key, "")
    if not sections.get("disclaimer"):
        sections["disclaimer"] = config.DISCLAIMER
    return sections


def _template_fallback(predicted_class, confidence, region_description, chunks) -> Explanation:
    """Deterministic, no-LLM-required explanation built directly from the
    retrieved knowledge chunks. Ensures the app degrades gracefully."""
    class_name = config.CLASS_NAMES[predicted_class]
    risk_tier = config.RISK_TIER[predicted_class]

    by_heading = {c["heading"].lower(): c["text"] for c in chunks}

    def find(*keywords):
        for heading, text in by_heading.items():
            if any(k in heading for k in keywords):
                return text
        return None

    overview = find("overview") or (chunks[0]["text"] if chunks else "")
    when_to_seek = find("when to seek", "seek care")
    treatment = find("treatment")

    patient = (
        f"The AI model predicts this lesion is most consistent with {class_name.lower()} "
        f"({confidence*100:.0f}% model confidence). The model's attention (Grad-CAM) was "
        f"concentrated on {region_description}, which contributed most to this prediction. "
        f"{overview.split('. ')[0] if overview else ''}. "
        f"This is an AI estimate, not a confirmed diagnosis."
    )

    clinician = (
        f"CNN prediction: {predicted_class.upper()} ({class_name}), confidence "
        f"{confidence*100:.1f}%, risk tier: {risk_tier}. Grad-CAM attention localized to "
        f"{region_description}. Retrieved reference notes: "
        f"{(overview or '')[:280]}"
    )

    if risk_tier in ("malignant", "pre-malignant"):
        action = (
            f"Given the {risk_tier} risk tier, recommend prompt dermatologist evaluation "
            f"(days, not months) for dermoscopic exam and possible biopsy. "
        )
    else:
        action = (
            f"Given the benign risk tier, routine dermatologist follow-up is reasonable "
            f"unless the lesion is changing, bleeding, or symptomatic. "
        )
    if when_to_seek:
        action += when_to_seek.split(". ")[0] + "."
    elif treatment:
        action += treatment.split(". ")[0] + "."

    return Explanation(
        patient_explanation=patient,
        clinician_summary=clinician,
        recommended_action=action,
        disclaimer=config.DISCLAIMER,
        sources=[f"{c['doc_title']} — {c['heading']}" for c in chunks],
        used_llm=False,
    )


class LLMExplainer:
    def __init__(self, retriever):
        self.retriever = retriever

    def explain(self, predicted_class: str, confidence: float, region_description: str) -> Explanation:
        query = build_query(predicted_class, confidence, region_description)
        chunks = self.retriever.retrieve(query, class_filter=predicted_class)
        sources = [f"{c['doc_title']} — {c['heading']}" for c in chunks]

        prompt = _build_prompt(predicted_class, confidence, region_description, chunks)
        try:
            raw = _call_ollama(prompt)
            sections = _parse_llm_sections(raw)
            return Explanation(
                patient_explanation=sections.get("patient_explanation", "").strip(),
                clinician_summary=sections.get("clinician_summary", "").strip(),
                recommended_action=sections.get("recommended_action", "").strip(),
                disclaimer=sections.get("disclaimer", config.DISCLAIMER).strip(),
                sources=sources,
                used_llm=True,
            )
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"[llm_explainer] Ollama unavailable ({e}); using template fallback.")
            return _template_fallback(predicted_class, confidence, region_description, chunks)


if __name__ == "__main__":
    from src.rag.retriever import KnowledgeRetriever

    retriever = KnowledgeRetriever()
    explainer = LLMExplainer(retriever)
    result = explainer.explain("mel", 0.93, "a small, localized area in the upper-right part of the lesion")
    print(json.dumps(result.__dict__, indent=2))
