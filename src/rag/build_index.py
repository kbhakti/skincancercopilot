"""
Build the local FAISS vector index over the curated dermatology
knowledge base (src/rag/knowledge/*.md).

Run once (and again whenever the .md files change):
    python -m src.rag.build_index

This is a fully local/offline RAG index: embeddings come from a small
sentence-transformers model that runs on CPU, and the vector store is a
FAISS flat index persisted to disk — no external API calls.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import glob
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


def split_into_sections(md_text: str):
    """Split a markdown file into (heading, body) sections on '## ' headers."""
    parts = re.split(r"\n(?=## )", md_text)
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        heading = lines[0].lstrip("#").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections.append((heading, body))
    return sections


def chunk_text(text: str, max_chars: int = config.CHUNK_MAX_CHARS,
               overlap: int = config.CHUNK_OVERLAP_CHARS):
    """Simple character-based chunking with overlap, splitting on sentence
    boundaries where possible so chunks stay coherent for retrieval."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            last_period = text.rfind(". ", start, end)
            if last_period != -1 and last_period > start + max_chars * 0.5:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


def build_chunks():
    chunks = []  # list of dicts: {class_code, class_name, heading, text}
    for path in sorted(glob.glob(os.path.join(config.KNOWLEDGE_DIR, "*.md"))):
        class_code = os.path.splitext(os.path.basename(path))[0]
        if class_code not in config.CLASS_NAMES:
            continue
        with open(path, "r", encoding="utf-8") as f:
            md_text = f.read()

        title_match = re.match(r"#\s+(.+)", md_text)
        doc_title = title_match.group(1).strip() if title_match else class_code

        for heading, body in split_into_sections(md_text):
            if not body:
                continue
            for piece in chunk_text(body):
                chunks.append({
                    "class_code": class_code,
                    "class_name": config.CLASS_NAMES[class_code],
                    "doc_title": doc_title,
                    "heading": heading,
                    "text": piece,
                })
    return chunks


def main():
    print(f"Reading knowledge base from: {config.KNOWLEDGE_DIR}")
    chunks = build_chunks()
    print(f"Built {len(chunks)} chunks from "
          f"{len(set(c['class_code'] for c in chunks))} documents.")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME} (first run downloads it)")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    texts = [f"{c['class_name']} - {c['heading']}: {c['text']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
    index.add(embeddings)

    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.CHUNK_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index -> {config.FAISS_INDEX_PATH}")
    print(f"Saved chunk metadata -> {config.CHUNK_STORE_PATH}")


if __name__ == "__main__":
    main()
