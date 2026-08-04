"""
Retrieval component of the RAG pipeline.

Loads the FAISS index built by build_index.py and retrieves the most
relevant knowledge-base chunks for a given query. Because we already
know the CNN's predicted class, retrieval is biased toward that class's
document (a "filtered RAG" pattern) while still allowing broader
semantic search when useful (e.g., a differential-diagnosis query).
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config


class KnowledgeRetriever:
    def __init__(self):
        if not (os.path.exists(config.FAISS_INDEX_PATH) and os.path.exists(config.CHUNK_STORE_PATH)):
            raise FileNotFoundError(
                "RAG index not found. Run `python -m src.rag.build_index` first."
            )
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.CHUNK_STORE_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def retrieve(self, query: str, top_k: int = config.RAG_TOP_K, class_filter: str = None):
        """Return the top_k most relevant chunks for `query`.

        Args:
            query: natural-language query, ideally enriched with the
                   predicted class, confidence, and Grad-CAM description
                   (see src/rag/llm_explainer.py::build_query).
            class_filter: if set to a class code (e.g. 'mel'), restrict
                          results to that class's document — the common
                          case, since we already know the CNN's prediction.
        """
        query_vec = self._embed(query)

        if class_filter:
            candidate_idx = [i for i, c in enumerate(self.chunks) if c["class_code"] == class_filter]
            if not candidate_idx:
                candidate_idx = list(range(len(self.chunks)))
            candidate_vecs = np.stack([
                self.index.reconstruct(i) for i in candidate_idx
            ])
            sims = candidate_vecs @ query_vec[0]
            order = np.argsort(-sims)[:top_k]
            results = [
                {**self.chunks[candidate_idx[i]], "score": float(sims[i])}
                for i in order
            ]
            return results

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results


if __name__ == "__main__":
    retriever = KnowledgeRetriever()
    for cls in ["mel", "bcc", "nv"]:
        print(f"\n=== Query for class '{cls}' ===")
        results = retriever.retrieve(
            f"Explain {config.CLASS_NAMES[cls]} to a patient: symptoms, risk, next steps",
            class_filter=cls,
        )
        for r in results:
            print(f"  [{r['score']:.3f}] {r['heading']}: {r['text'][:100]}...")
