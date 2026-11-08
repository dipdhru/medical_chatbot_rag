"""Dense retrieval with FAISS and an out-of-scope threshold."""

from __future__ import annotations

from typing import Any, Protocol

import faiss
import numpy as np
import pandas as pd

from .config import QUERY_PREFIX


class Encoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Inner product on L2-normalised vectors is cosine similarity."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))
    return index


class Retriever:
    def __init__(self, encoder: Encoder, documents: pd.DataFrame, embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError(f"{len(documents)} documents but {len(embeddings)} embeddings")
        self.encoder = encoder
        self.documents = documents.reset_index(drop=True)
        self.index = build_index(embeddings)

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, document positions) for the k nearest documents, best first."""
        vector = self.encoder.encode(
            [QUERY_PREFIX + query], convert_to_numpy=True, normalize_embeddings=True
        )
        # FAISS and PyTorch ship separate OpenMP runtimes, and using both segfaults on macOS.
        # The limit is per thread and Streamlit reruns on new threads, so set it on every search.
        faiss.omp_set_num_threads(1)
        scores, positions = self.index.search(np.asarray(vector, dtype="float32"), k)
        return scores[0], positions[0]

    def retrieve(self, query: str, k: int, threshold: float) -> tuple[list[dict] | None, float]:
        """Return (results, best score); results is None when the query is out of scope."""
        scores, positions = self.search(query, k)
        best_score = float(scores[0])
        if best_score < threshold:
            return None, best_score

        results = []
        for rank, (score, position) in enumerate(zip(scores, positions, strict=True), start=1):
            row = self.documents.iloc[int(position)]
            url = str(row["url"]) if pd.notna(row["url"]) else ""
            if url.startswith("http"):  # every answer must be citable
                results.append(
                    {
                        "rank": rank,
                        "score": float(score),
                        "focus": row["focus"],
                        "question": row["question"],
                        "url": url,
                        "answer": row["answer"].strip(),
                        "text": row["text"],
                    }
                )
        return results or None, best_score
