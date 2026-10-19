
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.embeddings.embedding_model import EmbeddingModel


def semantic_search(
    query: str,
    chunks: list[dict[str, Any]],
    embeddings: NDArray[np.float32],
    embedding_model: EmbeddingModel,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the most relevant chunks for a text query."""

    # Validate inputs
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have one embedding.")

    if not chunks:
        return []

    if any(
        not isinstance(chunk.get("text"), str)
        or not chunk["text"].strip()
        for chunk in chunks
    ):
        raise ValueError("Each chunk must contain non-empty text.")

    # Generate the query embedding
    query_embedding = embedding_model.embed_text(query)

    # Validate embedding dimensions and values
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != len(query_embedding)
    ):
        raise ValueError("Embedding dimensions do not match.")

    if (
        not np.isfinite(embeddings).all()
        or not np.isfinite(query_embedding).all()
    ):
        raise ValueError("Embeddings must contain finite values.")

    # Calculate cosine similarity
    query_norm = np.linalg.norm(query_embedding)
    chunk_norms = np.linalg.norm(embeddings, axis=1)

    if query_norm == 0 or np.any(chunk_norms == 0):
        raise ValueError("Embeddings must not be zero vectors.")

    scores = (embeddings @ query_embedding) / (
        chunk_norms * query_norm
    )

    # Rank chunks by descending similarity
    ranked_indices = np.argsort(
        -scores,
        kind="stable",
    )[:top_k]

    # Return matching chunks with their similarity scores
    return [
        {
            **chunks[int(index)],
            "similarity": float(scores[index]),
        }
        for index in ranked_indices
    ]
