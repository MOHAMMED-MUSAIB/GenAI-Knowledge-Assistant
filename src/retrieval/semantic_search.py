
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.embeddings.embedding_model import EmbeddingModel


def semantic_search(
    query: str,
    embedding_model: EmbeddingModel,
    qdrant_client: QdrantClient,
    collection_name: str,
    top_k: int = 3,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant document chunks from Qdrant.

    Args:
        query: User's natural-language question.
        embedding_model: Model used to embed the question.
        qdrant_client: Initialized Qdrant client.
        collection_name: Name of the collection containing document chunks.
        top_k: Maximum number of chunks to retrieve.

    Returns:
        Ranked chunks containing their payload, point ID, and similarity.

    Raises:
        ValueError: If an input or query embedding is invalid.
        QdrantClient errors: If the database query fails.
    """
    # 1. Validate inputs.
    if not query.strip():
        raise ValueError("Query must not be empty.")

    if (
        isinstance(top_k, bool)
        or type(top_k) is not int
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")

    if not collection_name.strip():
        raise ValueError("Collection name must not be empty.")

    if document_id is not None and not document_id.strip():
        raise ValueError("document_id must not be empty.")

    # 2. Generate the question embedding.
    query_embedding = np.asarray(
        embedding_model.embed_text(query),
        dtype=np.float32,
    )

    # 3. Validate the embedding.
    if query_embedding.ndim != 1 or query_embedding.size == 0:
        raise ValueError("Query embedding must be a non-empty 1D vector.")

    if not np.isfinite(query_embedding).all():
        raise ValueError("Query embedding must contain finite values.")

    if np.linalg.norm(query_embedding) == 0:
        raise ValueError("Query embedding must not be a zero vector.")
    query_filter = None

    if document_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )
    # 4. Retrieve nearest chunks from Qdrant.
    response = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    # 5. Convert Qdrant results into application-friendly dictionaries.
    results: list[dict[str, Any]] = []

    for point in response.points:
        payload = point.payload or {}

        results.append(
            {
                **payload,
                "id": str(point.id),
                "similarity": float(point.score),
            }
        )

    return results