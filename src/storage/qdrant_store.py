import os
from uuid import uuid5, NAMESPACE_URL

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.embeddings.embedding_model import EmbeddingModel

COLLECTION_NAME = "document_chunks"


def get_qdrant_client() -> QdrantClient:
    """Create a client connected to the configured Qdrant server."""
    return QdrantClient(
        url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        timeout=30,
    )


def create_collection_if_not_exists(
    client: QdrantClient,
) -> None:
    """Create the document collection if it does not already exist."""
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EmbeddingModel.EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )


def store_chunk(
    client: QdrantClient,
    chunk: dict[str, str | int],
    embedding: NDArray[np.float32],
) -> None:
    """Store a document chunk, its embedding, and its metadata."""
    required_fields = {
        "text",
        "chunk_index",
        "word_count",
        "document_id",
        "filename",
    }

    if not required_fields.issubset(chunk):
        raise ValueError("Chunk is missing required metadata fields.")

    if not isinstance(chunk["text"], str) or not chunk["text"].strip():
        raise ValueError("Chunk text must not be empty.")
    chunk_index = chunk["chunk_index"]
    
    document_id = chunk["document_id"]

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")
    
    filename = chunk["filename"]

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")



    if (
        not isinstance(chunk_index, int)
        or isinstance(chunk_index, bool)
        or chunk_index < 0
    ):
        raise ValueError("chunk_index must be a non-negative integer.")
    word_count = chunk["word_count"]

    if (
        not isinstance(word_count, int)
        or isinstance(word_count, bool)
        or word_count <= 0
    ):
        raise ValueError("word_count must be a positive integer.")

    if embedding.shape != (EmbeddingModel.EMBEDDING_DIMENSION,):
        raise ValueError(
            f"Embedding must contain exactly "
            f"{EmbeddingModel.EMBEDDING_DIMENSION} dimensions."
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid5(
                            NAMESPACE_URL,
                            f"{document_id}:{chunk_index}",
                        )
                    ),
                vector=embedding.tolist(),
                payload=chunk,
            )
        ],
    )


def get_chunk_by_id(
    client: QdrantClient,
    point_id: str,
) -> dict[str, object] | None:
    """Retrieve a stored chunk's payload by its point ID."""
    if not point_id.strip():
        raise ValueError("point_id must not be empty.")

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point_id],
        with_payload=True,
        with_vectors=False,
    )

    if not points:
        return None

    payload = points[0].payload
    if payload is None:
        return None

    return payload
def delete_document_chunks(
    client: QdrantClient,
    document_id: str,
) -> None:
    """Delete all stored chunks belonging to a document."""
    if not document_id.strip():
        raise ValueError("document_id must not be empty.")

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        ),
        wait=True,
    )