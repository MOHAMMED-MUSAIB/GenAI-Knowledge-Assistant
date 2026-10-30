from uuid import NAMESPACE_URL, uuid5

import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.embeddings.embedding_model import EmbeddingModel
from src.storage import qdrant_store
from src.storage.qdrant_store import (
    COLLECTION_NAME,
    create_collection_if_not_exists,
    get_chunk_by_id,
    get_qdrant_client,
    store_chunk,
)


def test_qdrant_client_uses_configured_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    fake_client = object()

    def fake_qdrant_client(*, url: str, timeout: int) -> object:
        captured["url"] = url
        captured["timeout"] = timeout
        return fake_client

    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setattr(qdrant_store, "QdrantClient", fake_qdrant_client)

    assert get_qdrant_client() is fake_client
    assert captured["url"] == "http://qdrant:6333"
    assert captured["timeout"] == 30


def test_collection_configuration() -> None:
    """Verify the vector configuration matches the embedding model."""
    assert EmbeddingModel.EMBEDDING_DIMENSION == 384
    assert models.Distance.COSINE.value == "Cosine"


def _assert_collection_vector_size(client: QdrantClient) -> None:
    info = client.get_collection(COLLECTION_NAME)
    vectors = info.config.params.vectors
    assert vectors is not None

    if isinstance(vectors, dict):
        assert len(vectors) == 1
        vectors = next(iter(vectors.values()))

    assert vectors.size == EmbeddingModel.EMBEDDING_DIMENSION


def test_create_collection_if_not_exists() -> None:
    """Create the collection with the expected vector configuration."""
    client = get_qdrant_client()

    try:
        create_collection_if_not_exists(client)

        assert client.collection_exists(COLLECTION_NAME)

        _assert_collection_vector_size(client)
    finally:
        client.close()


def test_create_collection_if_not_exists_is_idempotent() -> None:
    """Calling the function twice preserves the existing collection."""
    client = get_qdrant_client()

    try:
        create_collection_if_not_exists(client)
        create_collection_if_not_exists(client)

        assert client.collection_exists(COLLECTION_NAME)
        _assert_collection_vector_size(client)
    finally:
        client.close()


def test_store_chunk_preserves_payload() -> None:
    """Store a chunk and preserve its text and metadata."""
    client = QdrantClient(":memory:")

    chunk: dict[str, str | int] = {
        "text": "Python is a programming language.",
        "chunk_index": 0,
        "word_count": 5,
        "document_id": "doc-123",
        "filename": "python.txt",
    }
    vector = np.zeros(EmbeddingModel.EMBEDDING_DIMENSION, dtype=np.float32)
    vector[0] = 1.0

    try:
        create_collection_if_not_exists(client)

        store_chunk(
            client=client,
            chunk=chunk,
            embedding=vector,
        )

        results = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1,
            with_payload=True,
            with_vectors=True,
        )
        points = results[0]

        assert len(points) == 1
        assert points[0].payload is not None
        assert points[0].payload["text"] == chunk["text"]
        assert points[0].payload["document_id"] == "doc-123"
        assert points[0].payload["filename"] == "python.txt"
        assert points[0].payload["chunk_index"] == 0
        assert points[0].payload["word_count"] == 5
        point_vector = points[0].vector
        assert isinstance(point_vector, list)
        assert len(point_vector) == EmbeddingModel.EMBEDDING_DIMENSION
    finally:
        client.close()


def test_store_chunk_rejects_missing_metadata() -> None:
    """Reject chunks missing required metadata."""
    client = QdrantClient(":memory:")
    chunk: dict[str, str | int] = {
        "text": "Python is a programming language.",
        "chunk_index": 0,
        "word_count": 5,
        "document_id": "doc-123",
        # filename intentionally missing
    }
    vector = np.zeros(EmbeddingModel.EMBEDDING_DIMENSION, dtype=np.float32)

    try:
        create_collection_if_not_exists(client)

        with pytest.raises(ValueError, match="missing required metadata"):
            store_chunk(client, chunk, vector)
    finally:
        client.close()


def test_store_chunk_rejects_wrong_embedding_dimension() -> None:
    """Reject embeddings with an incorrect dimension."""
    client = QdrantClient(":memory:")
    chunk: dict[str, str | int] = {
        "text": "Python is a programming language.",
        "chunk_index": 0,
        "word_count": 5,
        "document_id": "doc-123",
        "filename": "python.txt",
    }
    vector = np.zeros(128, dtype=np.float32)

    try:
        create_collection_if_not_exists(client)

        with pytest.raises(ValueError, match="384 dimensions"):
            store_chunk(client, chunk, vector)
    finally:
        client.close()

def test_store_chunk_rejects_non_finite_embedding() -> None:
    """Reject embeddings containing NaN or infinity."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    chunk: dict[str, str | int] = {
        "text": "Python is a programming language.",
        "chunk_index": 0,
        "word_count": 5,
        "document_id": "doc-123",
        "filename": "python.txt",
    }
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = np.nan

    try:
        create_collection_if_not_exists(client)

        with pytest.raises(ValueError, match="NaN"):
            store_chunk(client, chunk, vector)
    finally:
        client.close()
def test_store_chunk_rejects_negative_chunk_index() -> None:
    """Reject a chunk with a negative index."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    chunk: dict[str, str | int] = {
        "text": "Python is a programming language.",
        "chunk_index": -1,
        "word_count": 5,
        "document_id": "doc-123",
        "filename": "python.txt",
    }
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1.0

    try:
        create_collection_if_not_exists(client)

        with pytest.raises(ValueError, match="chunk_index"):
            store_chunk(client, chunk, vector)
    finally:
        client.close()

def test_store_chunk_rejects_negative_word_count() -> None:
    """Reject a chunk with a negative word count."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    chunk: dict[str, str | int] = {
        "text": "Python is a programming language.",
        "chunk_index": 0,
        "word_count": -5,
        "document_id": "doc-123",
        "filename": "python.txt",
    }
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1.0

    try:
        create_collection_if_not_exists(client)

        with pytest.raises(ValueError, match="word_count"):
            store_chunk(client, chunk, vector)
    finally:
        client.close()

def test_store_chunk_rejects_empty_document_id() -> None:
    client = QdrantClient(":memory:")
    create_collection_if_not_exists(client)

    chunk: dict[str, str | int] = {
        "text": "A sample document chunk.",
        "chunk_index": 0,
        "word_count": 4,
        "document_id": "   ",
        "filename": "sample.txt",
    }
    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    with pytest.raises(ValueError, match="document_id"):
        store_chunk(client, chunk, embedding)

def test_store_chunk_rejects_empty_filename() -> None:
    client = QdrantClient(":memory:")
    create_collection_if_not_exists(client)

    chunk: dict[str, str | int] = {
        "text": "A sample document chunk.",
        "chunk_index": 0,
        "word_count": 4,
        "document_id": "doc-123",
        "filename": "   ",
    }
    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    with pytest.raises(ValueError, match="filename"):
        store_chunk(client, chunk, embedding)

def test_store_chunk_upsert_updates_existing_point() -> None:
    """Re-indexing the same chunk updates its existing point."""
    client = QdrantClient(":memory:")
    create_collection_if_not_exists(client)

    chunk: dict[str, str | int] = {
        "text": "Original text.",
        "chunk_index": 0,
        "word_count": 2,
        "document_id": "doc-upsert-test",
        "filename": "sample.txt",
    }

    first_embedding = np.zeros(384, dtype=np.float32)
    first_embedding[0] = 1.0

    updated_chunk: dict[str, str | int] = {
        **chunk,
        "text": "Updated text.",
        "word_count": 2,
    }

    second_embedding = np.zeros(384, dtype=np.float32)
    second_embedding[1] = 1.0

    try:
        store_chunk(client, chunk, first_embedding)
        store_chunk(client, updated_chunk, second_embedding)

        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=True,
        )

        assert len(points) == 1
        assert points[0].payload is not None
        assert points[0].payload["text"] == "Updated text."
        assert points[0].vector == second_embedding.tolist()

        point_id = str(uuid5(NAMESPACE_URL, "doc-upsert-test:0"))
        retrieved = get_chunk_by_id(client, point_id)
        assert retrieved is not None
        assert retrieved["text"] == "Updated text."
        assert retrieved["document_id"] == "doc-upsert-test"
        retrieved = get_chunk_by_id(client, str(points[0].id))

        assert retrieved is not None
        assert retrieved["text"] == "Updated text."
        assert retrieved["document_id"] == "doc-upsert-test"
    finally:
        client.close()
