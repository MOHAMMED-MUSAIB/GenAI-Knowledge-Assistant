
from typing import Any

import numpy as np
import pytest

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.semantic_search import semantic_search


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    """Load the embedding model once for this test module."""
    return EmbeddingModel()


def test_python_query_retrieves_python_chunk_first(
    embedding_model: EmbeddingModel,
) -> None:
    chunks: list[dict[str, Any]] = [
        {
            "text": "Python uses variables, functions, and loops to write programs.",
            "chunk_index": 0,
        },
        {
            "text": "PostgreSQL stores and retrieves data using SQL queries.",
            "chunk_index": 1,
        },
        {
            "text": "Sunny weather often brings clear skies and warm temperatures.",
            "chunk_index": 2,
        },
    ]

    chunk_embeddings = embedding_model.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    results = semantic_search(
        query="How do I write Python code?",
        chunks=chunks,
        embeddings=chunk_embeddings,
        embedding_model=embedding_model,
        top_k=3,
    )

    assert results
    assert results[0]["text"] == chunks[0]["text"]
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_semantic_search_respects_top_k(
    embedding_model: EmbeddingModel,
) -> None:
    chunks: list[dict[str, Any]] = [
        {"text": "Python is used to build software.", "chunk_index": 0},
        {"text": "PostgreSQL is a relational database.", "chunk_index": 1},
        {"text": "Sunny weather brings clear skies.", "chunk_index": 2},
    ]

    embeddings = embedding_model.embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    results = semantic_search(
        query="How do I write Python software?",
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["similarity"] >= results[1]["similarity"]


def test_semantic_search_rejects_invalid_query_and_top_k(
    embedding_model: EmbeddingModel,
) -> None:
    chunks = [{"text": "Python is a programming language."}]
    embeddings = embedding_model.embed_texts(
        [chunks[0]["text"]]
    )

    with pytest.raises(ValueError, match="Query must not be empty."):
        semantic_search(
            query="   ",
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
        )

    with pytest.raises(ValueError, match="top_k must be at least 1."):
        semantic_search(
            query="Python programming",
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            top_k=0,
        )


def test_semantic_search_rejects_mismatched_counts(
    embedding_model: EmbeddingModel,
) -> None:
    chunks = [
        {"text": "Python is a programming language."},
        {"text": "PostgreSQL is a database."},
    ]
    embeddings = embedding_model.embed_texts(
        [chunks[0]["text"]]
    )

    with pytest.raises(
        ValueError,
        match="Each chunk must have one embedding.",
    ):
        semantic_search(
            query="Python programming",
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
        )


def test_semantic_search_returns_empty_for_no_chunks(
    embedding_model: EmbeddingModel,
) -> None:
    embeddings = np.empty((0, 384), dtype=np.float32)

    results = semantic_search(
        query="Python programming",
        chunks=[],
        embeddings=embeddings,
        embedding_model=embedding_model,
    )

    assert results == []


def test_semantic_search_rejects_wrong_embedding_dimensions(
    embedding_model: EmbeddingModel,
) -> None:
    chunks = [{"text": "Python is a programming language."}]
    wrong_embeddings = np.zeros((1, 128), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="Embedding dimensions do not match.",
    ):
        semantic_search(
            query="Python programming",
            chunks=chunks,
            embeddings=wrong_embeddings,
            embedding_model=embedding_model,
        )
