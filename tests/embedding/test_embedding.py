
import numpy as np
import pytest

from src.embeddings.embedding_model import EmbeddingModel


@pytest.fixture(scope="module")
def embedding_model() -> EmbeddingModel:
    """Load the embedding model once for this test module."""
    return EmbeddingModel()


def test_embed_text_returns_384_dimensional_vector(
    embedding_model: EmbeddingModel,
) -> None:
    vector = embedding_model.embed_text("Python is a programming language.")

    assert vector.shape == (384,)
    assert np.isfinite(vector).all()


def test_embed_text_rejects_empty_string(
    embedding_model: EmbeddingModel,
) -> None:
    with pytest.raises(ValueError, match="Text must not be empty."):
        embedding_model.embed_text("")


def test_embed_text_rejects_whitespace_only(
    embedding_model: EmbeddingModel,
) -> None:
    with pytest.raises(ValueError, match="Text must not be empty."):
        embedding_model.embed_text("   ")


def test_embed_texts_preserves_input_order(
    embedding_model: EmbeddingModel,
) -> None:
    texts = [
        "Python is a programming language.",
        "PostgreSQL stores structured data.",
        "The weather is sunny today.",
    ]

    batch_vectors = embedding_model.embed_texts(texts)
    individual_vectors = [
        embedding_model.embed_text(text) for text in texts
    ]

    assert batch_vectors.shape == (3, 384)

    for batch_vector, individual_vector in zip(
        batch_vectors, individual_vectors
    ):
        assert np.allclose(batch_vector, individual_vector, atol=1e-5)


def test_embed_texts_rejects_invalid_input(
    embedding_model: EmbeddingModel,
) -> None:
    with pytest.raises(ValueError, match="Texts must not be empty."):
        embedding_model.embed_texts([])

    with pytest.raises(
        ValueError,
        match="Each text must contain non-whitespace characters.",
    ):
        embedding_model.embed_texts(["valid text", "   "])
