
from typing import Literal, Protocol, cast

from numpy.typing import NDArray
import numpy as np
from sentence_transformers import SentenceTransformer


class _SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: Literal[True],
        normalize_embeddings: bool,
    ) -> NDArray[np.float32]: ...


class EmbeddingModel:
    """Generate vector embeddings from text using a local model."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    def __init__(self) -> None:
        """Load the embedding model on the CPU."""
        self._model = cast(
            _SentenceEncoder,
            SentenceTransformer(self.MODEL_NAME, device="cpu"),
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> NDArray[np.float32]:
        """Convert multiple non-empty texts into embedding vectors."""
        if not texts:
            raise ValueError("Texts must not be empty.")

        if any(not text.strip() for text in texts):
            raise ValueError("Each text must contain non-whitespace characters.")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    def embed_text(self, text: str) -> NDArray[np.float32]:
        """Convert one non-empty text into a normalized embedding vector."""
        if not text.strip():
            raise ValueError("Text must not be empty.")

        return self.embed_texts([text])[0]
