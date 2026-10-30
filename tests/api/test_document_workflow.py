from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
import pytest

from src.api import main
from src.embeddings.embedding_model import EmbeddingModel
from src.storage.qdrant_store import (
    COLLECTION_NAME,
    create_collection_if_not_exists,
)


class DeterministicEmbeddingModel:
    def embed_texts(self, texts: list[str]) -> NDArray[np.float32]:
        return np.ones(
            (len(texts), EmbeddingModel.EMBEDDING_DIMENSION),
            dtype=np.float32,
        )

    def embed_text(self, text: str) -> NDArray[np.float32]:
        assert text.strip()
        return self.embed_texts([text])[0]


def test_upload_retrieval_grounded_answer_and_citation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    client = QdrantClient(":memory:")
    create_collection_if_not_exists(client)
    monkeypatch.setattr(main, "get_qdrant_client_instance", lambda: client)
    monkeypatch.setattr(
        main,
        "get_embedding_model",
        lambda: DeterministicEmbeddingModel(),
    )

    captured_prompt: list[str] = []

    def fake_generate_answer(prompt: str) -> str:
        captured_prompt.append(prompt)
        return "The handbook says embeddings have 384 dimensions. [S1]"

    monkeypatch.setattr("src.rag.pipeline.generate_answer", fake_generate_answer)

    source_text = (
        "The handbook states that this project uses a 384-dimensional "
        "embedding model."
    )
    try:
        with TestClient(main.app) as api_client:
            upload_response = api_client.post(
                "/documents/upload",
                files={
                    "file": (
                        "handbook.txt",
                        source_text.encode("utf-8"),
                        "text/plain",
                    )
                },
                headers=auth_headers,
            )
            assert upload_response.status_code == 200
            uploaded: dict[str, Any] = upload_response.json()
            document_id = uploaded["document_id"]

            listing = api_client.get("/documents", headers=auth_headers)
            assert listing.status_code == 200
            assert listing.json() == [
                {"document_id": document_id, "size_bytes": len(source_text)}
            ]

            answer_response = api_client.post(
                "/questions/ask",
                json={
                    "question": "How many dimensions does the embedding model use?",
                    "top_k": 3,
                },
                headers=auth_headers,
            )

        assert answer_response.status_code == 200
        assert answer_response.json() == {
            "answer": "The handbook says embeddings have 384 dimensions. [S1]",
            "sources": [
                {
                    "filename": "handbook.txt",
                    "document_id": document_id,
                    "chunk_index": 0,
                }
            ],
        }
        assert len(captured_prompt) == 1
        assert source_text in captured_prompt[0]
        assert "using only the supplied context" in captured_prompt[0]
        assert f"Document ID: {document_id}" in captured_prompt[0]
        assert client.count(collection_name=COLLECTION_NAME).count == 1
    finally:
        client.close()
