
from typing import Any

from fastapi.testclient import TestClient
import pytest
from qdrant_client import QdrantClient

from src.api import main
from src.embeddings.embedding_model import EmbeddingModel


client = TestClient(main.app)


def test_ask_question_returns_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    captured: dict[str, str | None] = {}

    def fake_answer_question(
        question: object,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient,
        top_k: int,
        document_id: str | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        captured["document_id"] = document_id
        captured["owner_id"] = owner_id
        return {
            "answer": "RAG retrieves relevant context before answering.",
            "sources": [
                {
                    "filename": "rag.txt",
                    "document_id": "rag.txt",
                    "chunk_index": 0,
                }
            ],
        }

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/questions/ask",
        json={"question": "What is RAG?", "document_id": "doc-123"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "RAG retrieves relevant context before answering."
    )
    assert response.json()["sources"][0]["filename"] == "rag.txt"
    assert captured["document_id"] == "doc-123"
    assert captured["owner_id"] is not None


def test_ask_question_rejects_blank_question(
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/questions/ask",
        json={"question": "   "},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_ask_question_rejects_invalid_top_k(
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/questions/ask",
        json={"question": "What is RAG?", "top_k": 0},
        headers=auth_headers,
    )

    assert response.status_code == 422

def test_ask_question_handles_pipeline_value_error(
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    def fake_answer_question(**_kwargs: Any) -> None:
        raise ValueError("Question must not be empty.")

    monkeypatch.setattr(
        main,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/questions/ask",
        json={"question": "What is RAG?"},
        headers=auth_headers,
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Question must not be empty."
