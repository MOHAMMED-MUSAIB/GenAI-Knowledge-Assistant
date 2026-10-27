
from typing import Any

from fastapi.testclient import TestClient
import pytest
from qdrant_client import QdrantClient

from src.api import main
from src.embeddings.embedding_model import EmbeddingModel


client = TestClient(main.app)


def test_ask_question_returns_answer_and_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_answer_question(
        question: object,
        embedding_model: EmbeddingModel,
        qdrant_client: QdrantClient,
        top_k: int,
    ) -> dict[str, Any]:
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
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "RAG retrieves relevant context before answering."
    )
    assert response.json()["sources"][0]["filename"] == "rag.txt"

def test_ask_question_rejects_blank_question():
    response = client.post(
        "/questions/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_question_rejects_invalid_top_k():
    response = client.post(
        "/questions/ask",
        json={"question": "What is RAG?", "top_k": 0},
    )

    assert response.status_code == 422

def test_ask_question_handles_pipeline_value_error(
    monkeypatch: pytest.MonkeyPatch,
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
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Question must not be empty."
