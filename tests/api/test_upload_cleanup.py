
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from numpy.typing import NDArray
import pytest
from qdrant_client import QdrantClient

from src.api import auth, main
from src.embeddings.embedding_model import EmbeddingModel


def test_uploaded_file_is_removed_when_text_extraction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    def fail_read_txt(file_path: str) -> str:
        assert file_path
        raise ValueError("Simulated extraction failure")

    monkeypatch.setattr(main, "read_txt", fail_read_txt)

    client = TestClient(main.app)

    response = client.post(
        "/documents/upload",
        files={"file": ("broken.txt", b"some text", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Simulated extraction failure"
    assert list(tmp_path.iterdir()) == []
def test_oversized_upload_returns_413_without_saving_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    client = TestClient(main.app)
    oversized_content = b"x" * (main.MAX_FILE_SIZE_BYTES + 1)

    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "large.txt",
                oversized_content,
                "text/plain",
            )
        },
        headers=auth_headers,
    )

    assert response.status_code == 413
    assert response.json()["detail"] == (
        "Uploaded file exceeds the 10 MB size limit."
    )
    assert list(tmp_path.iterdir()) == []

def test_qdrant_chunks_are_cleaned_up_when_storage_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    fake_qdrant_client = QdrantClient(":memory:")
    monkeypatch.setattr(
        main,
        "get_qdrant_client_instance",
        lambda: fake_qdrant_client,
    )

    def fake_read_txt(file_path: str) -> str:
        assert Path(file_path).is_file()
        return "This is test document content."

    monkeypatch.setattr(main, "read_txt", fake_read_txt)

    def fake_chunk_text(
        text: str,
        *,
        document_id: str | None = None,
        filename: str | None = None,
    ) -> list[dict[str, str | int]]:
        assert text.strip()
        return [
            {
                "text": "First chunk",
                "chunk_index": 0,
                "word_count": 2,
                "document_id": document_id or "",
                "filename": filename or "",
            },
            {
                "text": "Second chunk",
                "chunk_index": 1,
                "word_count": 2,
                "document_id": document_id or "",
                "filename": filename or "",
            },
        ]

    monkeypatch.setattr(main, "chunk_text", fake_chunk_text)

    fake_embedding_model = object.__new__(EmbeddingModel)

    def fake_embed_texts(texts: list[str]) -> NDArray[np.float32]:
        return np.full(
            (len(texts), EmbeddingModel.EMBEDDING_DIMENSION),
            0.1,
            dtype=np.float32,
        )

    monkeypatch.setattr(fake_embedding_model, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(main, "get_embedding_model", lambda: fake_embedding_model)

    owner = auth.authenticate_user("test-user", "test-password-long-enough")
    assert owner is not None
    store_calls = 0

    def fail_on_second_chunk(
        *,
        client: QdrantClient,
        chunk: dict[str, str | int],
        embedding: NDArray[np.float32],
    ) -> None:
        nonlocal store_calls
        assert client is not None
        assert chunk["text"]
        assert chunk["owner_id"] == owner.user_id
        assert embedding.shape == (EmbeddingModel.EMBEDDING_DIMENSION,)
        store_calls += 1

        if store_calls == 2:
            raise RuntimeError("Simulated Qdrant storage failure")

    monkeypatch.setattr(main, "store_chunk", fail_on_second_chunk)

    cleanup_calls: list[str] = []

    def record_cleanup(
        client: QdrantClient,
        document_id: str,
    ) -> None:
        assert client is fake_qdrant_client
        cleanup_calls.append(document_id)

    monkeypatch.setattr(
        main,
        "delete_document_chunks",
        record_cleanup,
    )

    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post(
        "/documents/upload",
        files={"file": ("sample.txt", b"test content", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 500
    assert store_calls == 2
    assert len(cleanup_calls) == 1
    assert cleanup_calls[0].endswith(".txt")
    assert list(tmp_path.iterdir()) == []
    fake_qdrant_client.close()
