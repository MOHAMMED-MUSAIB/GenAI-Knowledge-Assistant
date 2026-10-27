
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api import main


def test_list_documents_returns_supported_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    (tmp_path / "sample.txt").write_text("Hello", encoding="utf-8")
    (tmp_path / "report.docx").write_bytes(b"fake docx")
    (tmp_path / "image.png").write_bytes(b"not a document")
    (tmp_path / "subfolder").mkdir()

    client = TestClient(main.app)
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == [
        {"document_id": "report.docx", "size_bytes": 9},
        {"document_id": "sample.txt", "size_bytes": 5},
    ]
