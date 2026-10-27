
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api import main


def test_uploaded_file_is_removed_when_text_extraction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)

    def fail_read_txt(_file_path: str) -> str:
        raise ValueError("Simulated extraction failure")

    monkeypatch.setattr(main, "read_txt", fail_read_txt)

    client = TestClient(main.app)

    response = client.post(
        "/documents/upload",
        files={"file": ("broken.txt", b"some text", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Simulated extraction failure"
    assert list(tmp_path.iterdir()) == []
