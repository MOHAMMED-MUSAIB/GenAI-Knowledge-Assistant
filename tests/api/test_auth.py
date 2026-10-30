from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api import auth, main


def test_registration_and_login_issue_access_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(auth, "DATABASE_PATH", tmp_path / "auth.sqlite3")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-with-at-least-32-bytes")
    client = TestClient(main.app)

    registration = client.post(
        "/auth/register",
        json={
            "username": "Alice",
            "password": "a-secure-password-123",
        },
    )
    login = client.post(
        "/auth/login",
        json={
            "username": "alice",
            "password": "a-secure-password-123",
        },
    )

    assert registration.status_code == 201
    assert registration.json()["username"] == "alice"
    assert login.status_code == 200
    assert login.json()["access_token"]
    assert login.json()["token_type"] == "bearer"


def test_duplicate_usernames_are_rejected_case_insensitively(
    auth_headers: dict[str, str],
) -> None:
    response = TestClient(main.app).post(
        "/auth/register",
        json={
            "username": "TEST-USER",
            "password": "another-secure-password",
        },
    )

    assert response.status_code == 409


def test_protected_endpoints_require_a_valid_token() -> None:
    client = TestClient(main.app)

    responses = (
        client.get("/documents"),
        client.get(
            "/documents",
            headers={"Authorization": "Bearer not.a.valid-token"},
        ),
        client.post(
            "/documents/upload",
            files={"file": ("sample.txt", b"private content", "text/plain")},
        ),
        client.post("/questions/ask", json={"question": "What is this?"}),
    )

    assert [response.status_code for response in responses] == [
        401,
        401,
        401,
        401,
    ]


def test_document_list_is_scoped_to_the_authenticated_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    auth_headers: dict[str, str],
) -> None:
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path / "uploads")
    main.UPLOAD_DIR.mkdir()
    (main.UPLOAD_DIR / "mine.txt").write_text("mine", encoding="utf-8")
    (main.UPLOAD_DIR / "theirs.txt").write_text("theirs", encoding="utf-8")

    owner = auth.authenticate_user("test-user", "test-password-long-enough")
    other = auth.register_user("other-user", "another-secure-password")
    assert owner is not None
    auth.add_document_owner("mine.txt", owner.user_id, 4)
    auth.add_document_owner("theirs.txt", other.user_id, 6)

    response = TestClient(main.app).get("/documents", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == [{"document_id": "mine.txt", "size_bytes": 4}]
