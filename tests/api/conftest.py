from fastapi.testclient import TestClient
import pytest

from src.api import auth, main


@pytest.fixture
def auth_headers(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    auth_path = tmp_path_factory.mktemp("auth-state") / "auth.sqlite3"
    monkeypatch.setattr(auth, "DATABASE_PATH", auth_path)
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-with-at-least-32-bytes")
    with TestClient(main.app) as client:
        response = client.post(
            "/auth/register",
            json={
                "username": "test-user",
                "password": "test-password-long-enough",
            },
        )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
