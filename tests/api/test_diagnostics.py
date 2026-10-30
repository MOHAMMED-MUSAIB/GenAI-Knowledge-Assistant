from contextlib import nullcontext
import logging

from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from src.api import main


def test_request_logs_exclude_query_strings(caplog) -> None:
    caplog.set_level(logging.INFO, logger=main.__name__)

    response = TestClient(main.app).get("/health?access_token=query-secret")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    assert "event=request_complete" in caplog.text
    assert "route=/health" in caplog.text
    assert "status_code=200" in caplog.text
    assert "query-secret" not in caplog.text


def test_unexpected_error_is_correlated_and_sanitized(caplog) -> None:
    secret = "sensitive-exception-detail"

    def fail() -> None:
        raise RuntimeError(secret)

    route = APIRoute("/_diagnostics-test", fail, methods=["GET"])
    main.app.router.routes.append(route)
    caplog.set_level(logging.ERROR, logger=main.__name__)
    try:
        response = TestClient(main.app, raise_server_exceptions=False).get(
            "/_diagnostics-test?token=query-secret"
        )
    finally:
        main.app.router.routes.remove(route)

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error."
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert f"request_id={response.headers['X-Request-ID']}" in caplog.text
    assert "event=unhandled_exception" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
    assert secret not in caplog.text
    assert secret not in response.text
    assert "query-secret" not in caplog.text


def test_readiness_reports_dependency_failure_without_error_details(
    monkeypatch,
    caplog,
) -> None:
    secret = "private-qdrant-url"
    requested_urls: list[str] = []

    def fail_urlopen(url: str, timeout: int):
        requested_urls.append(url)
        raise OSError(secret)

    monkeypatch.setattr(main, "urlopen", fail_urlopen)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333/")
    caplog.set_level(logging.ERROR, logger=main.__name__)

    response = TestClient(main.app).get("/health/ready")

    assert requested_urls == ["http://qdrant:6333/healthz"]
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "components": {"qdrant": "unavailable"},
    }
    assert "event=readiness_failed" in caplog.text
    assert "error_type=OSError" in caplog.text
    assert secret not in caplog.text


def test_readiness_reports_qdrant_as_healthy(monkeypatch) -> None:
    requested: list[tuple[str, int]] = []

    def fake_urlopen(url: str, timeout: int):
        requested.append((url, timeout))
        return nullcontext()

    monkeypatch.setattr(main, "urlopen", fake_urlopen)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    response = TestClient(main.app).get("/health/ready")

    assert requested == [("http://qdrant:6333/healthz", 2)]
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "components": {"qdrant": "ok"},
    }
