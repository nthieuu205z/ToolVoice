import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.job_manager import manager
from backend.main import app


@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()
    manager._futures.clear()


def test_gemini_status_is_masked(client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "secret-value")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    response = client.get("/api/settings/gemini")

    assert response.status_code == 200
    assert response.json() == {"configured": True, "backend": "vertex"}
    assert "secret-value" not in response.text


def test_gemini_update_atomically_persists_without_echo(tmp_path, client, monkeypatch):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_SETTING=keep\n", encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    response = client.post(
        "/api/settings/gemini",
        json={"api_key": "key-with-#-hash", "backend": "vertex"},
    )

    assert response.json() == {"saved": True, "configured": True, "backend": "vertex"}
    assert env_path.read_text(encoding="utf-8") == (
        'OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\nGEMINI_API_KEY="key-with-#-hash"\n'
    )


def test_windows_tree_shutdown_uses_taskkill(monkeypatch):
    import backend.process_control as control

    calls = []
    monkeypatch.setattr(control.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))

    control._terminate_windows_tree(4321)

    assert calls[0][0] == ["taskkill", "/PID", "4321", "/T", "/F"]


def test_shutdown_route_returns_before_scheduling(client, monkeypatch):
    calls = []
    monkeypatch.setattr("backend.main.schedule_shutdown", lambda: calls.append("scheduled"))

    response = client.post("/api/shutdown")

    assert response.status_code == 200
    assert response.json() == {"shutting_down": True}
    assert calls == ["scheduled"]
