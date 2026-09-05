import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.job_manager import manager
from backend.main import app

_SECRET_MARKER = "task3-fix-round1-secret-marker"


@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()
    manager._futures.clear()


def test_gemini_status_is_masked(client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", f"{_SECRET_MARKER}-stored")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    response = client.get("/api/settings/gemini")

    assert response.status_code == 200
    assert response.json() == {"configured": True, "backend": "vertex"}
    assert _SECRET_MARKER not in response.text


def test_gemini_update_atomically_persists_without_echo(tmp_path, client, monkeypatch):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_SETTING=keep\n", encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    response = client.post(
        "/api/settings/gemini",
        json={"api_key": f"{_SECRET_MARKER}-with-#-hash", "backend": "vertex"},
    )

    assert response.json() == {"saved": True, "configured": True, "backend": "vertex"}
    assert _SECRET_MARKER not in response.text
    assert env_path.read_text(encoding="utf-8") == (
        f'OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\n'
        f'GEMINI_API_KEY="{_SECRET_MARKER}-with-#-hash"\n'
    )


@pytest.mark.parametrize(
    ("api_key", "backend"),
    [
        pytest.param("", "developer", id="empty-key"),
        pytest.param(f"{_SECRET_MARKER}\rINJECTED=yes", "developer", id="cr-injection"),
        pytest.param(f"{_SECRET_MARKER}\nINJECTED=yes", "developer", id="lf-injection"),
        pytest.param(_SECRET_MARKER, "unsupported", id="invalid-backend"),
        pytest.param(f"{_SECRET_MARKER}{'x' * 512}", "developer", id="oversized-key"),
    ],
)
def test_gemini_rejection_is_sanitized_and_preserves_file_and_runtime(
    tmp_path, client, monkeypatch, api_key, backend
):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    original_content = (
        'OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\nGEMINI_API_KEY="existing-key"\n'
    )
    env_path.write_text(original_content, encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "existing-key")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    response = client.post(
        "/api/settings/gemini",
        json={"api_key": api_key, "backend": backend},
    )

    assert _SECRET_MARKER not in response.text
    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == original_content
    assert settings.gemini_api_key == "existing-key"
    assert settings.gemini_backend == "vertex"


def test_gemini_replace_failure_is_sanitized_and_preserves_file_and_runtime(
    tmp_path, monkeypatch
):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    original_content = (
        'OTHER_SETTING=keep\nGEMINI_BACKEND=developer\nGEMINI_API_KEY="existing-key"\n'
    )
    env_path.write_text(original_content, encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "existing-key")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    def fail_replace(source, destination):
        raise OSError(f"replace failed for {_SECRET_MARKER}")

    monkeypatch.setattr(settings_route.os, "replace", fail_replace)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/settings/gemini",
        json={"api_key": _SECRET_MARKER, "backend": "vertex"},
    )

    assert response.status_code == 500
    assert _SECRET_MARKER not in response.text
    assert env_path.read_text(encoding="utf-8") == original_content
    assert settings.gemini_api_key == "existing-key"
    assert settings.gemini_backend == "developer"


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {"api_key": [_SECRET_MARKER], "backend": "developer"},
            id="api-key-list",
        ),
        pytest.param(
            {"api_key": {"secret": _SECRET_MARKER}, "backend": "developer"},
            id="api-key-object",
        ),
        pytest.param({"api_key": 2712, "backend": "developer"}, id="api-key-number"),
        pytest.param({"api_key": None, "backend": "developer"}, id="api-key-null"),
        pytest.param({"api_key": True, "backend": "developer"}, id="api-key-boolean"),
        pytest.param(
            {"api_key": _SECRET_MARKER, "backend": [_SECRET_MARKER]},
            id="backend-list",
        ),
        pytest.param([_SECRET_MARKER], id="top-level-list"),
        pytest.param(_SECRET_MARKER, id="top-level-string"),
        pytest.param(2712, id="top-level-number"),
        pytest.param(None, id="top-level-null"),
    ],
)
def test_gemini_malformed_body_is_sanitized_and_preserves_file_and_runtime(
    tmp_path, client, monkeypatch, body
):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    original_content = (
        'OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\nGEMINI_API_KEY="existing-key"\n'
    )
    env_path.write_text(original_content, encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "existing-key")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    response = client.post("/api/settings/gemini", json=body)

    assert _SECRET_MARKER not in response.text
    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == original_content
    assert settings.gemini_api_key == "existing-key"
    assert settings.gemini_backend == "vertex"


def test_gemini_invalid_json_is_sanitized_and_preserves_file_and_runtime(
    tmp_path, client, monkeypatch
):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    original_content = (
        'OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\nGEMINI_API_KEY="existing-key"\n'
    )
    env_path.write_text(original_content, encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "existing-key")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    response = client.post(
        "/api/settings/gemini",
        content=f'{{"api_key":"{_SECRET_MARKER}"',
        headers={"Content-Type": "application/json"},
    )

    assert _SECRET_MARKER not in response.text
    assert response.status_code == 400
    assert env_path.read_text(encoding="utf-8") == original_content
    assert settings.gemini_api_key == "existing-key"
    assert settings.gemini_backend == "vertex"


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


def test_schedule_shutdown_starts_daemon_without_inline_termination(monkeypatch):
    import backend.process_control as control

    captured = {}
    calls = []

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            captured.update(target=target, name=name, daemon=daemon)

        def start(self):
            calls.append("started")

    monkeypatch.setattr(control.threading, "Thread", FakeThread)
    monkeypatch.setattr(control, "terminate_process_tree", lambda pid: calls.append(("terminate", pid)))

    control.schedule_shutdown(root_pid=2468, delay_seconds=0.2)

    assert captured["name"] == "tool-shutdown"
    assert captured["daemon"] is True
    assert calls == ["started"]


def test_shutdown_worker_delays_before_terminating_after_scheduler_returns(monkeypatch):
    import backend.process_control as control

    captured = {}
    calls = []
    scheduler_returned = False

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            captured["target"] = target

        def start(self):
            calls.append("started")

    monkeypatch.setattr(control.threading, "Thread", FakeThread)
    monkeypatch.setattr(control.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    monkeypatch.setattr(
        control,
        "terminate_process_tree",
        lambda pid: calls.append(("terminate", pid, scheduler_returned)),
    )

    control.schedule_shutdown(root_pid=2468, delay_seconds=0.2)
    scheduler_returned = True

    assert calls == ["started"]
    captured["target"]()
    assert calls == ["started", ("sleep", 0.2), ("terminate", 2468, True)]
