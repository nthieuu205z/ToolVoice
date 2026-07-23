"""Tải model chạy trên máy (Whisper, VieNeu) thủ công, có tiến trình.

Không có màn này thì job đầu tiên đứng im vài phút để tải ngầm, không ai biết vì sao.
Test không chạm mạng và không tải gì.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import model_manager
from backend.config import settings
from backend.job_manager import Job, manager
from backend.main import app
from backend.model_manager import ModelDownloader
from pipeline.model_store import ModelInfo, ModelSpec, RepoSpec

WHISPER = ModelSpec(key="whisper", label="Whisper 'small'", repos=[RepoSpec("Systran/faster-whisper-small")])
VIENEU = ModelSpec(key="vieneu", label="VieNeu-TTS", repos=[RepoSpec("pnnbao-ump/VieNeu-TTS-v3-Turbo")])


@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()
    manager._futures.clear()
    model_manager.downloader = ModelDownloader()


def describe_as(states: dict[str, tuple[bool, int, int]]):
    """key → (ready, downloaded_bytes, total_bytes)."""
    def _describe(spec: ModelSpec) -> ModelInfo:
        ready, down, total = states[spec.key]
        return ModelInfo(key=spec.key, label=spec.label, ready=ready,
                         downloaded_bytes=down, total_bytes=total)
    return _describe


@pytest.fixture
def whisper_only(monkeypatch):
    monkeypatch.setattr(type(settings), "model_specs", property(lambda self: [WHISPER]))


@pytest.fixture
def both_models(monkeypatch):
    monkeypatch.setattr(type(settings), "model_specs", property(lambda self: [WHISPER, VIENEU]))


# ─── báo cáo trạng thái ───

def test_a_missing_model_is_reported_with_its_download_size(client, whisper_only, monkeypatch):
    monkeypatch.setattr("pipeline.model_store.describe", describe_as({"whisper": (False, 0, 486_200_000)}))
    body = client.get("/api/model").json()

    assert body["required"] is True
    assert body["ready"] is False
    m = body["models"][0]
    assert m["key"] == "whisper" and m["status"] == "idle"
    assert m["total_mb"] == 486.2 and m["percent"] == 0.0


def test_a_cached_model_is_reported_ready(client, whisper_only, monkeypatch):
    monkeypatch.setattr("pipeline.model_store.describe", describe_as({"whisper": (True, 486_200_000, 486_200_000)}))
    body = client.get("/api/model").json()
    assert body["ready"] is True
    assert body["models"][0]["status"] == "ready"
    assert body["models"][0]["percent"] == 100.0


def test_percent_comes_from_bytes_actually_on_disk(client, whisper_only, monkeypatch):
    monkeypatch.setattr("pipeline.model_store.describe", describe_as({"whisper": (False, 243_100_000, 486_200_000)}))
    assert client.get("/api/model").json()["models"][0]["percent"] == 50.0


def test_unknown_total_never_produces_a_bogus_percent(client, whisper_only, monkeypatch):
    """Hỏi Hugging Face hỏng thì total = 0 — chia cho 0 là hỏng cả trang."""
    monkeypatch.setattr("pipeline.model_store.describe", describe_as({"whisper": (False, 1000, 0)}))
    assert client.get("/api/model").json()["models"][0]["percent"] == 0.0


def test_an_invalid_model_name_is_an_error_not_a_crash(client, whisper_only, monkeypatch):
    def boom(spec):
        raise ValueError("Model Whisper không hợp lệ: xyz")

    monkeypatch.setattr("pipeline.model_store.describe", boom)
    m = client.get("/api/model").json()["models"][0]
    assert m["status"] == "error" and "không hợp lệ" in m["message"]


# ─── nhiều model cùng lúc ───

def test_both_models_are_listed_when_vieneu_is_the_tts_provider(client, both_models, monkeypatch):
    monkeypatch.setattr("pipeline.model_store.describe",
                        describe_as({"whisper": (True, 100, 100), "vieneu": (False, 0, 609_000_000)}))
    body = client.get("/api/model").json()

    assert [m["key"] for m in body["models"]] == ["whisper", "vieneu"]
    assert body["ready"] is False  # một cái chưa sẵn sàng thì cả bộ chưa sẵn sàng


def test_downloading_one_model_does_not_mark_the_other_as_downloading(client, both_models, monkeypatch):
    """Chỉ model đang được tải mới hiện thanh tiến trình."""
    monkeypatch.setattr("pipeline.model_store.describe",
                        describe_as({"whisper": (False, 0, 100), "vieneu": (False, 0, 100)}))
    monkeypatch.setattr("pipeline.model_store.download", lambda spec: None)

    model_manager.downloader._active_key = "vieneu"
    model_manager.downloader._status = "downloading"

    models = {m["key"]: m for m in client.get("/api/model").json()["models"]}
    assert models["vieneu"]["status"] == "downloading"
    assert models["whisper"]["status"] == "idle"


def test_no_local_models_when_everything_runs_in_the_cloud(client, monkeypatch):
    monkeypatch.setattr(type(settings), "model_specs", property(lambda self: []))
    body = client.get("/api/model").json()
    assert body == {"required": False, "ready": True, "models": []}


def _clone_resolves_to(monkeypatch, value):
    """Ép engine nhân bản đã-tính, để test khỏi phụ thuộc gói omnivoice có cài trên máy hay không."""
    monkeypatch.setattr(type(settings), "resolved_clone_provider", property(lambda self: value))


def test_omnivoice_model_listed_when_clone_provider_is_omnivoice(monkeypatch):
    monkeypatch.setattr(settings, "stt_provider", "gemini")   # bỏ spec whisper
    monkeypatch.setattr(settings, "tts_provider", "edge")     # bỏ spec vieneu
    _clone_resolves_to(monkeypatch, "omnivoice")              # coi như omnivoice đã cài
    assert [s.key for s in settings.model_specs] == ["omnivoice"]


def test_vieneu_model_listed_when_clone_falls_back_to_vieneu(monkeypatch):
    """CLONE_TTS_PROVIDER=omnivoice nhưng CHƯA cài gói → lùi về vieneu (cấu hình MẶC ĐỊNH khi
    người dùng chưa cài omnivoice). VieNeu vẫn phải được liệt kê để tải trước, nếu không job
    nhân bản đầu tải ngầm ~610 MB giữa chừng, không có thanh tiến trình."""
    monkeypatch.setattr(settings, "stt_provider", "gemini")   # bỏ spec whisper
    monkeypatch.setattr(settings, "tts_provider", "edge")     # preset KHÔNG phải vieneu
    _clone_resolves_to(monkeypatch, "vieneu")
    assert [s.key for s in settings.model_specs] == ["vieneu"]


def test_omnivoice_model_absent_when_cloning_disabled(monkeypatch):
    monkeypatch.setattr(settings, "stt_provider", "gemini")
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(settings, "clone_tts_provider", "none")
    assert settings.model_specs == []


# ─── bắt đầu tải ───

def test_download_starts_the_requested_model(client, both_models, monkeypatch):
    started = []
    monkeypatch.setattr(model_manager.downloader, "start", lambda spec: started.append(spec.key))

    assert client.post("/api/model/download?key=vieneu").json() == {"started": True, "key": "vieneu"}
    assert started == ["vieneu"]


def test_downloading_an_unknown_model_is_a_404(client, whisper_only):
    assert client.post("/api/model/download?key=vieneu").status_code == 404


def test_a_second_download_is_refused_while_one_runs(client, whisper_only, monkeypatch):
    def busy(spec):
        raise RuntimeError("Đang tải một model rồi")

    monkeypatch.setattr(model_manager.downloader, "start", busy)
    assert client.post("/api/model/download?key=whisper").status_code == 409


def test_downloading_is_refused_while_a_video_is_being_processed(client, whisper_only, tmp_path):
    """Tải model giữa lúc chạy pipeline sẽ tranh băng thông và CPU."""
    manager._jobs["x"] = Job(id="x", filename="a.mp4", workdir=tmp_path, voice_id="v", status="running")
    assert client.post("/api/model/download?key=whisper").status_code == 409


# ─── vòng đời của bộ tải ───

def test_a_failed_download_surfaces_its_reason(monkeypatch):
    monkeypatch.setattr("pipeline.model_store.download",
                        lambda spec: (_ for _ in ()).throw(OSError("mất mạng")))
    monkeypatch.setattr("pipeline.model_store.describe", describe_as({"whisper": (False, 0, 100)}))

    d = ModelDownloader()
    d.start(WHISPER)
    d._thread.join(timeout=5)

    snap = d.snapshot(WHISPER)
    assert snap["status"] == "error" and "mất mạng" in snap["message"]


def test_a_finished_download_reports_ready(monkeypatch):
    monkeypatch.setattr("pipeline.model_store.download", lambda spec: None)
    monkeypatch.setattr("pipeline.model_store.describe", describe_as({"whisper": (True, 100, 100)}))

    d = ModelDownloader()
    d.start(WHISPER)
    d._thread.join(timeout=5)

    assert d.snapshot(WHISPER)["status"] == "ready"
    assert d.is_downloading() is False
