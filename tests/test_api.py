"""Kiểm tra các route mà giao diện gọi tới, không đụng ffmpeg lẫn Gemini."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.job_manager import Job, manager
from backend.main import app
from pipeline.voices import voices_for

VOICES = voices_for(settings.tts_provider)


def _put(job: Job) -> Job:
    manager._jobs[job.id] = job
    return job


def test_job_telemetry_route_returns_the_monitor_snapshot(client, tmp_path):
    _put(Job(
        id="telemetry-job",
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="voice",
        status="running",
        started_at=100.0,
        stage_started_at=110.0,
    ))

    response = client.get("/api/jobs/telemetry-job/telemetry")

    assert response.status_code == 200
    assert response.json()["job_id"] == "telemetry-job"
    assert response.json()["telemetry_only"] is True


@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()  # dọn sổ đăng ký giữa các test
    manager._futures.clear()


def test_voices_endpoint_shape_matches_frontend_expectations(client):
    body = client.get("/api/voices").json()
    # Tính trong thân test: danh sách giọng phụ thuộc kho giọng nhân bản mà fixture
    # autouse đã cách ly — VOICES ở cấp module được tính TRƯỚC khi fixture chạy.
    assert len(body) == len(voices_for(settings.tts_provider))
    assert set(body[0]) == {"id", "display_name", "preview_url", "custom"}


def test_voice_list_follows_the_configured_provider(client, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "gemini")
    assert len(client.get("/api/voices").json()) == 10

    monkeypatch.setattr(settings, "tts_provider", "edge")
    ids = [v["id"] for v in client.get("/api/voices").json()]
    assert ids[:2] == ["vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural"]
    assert len(ids) == 14


def test_edge_offers_native_and_multilingual_voices(client, monkeypatch):
    """12 giọng "Multilingual" đọc được tiếng Việt — đã kiểm chứng bằng Whisper nghe lại."""
    monkeypatch.setattr(settings, "tts_provider", "edge")
    ids = [v["id"] for v in client.get("/api/voices").json()]

    assert sum(1 for i in ids if i.startswith("vi-VN-")) == 2
    assert sum(1 for i in ids if "Multilingual" in i) == 12
    assert len(set(ids)) == len(ids)  # không trùng lặp


def test_a_voice_from_the_other_provider_is_rejected(client, monkeypatch):
    """Chọn giọng Gemini trong khi đang chạy edge-tts thì phải chặn, không để hỏng giữa chừng."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "tts_provider", "edge")

    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": "Charon"},
    )
    assert response.status_code == 400


def test_index_page_is_served_at_root(client):
    page = client.get("/").text
    assert 'id="uploadForm"' in page
    assert 'src="app.js?v=' in page


def test_current_job_is_empty_when_idle(client):
    assert client.get("/api/jobs/current").json() == {}


def test_unknown_job_returns_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


def test_rejects_unknown_voice(client, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": "KhongCoThat"},
    )
    assert response.status_code == 400


def test_rejects_upload_when_api_key_missing(client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    response = client.post(
        "/api/jobs",
        files={"video": ("a.mp4", b"data", "video/mp4")},
        data={"voice_id": VOICES[0].id},
    )
    assert response.status_code == 500
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_a_second_upload_is_accepted_while_a_job_runs(client, monkeypatch, tmp_path):
    """Nhiều video cùng lúc: video thứ hai vào danh sách chứ không bị 409 như trước."""
    import backend.job_manager as jm
    from pipeline.models import MediaInfo, PipelineResult

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    # KHÔNG để test ghi vào thư mục jobs/ thật — job ma sẽ được restore vào phiên chạy thật.
    monkeypatch.setattr(type(settings), "jobs_dir", property(lambda self: tmp_path))
    monkeypatch.setattr("backend.routes.jobs.probe_video",
                        lambda p: MediaInfo(duration=1.0, video_codec="h264", has_audio=True))
    monkeypatch.setattr(jm, "run_pipeline",
                        lambda *a, **k: PipelineResult(video_path="v.mp4", srt_path="v.srt"))
    _put(Job(id="busy", filename="a.mp4", workdir=tmp_path, voice_id="Kore", status="running"))

    response = client.post(
        "/api/jobs",
        files={"video": ("b.mp4", b"data", "video/mp4")},
        data={"voice_id": VOICES[0].id},
    )
    assert response.status_code == 200
    assert response.json()["job_id"]

    ids = [j["job_id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert "busy" in ids and len(ids) == 2


def test_the_job_list_is_newest_first(client, tmp_path):
    _put(Job(id="old", filename="a.mp4", workdir=tmp_path, voice_id="v",
             status="done", created_at=100.0))
    _put(Job(id="new", filename="b.mp4", workdir=tmp_path, voice_id="v",
             status="running", created_at=200.0))

    ids = [j["job_id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert ids == ["new", "old"]


def test_download_is_refused_until_the_job_finishes(client, tmp_path):
    _put(Job(id="abc", filename="a.mp4", workdir=tmp_path, voice_id="Kore", status="running"))
    assert client.get("/api/jobs/abc/download/video").status_code == 409


def test_download_serves_the_result_with_a_vietnamese_suffix(client, tmp_path):
    video = tmp_path / "output.mp4"
    video.write_bytes(b"fake")
    _put(Job(
        id="abc", filename="phim.mkv", workdir=tmp_path, voice_id="Kore",
        status="done", video_path=str(video), srt_path=str(video),
    ))
    response = client.get("/api/jobs/abc/download/video")
    assert response.status_code == 200
    assert "phim_vi.mp4" in response.headers["content-disposition"]


# ─── số liệu chất lượng mà giao diện dựa vào để cảnh báo ───

def _done_job(tmp_path, *, attempted, spoken, warnings=()):
    job = Job(id="q", filename="a.mp4", workdir=tmp_path, voice_id="Kore", status="done",
              attempted_count=attempted, spoken_count=spoken, warnings=list(warnings))
    manager._jobs[job.id] = job
    return job


def test_snapshot_reports_how_many_lines_were_actually_spoken(client, tmp_path):
    _done_job(tmp_path, attempted=100, spoken=23)
    body = client.get("/api/jobs/q").json()
    assert body["spoken_count"] == 23
    assert body["attempted_count"] == 100
    assert body["silent_ratio"] == 0.77


def test_mostly_silent_result_is_flagged_degraded(client, tmp_path):
    """77% câm là đúng thứ đã xảy ra thật — giao diện phải báo đỏ chứ không ăn mừng."""
    _done_job(tmp_path, attempted=252, spoken=58)
    assert client.get("/api/jobs/q").json()["degraded"] is True


def test_a_few_missing_lines_is_not_degraded(client, tmp_path):
    _done_job(tmp_path, attempted=100, spoken=95)
    assert client.get("/api/jobs/q").json()["degraded"] is False


def test_a_clean_run_is_never_degraded(client, tmp_path):
    _done_job(tmp_path, attempted=40, spoken=40)
    body = client.get("/api/jobs/q").json()
    assert body["degraded"] is False
    assert body["silent_ratio"] == 0.0


def test_silent_ratio_is_zero_when_nothing_was_attempted(client, tmp_path):
    _done_job(tmp_path, attempted=0, spoken=0)
    assert client.get("/api/jobs/q").json()["silent_ratio"] == 0.0


def test_a_running_job_is_never_degraded(client, tmp_path):
    job = _done_job(tmp_path, attempted=100, spoken=0)
    job.status = "running"
    manager._jobs[job.id] = job
    assert client.get("/api/jobs/q").json()["degraded"] is False


def test_warnings_reach_the_frontend(client, tmp_path):
    _done_job(tmp_path, attempted=10, spoken=10, warnings=["hết hạn mức"])
    assert client.get("/api/jobs/q").json()["warnings"] == ["hết hạn mức"]
