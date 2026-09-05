"""Trạng thái job phải nhất quán tại mọi thời điểm giao diện có thể đọc được nó."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from backend import job_manager as jm
from backend.job_manager import Job, JobManager
from pipeline.errors import NoSpeechDetectedError
from pipeline.models import MediaInfo, PipelineResult
from pipeline.runner import PipelineOptions

MEDIA = MediaInfo(duration=10.0, video_codec="h264", has_audio=True)
OPTIONS = PipelineOptions(voice_id="Kore")


def _start(manager: JobManager, tmp_path) -> Job:
    return manager.start(filename="a.mp4", workdir=tmp_path, voice_id="Kore",
                         video_path=Path("a.mp4"), media=MEDIA,
                         backend_factory=lambda: None, options=OPTIONS)


def _wait(manager: JobManager, job: Job) -> Job:
    manager._futures[job.id].result(timeout=10)
    return job


class _SpyJob(Job):
    """Chụp lại trạng thái tại đúng khoảnh khắc `status` được gán."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "status" and value in ("done", "error"):
            object.__setattr__(self, "captured", self.snapshot())


def test_result_fields_are_visible_the_moment_status_turns_done(tmp_path, monkeypatch):
    """SSE dừng đọc ngay khi thấy "done" — lật status trước khi gán warnings là mất cảnh báo."""
    monkeypatch.setattr(jm, "run_pipeline", lambda *a, **k: PipelineResult(
        video_path="v.mp4", srt_path="v.srt",
        attempted_count=10, spoken_count=3, warnings=["thiếu giọng đọc"],
    ))
    monkeypatch.setattr(jm, "Job", _SpyJob)

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path))

    # Ảnh chụp tại lúc status vừa thành "done", không phải sau khi mọi thứ đã lắng.
    assert job.captured["status"] == "done"
    assert job.captured["warnings"] == ["thiếu giọng đọc"]
    assert job.captured["spoken_count"] == 3
    assert job.captured["degraded"] is True
    assert job.video_path == "v.mp4"


def test_pipeline_error_becomes_a_vietnamese_message(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise NoSpeechDetectedError()

    monkeypatch.setattr(jm, "run_pipeline", boom)
    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path))

    assert job.status == "error"
    assert job.message == NoSpeechDetectedError.user_message


def test_unexpected_crash_still_surfaces_to_the_ui(tmp_path, monkeypatch):
    monkeypatch.setattr(jm, "run_pipeline", lambda *a, **k: 1 / 0)
    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path))

    assert job.status == "error"
    assert "Lỗi không lường trước" in job.message


def test_message_is_set_before_status_flips_to_error(tmp_path, monkeypatch):
    """Cùng lý do với "done": UI đọc message ngay khi thấy status error."""
    monkeypatch.setattr(jm, "run_pipeline", lambda *a, **k: (_ for _ in ()).throw(NoSpeechDetectedError()))
    monkeypatch.setattr(jm, "Job", _SpyJob)

    manager = JobManager(max_workers=1)
    job = _wait(manager, _start(manager, tmp_path))

    assert job.captured["status"] == "error"
    assert job.captured["message"] == NoSpeechDetectedError.user_message


def test_two_jobs_run_at_the_same_time(tmp_path, monkeypatch):
    """Đây là toàn bộ mục đích của đa job: video thứ hai không phải chờ video thứ nhất."""
    both_running = threading.Barrier(3, timeout=10)

    def pipeline(backend, video, workdir, options, progress, media, should_cancel):
        both_running.wait()   # chỉ vượt qua được khi CẢ HAI job cùng đứng ở đây
        return PipelineResult(video_path="v.mp4", srt_path="v.srt")

    monkeypatch.setattr(jm, "run_pipeline", pipeline)
    manager = JobManager(max_workers=2)

    first = _start(manager, tmp_path)
    second = _start(manager, tmp_path)
    both_running.wait()   # bên test là bên thứ ba của barrier

    _wait(manager, first)
    _wait(manager, second)
    assert first.status == "done" and second.status == "done"


def test_a_third_job_queues_behind_the_concurrency_cap(tmp_path, monkeypatch):
    release = threading.Event()

    def pipeline(backend, video, workdir, options, progress, media, should_cancel):
        release.wait(10)
        return PipelineResult(video_path="v.mp4", srt_path="v.srt")

    monkeypatch.setattr(jm, "run_pipeline", pipeline)
    manager = JobManager(max_workers=2)

    jobs = [_start(manager, tmp_path) for _ in range(3)]
    for _ in range(500):
        if sum(1 for j in jobs if j.status == "running") == 2:
            break
        threading.Event().wait(0.01)

    assert sum(1 for j in jobs if j.status == "running") == 2
    assert sum(1 for j in jobs if j.status == "queued") == 1

    release.set()
    for job in jobs:
        _wait(manager, job)
    assert all(j.status == "done" for j in jobs)


def test_delete_removes_terminal_job_and_its_workdir(tmp_path):
    manager = JobManager(max_workers=1)
    workdir = tmp_path / "done"
    workdir.mkdir()
    (workdir / "output.mp4").write_bytes(b"video")
    job = Job(id="done", filename="clip.mp4", workdir=workdir, voice_id="voice", status="done")
    manager._jobs[job.id] = job

    assert manager.delete(job.id) == "done"
    assert manager.get(job.id) is None
    assert not workdir.exists()


def test_delete_rejects_active_job(tmp_path):
    manager = JobManager(max_workers=1)
    job = Job(id="live", filename="clip.mp4", workdir=tmp_path, voice_id="voice", status="running")
    manager._jobs[job.id] = job

    with pytest.raises(RuntimeError, match="đang chạy"):
        manager.delete(job.id)
