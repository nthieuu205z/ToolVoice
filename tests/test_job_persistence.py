"""Đóng trình duyệt hay khởi động lại server không được làm mất danh sách video.

Mỗi job ghi `job.json` vào thư mục của nó tại các mốc chuyển trạng thái; lúc server
khởi động, `restore()` nạp lại toàn bộ. Job đang chạy dở lúc server chết không thể
tiếp tục — phải bị đánh dấu lỗi, không được hiện "running" với một tiến trình ma.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

from backend import job_manager as jm
from backend.job_manager import Job, JobManager
from pipeline.errors import JobCancelledError
from pipeline.models import MediaInfo, PipelineResult
from pipeline.runner import PipelineOptions

MEDIA = MediaInfo(duration=10.0, video_codec="h264", has_audio=True)
OPTIONS = PipelineOptions(voice_id="Kore")


def _run_one(manager: JobManager, workdir: Path) -> Job:
    job = manager.start(filename="a.mp4", workdir=workdir, voice_id="Kore",
                        video_path=Path("a.mp4"), media=MEDIA,
                        backend_factory=lambda: None, options=OPTIONS)
    manager._futures[job.id].result(timeout=10)
    return job


def _write_meta(jobs_dir: Path, job_id: str, status: str, **extra) -> Path:
    workdir = jobs_dir / job_id
    workdir.mkdir(parents=True)
    data = {"job_id": job_id, "filename": "a.mp4", "status": status,
            "created_at": extra.pop("created_at", 100.0), **extra}
    (workdir / "job.json").write_text(json.dumps(data), encoding="utf-8")
    return workdir


def test_a_finished_job_survives_a_server_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(jm, "run_pipeline", lambda *a, **k: PipelineResult(
        video_path=str(tmp_path / "output.mp4"), srt_path=str(tmp_path / "output.srt"),
        attempted_count=3, spoken_count=3,
    ))
    workdir = tmp_path / "job1"
    workdir.mkdir()
    done = _run_one(JobManager(max_workers=1), workdir)

    reborn = JobManager()          # "server mới"
    assert reborn.restore(tmp_path) == 1

    restored = reborn.get(done.id)
    assert restored.status == "done"
    assert restored.video_path == str(tmp_path / "output.mp4")
    assert restored.spoken_count == 3


def test_a_job_that_died_mid_run_is_marked_as_error(tmp_path):
    _write_meta(tmp_path, "deadjob", "running", stage="synthesize", percent=60.0)

    manager = JobManager()
    manager.restore(tmp_path)

    job = manager.get("deadjob")
    assert job.status == "error"
    assert "Máy chủ đã dừng" in job.message
    # và trạng thái sửa lại phải được ghi xuống đĩa, để lần khởi động sau khỏi sửa lại nữa
    saved = json.loads((tmp_path / "deadjob" / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "error"


def test_a_job_stuck_cancelling_is_restored_as_cancelled(tmp_path):
    _write_meta(tmp_path, "cancelme", "cancelling")

    manager = JobManager()
    manager.restore(tmp_path)
    assert manager.get("cancelme").status == "cancelled"
    assert manager.get("cancelme").message == JobCancelledError.user_message


def test_a_corrupt_meta_file_does_not_break_the_others(tmp_path):
    _write_meta(tmp_path, "goodjob", "done")
    bad = tmp_path / "badjob"
    bad.mkdir()
    (bad / "job.json").write_text("{hỏng", encoding="utf-8")

    manager = JobManager()
    assert manager.restore(tmp_path) == 1
    assert manager.get("goodjob") is not None


def test_restored_jobs_keep_their_creation_order(tmp_path):
    _write_meta(tmp_path, "older", "done", created_at=100.0)
    _write_meta(tmp_path, "newer", "done", created_at=200.0)

    manager = JobManager()
    manager.restore(tmp_path)
    assert [j["job_id"] for j in manager.jobs()] == ["newer", "older"]


# ─── dọn thư mục ───

def test_prune_never_touches_an_active_job(tmp_path):
    """Job ĐANG CHẠY là thư mục CŨ NHẤT — nằm ngay giữa vùng bị dọn nếu thiếu chốt chặn."""
    import os

    manager = JobManager()
    # job0 đang chạy và cũ nhất; job1, job2 đã xong và mới hơn.
    for i, status in enumerate(["running", "done", "done"]):
        workdir = _write_meta(tmp_path, f"job{i}", status, created_at=float(i))
        os.utime(workdir, (i + 1, i + 1))
        manager._jobs[f"job{i}"] = Job(id=f"job{i}", filename="a", workdir=workdir,
                                       voice_id="v", status=status, created_at=float(i))

    manager.prune(tmp_path, keep=1)

    assert (tmp_path / "job0").is_dir()          # đang chạy — bất khả xâm phạm dù cũ nhất
    assert not (tmp_path / "job1").is_dir()      # đã xong, ngoài cửa sổ giữ — dọn
    assert manager.get("job1") is None           # mất file thì job cũng rời danh sách
    assert manager.get("job0") is not None


def test_prune_keeps_the_newest_finished_jobs(tmp_path):
    manager = JobManager()
    for i in range(4):
        workdir = _write_meta(tmp_path, f"job{i}", "done", created_at=float(i))
        import os
        os.utime(workdir, (i + 1, i + 1))
        manager._jobs[f"job{i}"] = Job(id=f"job{i}", filename="a", workdir=workdir,
                                       voice_id="v", status="done", created_at=float(i))

    manager.prune(tmp_path, keep=2)

    kept = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
    assert kept == ["job2", "job3"]


def test_prune_keeps_registry_state_when_directory_cleanup_fails(tmp_path, monkeypatch):
    manager = JobManager()
    workdir = _write_meta(tmp_path, "stale", "done")
    manager._jobs["stale"] = Job(
        id="stale", filename="a", workdir=workdir, voice_id="v", status="done"
    )

    def fail_cleanup(_path, *args, **kwargs):
        raise PermissionError("file is held open")

    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)

    manager.prune(tmp_path, keep=0)

    assert manager.get("stale") is not None
    assert workdir.is_dir()


def test_prune_cannot_remove_a_concurrently_replaced_registry_entry(tmp_path, monkeypatch):
    manager = JobManager()
    stale_dir = _write_meta(tmp_path, "same-id", "done")
    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    stale = Job(
        id="same-id", filename="old", workdir=stale_dir, voice_id="v", status="done"
    )
    replacement = Job(
        id="same-id", filename="new", workdir=replacement_dir, voice_id="v", status="running"
    )
    manager._jobs[stale.id] = stale
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    real_rmtree = shutil.rmtree

    def blocking_cleanup(path, *args, **kwargs):
        cleanup_started.set()
        assert allow_cleanup.wait(timeout=2)
        real_rmtree(path)

    monkeypatch.setattr(jm.shutil, "rmtree", blocking_cleanup)
    worker = threading.Thread(target=manager.prune, args=(tmp_path, 0))
    worker.start()
    assert cleanup_started.wait(timeout=2)
    with manager._lock:
        manager._jobs[replacement.id] = replacement
    allow_cleanup.set()
    worker.join(timeout=2)

    assert worker.is_alive() is False
    assert manager.get("same-id") is replacement
