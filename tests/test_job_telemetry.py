import json
from pathlib import Path

import pytest

from backend import job_manager as jm
from backend.job_manager import Job, JobManager
from pipeline.errors import JobCancelledError


def test_running_snapshot_exposes_live_timing_runtime_and_eta():
    job = Job(
        id="abc",
        filename="clip.mp4",
        workdir=Path("jobs/abc"),
        voice_id="voice",
        status="running",
        stage="synthesize",
        percent=50.0,
        created_at=90.0,
        started_at=100.0,
        stage_started_at=112.0,
        engine="vieneu",
        device="cuda",
        batch_size=16,
    )

    snapshot = job.snapshot(now=130.0)

    assert snapshot["elapsed_seconds"] == 30.0
    assert snapshot["stage_elapsed_seconds"] == 18.0
    assert snapshot["eta_seconds"] == 30.0
    assert snapshot["engine"] == "vieneu"
    assert snapshot["device"] == "cuda"
    assert snapshot["batch_size"] == 16


def test_progress_is_monotonic_and_records_stage_events():
    job = Job(id="abc", filename="clip.mp4", workdir=Path("jobs/abc"), voice_id="voice")
    job.update_progress("transcribe", 1.0, "Xong nhận diện", now=10.0)
    job.update_progress("extract", 0.0, "Tín hiệu trễ", now=11.0)

    snapshot = job.snapshot(now=12.0)
    assert snapshot["percent"] == 30.0
    assert [item["stage"] for item in snapshot["stage_history"]] == [
        "transcribe",
        "extract",
    ]
    assert snapshot["events"][-1]["message"] == "Tín hiệu trễ"


@pytest.mark.parametrize(
    ("persisted_status", "terminal_status", "terminal_message"),
    [
        (
            "queued",
            "error",
            "Máy chủ đã dừng khi video này đang xử lý. Hãy chạy lại video.",
        ),
        (
            "running",
            "error",
            "Máy chủ đã dừng khi video này đang xử lý. Hãy chạy lại video.",
        ),
        ("cancelling", "cancelled", JobCancelledError.user_message),
    ],
)
def test_restore_terminalizes_interrupted_job_telemetry(
    tmp_path, monkeypatch, persisted_status, terminal_status, terminal_message
):
    workdir = tmp_path / persisted_status
    workdir.mkdir()
    metadata = {
        "job_id": persisted_status,
        "filename": "clip.mp4",
        "voice_id": "voice",
        "status": persisted_status,
        "stage": "synthesize",
        "percent": 60.0,
        "message": "Đang tạo giọng đọc",
        "created_at": 90.0,
        "started_at": 100.0,
        "stage_started_at": 150.0,
        "stage_fraction": 0.5,
        "stage_history": [
            {
                "stage": "synthesize",
                "started_at": 150.0,
                "ended_at": None,
                "duration_seconds": None,
            }
        ],
        "events": [],
    }
    (workdir / "job.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(jm.time, "time", lambda: 200.0)

    manager = JobManager(max_workers=1)
    assert manager.restore(tmp_path) == 1

    restored = manager.get(persisted_status)
    snapshot = restored.snapshot(now=300.0)
    assert snapshot["status"] == terminal_status
    assert snapshot["message"] == terminal_message
    assert snapshot["finished_at"] == 200.0
    assert snapshot["elapsed_seconds"] == 100.0
    assert snapshot["stage_elapsed_seconds"] == 50.0
    assert snapshot["stage_history"][-1]["ended_at"] == 200.0
    assert snapshot["events"][-1] == {
        "at": 200.0,
        "kind": "finished",
        "stage": "synthesize",
        "status": terminal_status,
        "message": terminal_message,
    }

    saved = json.loads((workdir / "job.json").read_text(encoding="utf-8"))
    assert saved["status"] == terminal_status
    assert saved["message"] == terminal_message
    assert saved["finished_at"] == 200.0
    assert saved["events"][-1] == snapshot["events"][-1]
