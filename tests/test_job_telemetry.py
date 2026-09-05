from pathlib import Path

from backend.job_manager import Job


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
