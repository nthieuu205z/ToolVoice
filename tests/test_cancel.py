"""Hủy là hợp tác: không giết thread giữa chừng (ffmpeg đang ghi file, ONNX đang chạy),
mà kiểm tra cờ ở các mốc an toàn. Test khoá lại rằng nó DỪNG THẬT, không chỉ đổi nhãn.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import job_manager as jm
from backend.job_manager import Job, JobManager
from backend.main import app
from backend.job_manager import manager
from pipeline.errors import JobCancelledError
from pipeline.models import MediaInfo, PipelineResult, Segment
from pipeline.runner import PipelineOptions
from pipeline.segmentation import Region
from pipeline.stt import transcribe_regions
from pipeline.translate import translate_segments
from pipeline.tts import synthesize_segments
from tests.conftest import FakeGemini

MEDIA = MediaInfo(duration=10.0, video_codec="h264", has_audio=True)
OPTIONS = PipelineOptions(voice_id="v")
RATE = 16000


# ─── các bước dài phải tự dừng ───

def test_transcription_stops_when_cancel_is_requested(tmp_path):
    backend = FakeGemini(clips=["a"])
    regions = [Region(i * 5.0, i * 5.0 + 4.0) for i in range(8)]
    samples = np.zeros(RATE * 60, dtype="<i2")

    with pytest.raises(JobCancelledError):
        transcribe_regions(backend, samples, RATE, regions, tmp_path,
                           workers=1, should_cancel=lambda: True)


def test_transcription_does_not_process_every_region_after_cancel(tmp_path):
    """Nếu chỉ đổi nhãn mà vẫn chạy hết thì hủy vô nghĩa."""
    backend = FakeGemini(clips=["a"])
    regions = [Region(i * 5.0, i * 5.0 + 4.0) for i in range(20)]
    samples = np.zeros(RATE * 120, dtype="<i2")

    with pytest.raises(JobCancelledError):
        transcribe_regions(backend, samples, RATE, regions, tmp_path,
                           workers=1, should_cancel=lambda: True)

    assert len(backend.transcribe_calls) < len(regions)


def test_synthesis_stops_when_cancel_is_requested():
    backend = FakeGemini(tts_duration=0.2)
    segments = [Segment(float(i), float(i) + 1.0, "src", f"câu {i}") for i in range(20)]

    with pytest.raises(JobCancelledError):
        synthesize_segments(backend, segments, "v", workers=1, should_cancel=lambda: True)

    assert len(backend.synthesize_calls) < len(segments)


def test_translation_stops_between_batches():
    from pipeline.translate import BATCH_SIZE

    backend = FakeGemini()
    segments = [Segment(float(i), float(i) + 1.0, f"line {i}") for i in range(BATCH_SIZE * 3)]

    with pytest.raises(JobCancelledError):
        translate_segments(backend, segments, should_cancel=lambda: True)

    assert backend.translate_calls == []  # dừng trước cả lô đầu tiên


def test_nothing_stops_when_cancel_is_never_requested(tmp_path):
    backend = FakeGemini(clips=["a", "b"])
    regions = [Region(0, 4), Region(5, 9)]
    _, segments, _ = transcribe_regions(backend, np.zeros(RATE * 10, dtype="<i2"), RATE,
                                        regions, tmp_path, workers=1)
    assert len(segments) == 2


# ─── vòng đời job ───

def _wait(m: JobManager, job: Job) -> Job:
    m._futures[job.id].result(timeout=10)
    return job


def _start(m: JobManager, tmp_path) -> Job:
    return m.start(filename="a.mp4", workdir=tmp_path, voice_id="v", video_path=Path("a.mp4"),
                   media=MEDIA, backend_factory=lambda: None, options=OPTIONS)


def test_cancelling_a_running_job_ends_it_as_cancelled(tmp_path, monkeypatch):
    started = threading.Event()

    def slow_pipeline(backend, video, workdir, options, progress, media, should_cancel):
        started.set()
        for _ in range(200):
            if should_cancel():
                raise JobCancelledError()
            threading.Event().wait(0.01)
        return PipelineResult(video_path="v.mp4", srt_path="v.srt")

    monkeypatch.setattr(jm, "run_pipeline", slow_pipeline)

    m = JobManager(max_workers=1)
    job = _start(m, tmp_path)
    assert started.wait(5)

    m.cancel(job.id)
    assert job.status == "cancelling"

    _wait(m, job)
    assert job.status == "cancelled"
    assert job.message == JobCancelledError.user_message


def test_the_slot_is_free_again_after_cancelling(tmp_path, monkeypatch):
    monkeypatch.setattr(jm, "run_pipeline",
                        lambda *a, **k: (_ for _ in ()).throw(JobCancelledError()))
    m = JobManager(max_workers=1)
    job = _start(m, tmp_path)
    _wait(m, job)

    assert m.is_busy() is False


def test_a_queued_job_cancels_instantly_without_waiting_its_turn(tmp_path, monkeypatch):
    """Job xếp hàng sau một video dài phải hủy được NGAY, không chờ 20 phút tới lượt."""
    release = threading.Event()

    def blocking_pipeline(backend, video, workdir, options, progress, media, should_cancel):
        release.wait(10)
        return PipelineResult(video_path="v.mp4", srt_path="v.srt")

    monkeypatch.setattr(jm, "run_pipeline", blocking_pipeline)

    m = JobManager(max_workers=1)
    first = _start(m, tmp_path)
    queued = _start(m, tmp_path)
    assert queued.status == "queued"

    m.cancel(queued.id)
    assert queued.status == "cancelled"  # ngay lập tức, không phải "cancelling"

    release.set()
    _wait(m, first)
    assert first.status == "done"


def test_a_job_cancelled_before_its_thread_runs_never_starts_the_pipeline(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(jm, "run_pipeline", lambda *a, **k: ran.append(1))

    m = JobManager()
    job = Job(id="x", filename="a.mp4", workdir=tmp_path, voice_id="v")
    job.cancel_event.set()
    m._jobs[job.id] = job
    m._run(job, Path("a.mp4"), MEDIA, lambda: None, OPTIONS)

    assert ran == []
    assert job.status == "cancelled"


def test_cancelling_a_finished_job_is_refused(tmp_path):
    m = JobManager()
    m._jobs["x"] = Job(id="x", filename="a", workdir=tmp_path, voice_id="v", status="done")
    with pytest.raises(RuntimeError):
        m.cancel("x")


def test_cancelling_an_unknown_job_raises_lookup(tmp_path):
    m = JobManager()
    m._jobs["x"] = Job(id="x", filename="a", workdir=tmp_path, voice_id="v", status="running")
    with pytest.raises(LookupError):
        m.cancel("khác")


def test_progress_does_not_overwrite_the_stopping_message(tmp_path, monkeypatch):
    """Trong lúc dừng, tiến trình cũ vẫn chạy nốt — đừng để nó xóa mất "Đang dừng…"."""
    progressed_after_cancel = threading.Event()
    released = threading.Event()

    def pipeline(backend, video, workdir, options, progress, media, should_cancel):
        progress("extract", 0.5, "Đang tách âm thanh")
        while not should_cancel():
            threading.Event().wait(0.01)
        progress("transcribe", 0.9, "Nhận diện đoạn 9/10")  # gọi sau khi đã bấm hủy
        progressed_after_cancel.set()
        released.wait(5)   # đứng yên cho test kiểm tra message, tránh race với _run
        raise JobCancelledError()

    monkeypatch.setattr(jm, "run_pipeline", pipeline)
    m = JobManager(max_workers=1)
    job = _start(m, tmp_path)

    for _ in range(500):
        if job.message == "Đang tách âm thanh":
            break
        threading.Event().wait(0.01)

    m.cancel(job.id)
    assert progressed_after_cancel.wait(5)   # progress() sau-hủy chắc chắn đã chạy
    assert job.message == "Đang dừng…"       # …mà không ghi đè được thông điệp dừng
    released.set()
    _wait(m, job)
    assert job.status == "cancelled"


# ─── route ───

@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()
    manager._futures.clear()


def test_cancel_route_returns_the_cancelling_snapshot(client, tmp_path):
    manager._jobs["abc"] = Job(id="abc", filename="a.mp4", workdir=tmp_path, voice_id="v", status="running")
    body = client.post("/api/jobs/abc/cancel").json()

    assert body["status"] == "cancelling"
    assert manager._jobs["abc"].cancel_requested is True


def test_cancelling_an_unknown_job_is_a_404(client):
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_cancelling_a_done_job_is_a_409(client, tmp_path):
    manager._jobs["abc"] = Job(id="abc", filename="a.mp4", workdir=tmp_path, voice_id="v", status="done")
    assert client.post("/api/jobs/abc/cancel").status_code == 409


def test_a_cancelling_job_still_holds_the_slot(client, tmp_path):
    """Nộp video mới khi job cũ chưa dừng hẳn sẽ giẫm lên thư mục đang dùng."""
    manager._jobs["abc"] = Job(id="abc", filename="a.mp4", workdir=tmp_path, voice_id="v", status="cancelling")
    assert manager.is_busy() is True


def test_a_cancelled_job_frees_the_slot(client, tmp_path):
    manager._jobs["abc"] = Job(id="abc", filename="a.mp4", workdir=tmp_path, voice_id="v", status="cancelled")
    assert manager.is_busy() is False
