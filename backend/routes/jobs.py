"""Vòng đời một công việc: nhận video → chạy pipeline → trả file kết quả."""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.config import settings
from backend.job_manager import Job, manager
from backend.secure_files import HeldFileResponse, OpenedOwnedFile, open_owned_file
from pipeline.backends import CompositeBackend, build_backend
from pipeline.errors import UnsupportedMediaError
from pipeline.probe import probe_video
from pipeline.runner import PipelineOptions
from pipeline.voices import is_available, route_provider

router = APIRouter()

_UPLOAD_CHUNK = 1024 * 1024
_SSE_POLL_SECONDS = 0.4
_SSE_HEARTBEAT_SECONDS = 15.0
# Giữ lại các job gần nhất để còn tải kết quả về; cũ hơn thì dọn cho nhẹ đĩa.
_KEEP_JOB_DIRS = 10


def _make_backend(tts_provider: str) -> CompositeBackend:
    """Engine giọng đọc CHỈ ĐỊNH cho job — provider đã được định tuyến theo giọng đã chọn."""
    return build_backend(settings.provider_config_for(tts_provider))


@router.post("/api/jobs")
async def create_job(video: UploadFile = File(...), voice_id: str = Form(...)) -> dict:
    # Bước dịch luôn cần Gemini, kể cả khi nhận diện và giọng đọc đã chạy miễn phí.
    gemini_api_key, _ = settings.gemini_runtime()
    if not gemini_api_key:
        raise HTTPException(500, "Chưa có GEMINI_API_KEY. Tạo file .env từ .env.example rồi điền khóa.")
    clone_provider = settings.resolved_clone_provider
    if not is_available(voice_id, settings.tts_provider, clone_provider):
        raise HTTPException(400, f"Giọng đọc không hợp lệ: {voice_id}")

    # Định tuyến một lần theo giọng: giọng nhân bản → engine clone; còn lại → tts_provider.
    effective_tts = route_provider(voice_id, settings.tts_provider, clone_provider)

    workdir = settings.jobs_dir / uuid.uuid4().hex[:12]
    with manager.reserve_workdir(workdir):
        manager.prune(settings.jobs_dir, keep=_KEEP_JOB_DIRS)
        workdir.mkdir(parents=True)

        video_path = workdir / f"input{Path(video.filename or 'video.mp4').suffix or '.mp4'}"
        await _save_upload(video, video_path)

        # ffprobe trước khi tiêu tốn bất kỳ token Gemini nào.
        try:
            media = probe_video(video_path)
        except UnsupportedMediaError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise HTTPException(400, exc.user_message) from exc

        options = PipelineOptions(
            voice_id=voice_id,
            max_utterance_seconds=settings.max_utterance_seconds,
            max_utterance_gap=settings.max_utterance_gap,
            sentence_level_timing=settings.sentence_level_timing,
            stt_workers=settings.stt_workers,
            tts_workers=settings.tts_workers,
            translate_workers=settings.translate_workers,
            tts_max_speedup=settings.tts_max_speedup,
            tts_fill_slowdown=settings.tts_fill_slowdown,
            tts_daily_budget=settings.tts_daily_budget,
            tts_is_metered=effective_tts == "gemini",
            # OmniVoice tự vá lỗ hổng im lặng (postprocess) nên KHÔNG chạy bước đọc-lại tốn kém
            # (mỗi lần là một single-synth ~5,8s, không gộp lô). VieNeu/edge vẫn cần.
            resynthesize_holes=effective_tts != "omnivoice",
        )
        job = manager.start(
            filename=video.filename or video_path.name,
            workdir=workdir,
            voice_id=voice_id,
            video_path=video_path,
            media=media,
            backend_factory=lambda: _make_backend(effective_tts),
            options=options,
        )
    return {"job_id": job.id}


@router.get("/api/jobs")
def list_jobs() -> dict:
    """Danh sách mọi job, mới nhất trước — giao diện vẽ thẳng từ đây."""
    return {"jobs": manager.jobs()}


async def _save_upload(video: UploadFile, dest: Path) -> None:
    """Ghi từng khối để video lớn không phải nằm hết trong RAM, và chặn file quá cỡ."""
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as out:
        while chunk := await video.read(_UPLOAD_CHUNK):
            written += len(chunk)
            if written > limit:
                out.close()
                shutil.rmtree(dest.parent, ignore_errors=True)
                raise HTTPException(413, f"Video vượt quá giới hạn {settings.max_upload_mb} MB.")
            out.write(chunk)
    if written == 0:
        raise HTTPException(400, "File tải lên rỗng.")


@router.get("/api/jobs/current")
def current_job() -> dict:
    job = manager.current
    return job.snapshot() if job else {}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _require(job_id).snapshot()


@router.get("/api/jobs/{job_id}/telemetry")
def job_telemetry(job_id: str) -> dict:
    payload = _require(job_id).snapshot()
    payload["telemetry_only"] = True
    return payload


@router.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Yêu cầu dừng. Pipeline dừng ở mốc an toàn gần nhất, không giết thread giữa chừng."""
    try:
        return manager.cancel(job_id).snapshot()
    except LookupError:
        raise HTTPException(404, "Không tìm thấy công việc này.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    try:
        return {"deleted": manager.delete(job_id)}
    except LookupError:
        raise HTTPException(404, "Không tìm thấy công việc này.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """SSE bằng cách theo dõi trạng thái job — không cần hàng đợi giữa thread và event loop."""
    _require(job_id)

    async def stream():
        last: dict | None = None
        idle = 0.0
        while True:
            job = manager.get(job_id)
            if job is None:
                break

            snapshot = job.snapshot()
            if snapshot != last:
                yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n"
                last = snapshot
                idle = 0.0
            elif idle >= _SSE_HEARTBEAT_SECONDS:
                yield ": keepalive\n\n"  # giữ kết nối qua proxy/trình duyệt
                idle = 0.0

            if snapshot["status"] in ("done", "error", "cancelled"):
                break

            await asyncio.sleep(_SSE_POLL_SECONDS)
            idle += _SSE_POLL_SECONDS

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/jobs/{job_id}/download/video")
def download_video(job_id: str) -> HeldFileResponse:
    job = _require_done(job_id)
    opened = _job_file(job, job.video_path)
    stem = Path(job.filename).stem
    return _serve(opened, f"{stem}_vi{opened.path.suffix}")


@router.get("/api/jobs/{job_id}/download/srt")
def download_srt(job_id: str) -> HeldFileResponse:
    job = _require_done(job_id)
    return _serve(_job_file(job, job.srt_path), f"{Path(job.filename).stem}_vi.srt")


def _job_file(job: Job, value: str) -> OpenedOwnedFile:
    """Mở và giữ file kết quả trực tiếp trong thư mục của chính job."""
    return open_owned_file(job.workdir, value, "Không tìm thấy file kết quả.")


def _serve(opened: OpenedOwnedFile, download_name: str) -> HeldFileResponse:
    return HeldFileResponse(opened, filename=download_name)


def _require(job_id: str) -> Job:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Không tìm thấy công việc này.")
    return job


def _require_done(job_id: str) -> Job:
    job = _require(job_id)
    if job.status != "done":
        raise HTTPException(409, "Công việc chưa hoàn tất.")
    return job
