"""Ứng dụng FastAPI: phục vụ giao diện tĩnh và các API của pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.job_manager import manager
from backend.process_control import schedule_shutdown
from backend.routes import jobs, model, settings as settings_routes, voices
from pipeline import custom_voices
from pipeline.ffmpeg_utils import set_binaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Nằm trong lifespan chứ KHÔNG ở cấp import: test import app hàng chục lần,
    # không được phép mỗi lần lại nạp thật 609 MB engine vào RAM.
    clone = settings.resolved_clone_provider
    # VieNeu nạp sẵn khi là giọng đọc DỰNG SẴN hoặc là engine NHÂN BẢN — kể cả khi
    # CLONE_TTS_PROVIDER=omnivoice tự lùi về vieneu vì chưa cài omnivoice. Không thì job
    # nhân bản đầu phải chờ nạp lạnh ~15–90s.
    if settings.tts_provider == "vieneu" or clone == "vieneu":
        from pipeline import vieneu_speech

        vieneu_speech.configure(batch_size=settings.vieneu_batch_size)
        # VieNeu nạp mất ~15–90 giây lần lạnh — nạp nền từ lúc boot để job đầu khỏi chờ.
        vieneu_speech.prewarm()
    if settings.stt_provider == "whisper":
        from pipeline import whisper_stt

        # Whisper nạp mất ~11s lần lạnh — nạp nền song song để bước "quét" job đầu khỏi chờ.
        whisper_stt.prewarm(settings.whisper_model, settings.whisper_compute_type)
    if clone == "omnivoice":
        from pipeline import omnivoice_speech

        # OmniVoice nạp mất ~30s lần lạnh — nạp nền để job giọng nhân bản đầu khỏi chờ.
        omnivoice_speech.prewarm()
    yield


app = FastAPI(title="ToolVietSub", docs_url=None, redoc_url=None, lifespan=_lifespan)

set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)
# Giọng nhân bản lưu trong thư mục riêng, sống qua khởi động lại server.
custom_voices.configure(settings.custom_voices_dir)
# Nạp lại danh sách job từ đĩa — đóng trình duyệt hay khởi động lại server không làm mất lịch sử.
manager.restore(settings.jobs_dir)

app.include_router(voices.router)
app.include_router(model.router)
app.include_router(jobs.router)
app.include_router(settings_routes.router)


@app.post("/api/shutdown")
def shutdown_tool() -> dict:
    schedule_shutdown()
    return {"shutting_down": True}

# Mount sau cùng để các route /api/* được khớp trước.
app.mount("/previews", StaticFiles(directory=settings.previews_dir, check_dir=False), name="previews")
app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="static")
