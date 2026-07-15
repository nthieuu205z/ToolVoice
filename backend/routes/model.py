"""Tải các model chạy trên máy (Whisper, VieNeu) thủ công và theo dõi tiến trình."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend import model_manager
from backend.config import settings
from backend.job_manager import manager
from pipeline.model_store import ModelSpec

router = APIRouter()

_POLL_SECONDS = 0.5


def _spec(key: str) -> ModelSpec:
    for spec in settings.model_specs:
        if spec.key == key:
            return spec
    raise HTTPException(404, f"Không có model nào tên '{key}' với cấu hình hiện tại.")


def _all() -> list[dict]:
    return [model_manager.downloader.snapshot(s) for s in settings.model_specs]


@router.get("/api/model")
def model_status() -> dict:
    """Danh sách model cần có trên máy. Rỗng khi mọi bước đều chạy trên đám mây."""
    models = _all()
    return {
        "required": bool(models),
        "ready": all(m["ready"] for m in models),
        "models": models,
    }


@router.post("/api/model/download")
def start_download(key: str) -> dict:
    if manager.is_busy():
        raise HTTPException(409, "Đang xử lý một video. Đợi xong rồi hãy tải model.")

    spec = _spec(key)
    try:
        model_manager.downloader.start(spec)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc

    return {"started": True, "key": spec.key}


@router.get("/api/model/events")
async def model_events() -> StreamingResponse:
    """SSE bằng cách theo dõi dung lượng trên đĩa — đóng khi không còn model nào đang tải."""
    if not settings.model_specs:
        raise HTTPException(400, "Không có model nào để theo dõi.")

    async def stream():
        last: list[dict] | None = None
        while True:
            models = _all()
            if models != last:
                yield f"data: {json.dumps({'models': models}, ensure_ascii=False)}\n\n"
                last = models
            if not any(m["status"] == "downloading" for m in models):
                break
            await asyncio.sleep(_POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
