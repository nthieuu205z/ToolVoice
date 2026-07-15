"""Danh sách giọng đọc + tạo/xóa giọng nhân bản từ audio mẫu (chỉ VieNeu)."""

from __future__ import annotations

import logging
import threading

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from pipeline import custom_voices
from pipeline.audio import decode_to_pcm, write_wav
from pipeline.errors import FFmpegError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.voices import voices_for

log = logging.getLogger(__name__)
router = APIRouter()

# Audio mẫu: VieNeu tự cắt về ≤8 giây, nhưng nhận vào 12 giây cho thoải mái.
_SAMPLE_MAX_SECONDS = 12.0
_SAMPLE_MIN_SECONDS = 2.0
_SAMPLE_MAX_UPLOAD = 30 * 1024 * 1024

# Cùng câu với scripts/generate_voice_previews.py — file nghe thử phải nghe giống nhau.
_PREVIEW_TEXT = "Xin chào, đây là giọng đọc tiếng Việt dùng để lồng tiếng cho video của bạn."


@router.get("/api/voices")
def list_voices() -> list[dict]:
    """Kèm `preview_url` khi đã có file nghe thử; chưa có thì để rỗng, giao diện tự ẩn nút."""
    result = []
    for voice in voices_for(settings.tts_provider):
        preview = settings.previews_dir / f"{voice.id}.wav"
        result.append({
            "id": voice.id,
            "display_name": voice.display_name,
            "preview_url": f"/previews/{voice.id}.wav" if preview.is_file() else "",
            "custom": custom_voices.is_custom(voice.id),
        })
    return result


@router.get("/api/voices/cloning")
def cloning_status() -> dict:
    """Nhân bản giọng cần engine VieNeu — các provider khác không học được giọng mới."""
    return {"enabled": settings.tts_provider == "vieneu"}


@router.post("/api/voices/custom")
async def create_custom_voice(name: str = Form(...), audio: UploadFile = File(...)) -> dict:
    if settings.tts_provider != "vieneu":
        raise HTTPException(400, "Nhân bản giọng chỉ hoạt động khi TTS_PROVIDER=vieneu trong .env.")

    name = name.strip()
    if not 1 <= len(name) <= 40:
        raise HTTPException(400, "Tên giọng phải từ 1 đến 40 ký tự.")

    raw = await audio.read()
    if not raw:
        raise HTTPException(400, "File audio rỗng.")
    if len(raw) > _SAMPLE_MAX_UPLOAD:
        raise HTTPException(413, "File mẫu quá lớn — chỉ cần một đoạn 3–8 giây.")

    voice_id = custom_voices.unique_id(name)
    _normalize_sample(raw, voice_id)
    voice = custom_voices.register(voice_id, name)

    # Nghe thử tạo ở nền: lần đầu phải nạp engine (~90 giây), không bắt request đợi.
    threading.Thread(target=_generate_preview, args=(voice_id,), daemon=True,
                     name=f"preview-{voice_id}").start()

    return {"id": voice.id, "display_name": f"{voice.display_name} — giọng nhân bản",
            "preview_url": "", "custom": True}


def _normalize_sample(raw: bytes, voice_id: str) -> None:
    """Mọi định dạng audio → WAV mono 24 kHz tối đa 12 giây, đúng thứ engine cần."""
    try:
        pcm = decode_to_pcm(raw, TTS_SAMPLE_RATE)
    except FFmpegError as exc:
        raise HTTPException(400, "Không đọc được file — cần một file audio (wav, mp3, m4a…).") from exc

    duration = len(pcm) / 2 / TTS_SAMPLE_RATE
    if duration < _SAMPLE_MIN_SECONDS:
        raise HTTPException(400, f"Mẫu chỉ dài {duration:.1f} giây — cần ít nhất 3 giây lời nói.")

    keep = int(_SAMPLE_MAX_SECONDS * TTS_SAMPLE_RATE) * 2
    samples = np.frombuffer(pcm[:keep], dtype="<i2")
    write_wav(custom_voices.sample_path(voice_id), samples, TTS_SAMPLE_RATE)


def _generate_preview(voice_id: str) -> None:
    from pipeline.audio import pcm_to_array
    from pipeline.vieneu_speech import VieNeuSynthesizer

    try:
        pcm = VieNeuSynthesizer(watermark=settings.vieneu_watermark).synthesize(_PREVIEW_TEXT, voice_id)
        write_wav(settings.previews_dir / f"{voice_id}.wav", pcm_to_array(pcm))
        log.info("Đã tạo file nghe thử cho giọng nhân bản %s", voice_id)
    except Exception as exc:  # thiếu nghe thử không phải lỗi chết người
        log.warning("Không tạo được nghe thử cho %s: %s", voice_id, exc)


@router.delete("/api/voices/custom/{voice_id}")
def delete_custom_voice(voice_id: str) -> dict:
    if not custom_voices.is_custom(voice_id) or custom_voices.get(voice_id) is None:
        raise HTTPException(404, "Không tìm thấy giọng nhân bản này.")

    custom_voices.remove(voice_id)
    (settings.previews_dir / f"{voice_id}.wav").unlink(missing_ok=True)

    # Engine có thể còn giữ embedding trong RAM — bảo nó quên đi.
    from pipeline.vieneu_speech import forget_clone

    forget_clone(voice_id)
    return {"deleted": voice_id}
