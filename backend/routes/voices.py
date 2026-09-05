"""Danh sách giọng đọc + tạo/xóa giọng nhân bản từ audio mẫu (chỉ VieNeu)."""

from __future__ import annotations

import logging
import threading
import wave
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.secure_files import HeldFileResponse, atomic_owned_file, owned_file_response
from pipeline import custom_voices
from pipeline.audio import decode_to_pcm, write_wav
from pipeline.errors import FFmpegError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.voices import available_voices

log = logging.getLogger(__name__)
router = APIRouter()
_preview_locks_guard = threading.Lock()
_preview_locks: dict[tuple[str, str], threading.RLock] = {}
_preview_states: dict[tuple[str, str], str] = {}


def _preview_key(voice_id: str) -> tuple[str, str]:
    return str(settings.previews_dir.absolute()), voice_id


def _preview_lock(voice_id: str) -> threading.RLock:
    key = _preview_key(voice_id)
    with _preview_locks_guard:
        return _preview_locks.setdefault(key, threading.RLock())


def _set_preview_state(voice_id: str, status: str | None) -> None:
    key = _preview_key(voice_id)
    with _preview_locks_guard:
        if status is None:
            _preview_states.pop(key, None)
        else:
            _preview_states[key] = status


def _preview_root() -> Path:
    root = settings.previews_dir.absolute()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _preview_path(voice_id: str) -> Path:
    """Trả về đường dẫn nghe thử đã xác thực cho một giọng đã đăng ký."""
    registered_custom = custom_voices.is_custom(voice_id) and custom_voices.get(voice_id) is not None
    catalogued = any(
        voice.id == voice_id
        for voice in available_voices(settings.tts_provider, settings.resolved_clone_provider)
    )
    if not registered_custom and not catalogued:
        raise HTTPException(404, "Không tìm thấy giọng đọc này.")

    try:
        # Keep this lexical: the held-open layer must see and reject any symlink/reparse point.
        root = settings.previews_dir.absolute()
        path = root / f"{voice_id}.wav"
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(404, "Không tìm thấy giọng đọc này.") from None
    if path.parent != root:
        raise HTTPException(404, "Không tìm thấy giọng đọc này.")
    return path


def _preview_status_value(voice_id: str) -> str:
    path = _preview_path(voice_id)
    with _preview_locks_guard:
        tracked = _preview_states.get(_preview_key(voice_id))
    if tracked in {"generating", "failed"}:
        return tracked
    if path.is_file():
        return "ready"
    return tracked or "missing"


def _clone_synthesizer():
    """Engine đọc giọng nhân bản đang cấu hình (OmniVoice hoặc VieNeu)."""
    if settings.resolved_clone_provider == "omnivoice":
        from pipeline.omnivoice_speech import OmniVoiceSynthesizer

        return OmniVoiceSynthesizer(whisper_model=settings.whisper_model,
                                    whisper_compute_type=settings.whisper_compute_type)
    from pipeline.vieneu_speech import VieNeuSynthesizer

    return VieNeuSynthesizer(watermark=settings.vieneu_watermark)

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
    for voice in available_voices(settings.tts_provider, settings.resolved_clone_provider):
        preview_status = _preview_status_value(voice.id)
        result.append({
            "id": voice.id,
            "display_name": voice.display_name,
            "preview_url": (
                f"/api/voices/{voice.id}/preview" if preview_status == "ready" else ""
            ),
            "preview_status": preview_status,
            "custom": custom_voices.is_custom(voice.id),
        })
    return result


@router.get("/api/voices/{voice_id}/preview")
def preview_voice(voice_id: str) -> HeldFileResponse:
    path = _preview_path(voice_id)
    if custom_voices.is_custom(voice_id) and _preview_status_value(voice_id) == "generating":
        raise HTTPException(409, "File demo đang được tạo.")
    return owned_file_response(
        settings.previews_dir,
        path,
        media_type="audio/wav",
        filename=f"{voice_id}.wav",
        not_found_detail="Giọng này chưa có file demo.",
    )


@router.get("/api/voices/{voice_id}/preview-status")
def preview_status(voice_id: str) -> dict:
    return {"status": _preview_status_value(voice_id)}


@router.get("/api/voices/cloning")
def cloning_status() -> dict:
    """Nhân bản bật khi có engine clone (OmniVoice hoặc VieNeu) — không phụ thuộc giọng dựng sẵn."""
    return {"enabled": settings.resolved_clone_provider is not None}


@router.post("/api/voices/custom")
async def create_custom_voice(name: str = Form(...), audio: UploadFile = File(...)) -> dict:
    if settings.resolved_clone_provider is None:
        raise HTTPException(400, "Nhân bản giọng đang tắt (CLONE_TTS_PROVIDER=none trong .env).")

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
    _set_preview_state(voice_id, "generating")

    # Nghe thử tạo ở nền: lần đầu phải nạp engine (~90 giây), không bắt request đợi.
    threading.Thread(target=_generate_preview, args=(voice_id,), daemon=True,
                     name=f"preview-{voice_id}").start()

    return {"id": voice.id, "display_name": f"{voice.display_name} — giọng nhân bản",
            "preview_url": "", "preview_status": "generating",
            "custom": True}


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

    with _preview_lock(voice_id):
        if custom_voices.get(voice_id) is None:
            _set_preview_state(voice_id, None)
            return
        _set_preview_state(voice_id, "generating")
        try:
            root = _preview_root()
            pcm = _clone_synthesizer().synthesize(_PREVIEW_TEXT, voice_id)
            preview = _preview_path(voice_id)
            samples = pcm_to_array(pcm)
            with atomic_owned_file(root, preview) as file:
                with wave.open(file, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(TTS_SAMPLE_RATE)
                    wav.writeframes(samples.astype("<i2").tobytes())
            _set_preview_state(voice_id, "ready")
            log.info("Đã tạo file nghe thử cho giọng nhân bản %s", voice_id)
        except Exception as exc:  # thiếu nghe thử không phải lỗi chết người
            _set_preview_state(voice_id, "failed")
            log.warning("Không tạo được nghe thử cho %s: %s", voice_id, exc)


@router.delete("/api/voices/custom/{voice_id}")
def delete_custom_voice(voice_id: str) -> dict:
    with _preview_lock(voice_id):
        if not custom_voices.is_custom(voice_id) or custom_voices.get(voice_id) is None:
            raise HTTPException(404, "Không tìm thấy giọng nhân bản này.")

        preview = _preview_path(voice_id)
        try:
            preview.unlink(missing_ok=True)
        except OSError:
            raise HTTPException(
                409,
                "File demo đang được sử dụng. Hãy dừng phát rồi thử xóa lại.",
            ) from None
        custom_voices.remove(voice_id)
        _set_preview_state(voice_id, None)

    # Bảo mọi engine clone quên giọng: VieNeu giữ embedding trong RAM, OmniVoice giữ ref_text
    # trong file sidecar. Gọi cả hai để id không trỏ vào dữ liệu cũ dù đang cấu hình engine nào.
    from pipeline.omnivoice_speech import forget_clone as omnivoice_forget
    from pipeline.vieneu_speech import forget_clone as vieneu_forget

    vieneu_forget(voice_id)
    omnivoice_forget(voice_id)
    return {"deleted": voice_id}
