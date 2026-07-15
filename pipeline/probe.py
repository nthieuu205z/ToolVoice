"""Kiểm tra file đầu vào trước khi tiêu tốn bất kỳ token Gemini nào."""

from __future__ import annotations

from pathlib import Path

from .errors import FFmpegError, UnsupportedMediaError
from .ffmpeg_utils import ffprobe_json
from .models import MediaInfo


def probe_video(path: Path) -> MediaInfo:
    """Đọc metadata video. Ném UnsupportedMediaError nếu không phải video dùng được."""
    try:
        data = ffprobe_json(path)
    except FFmpegError as exc:
        raise UnsupportedMediaError(str(exc)) from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise UnsupportedMediaError("File không chứa luồng video nào.")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not has_audio:
        raise UnsupportedMediaError(
            "",
            user_message="Video này không có âm thanh nên không thể lồng tiếng.",
        )

    duration = _duration(data, video)
    if duration <= 0:
        raise UnsupportedMediaError("Không đọc được thời lượng video.")

    return MediaInfo(
        duration=duration,
        video_codec=video.get("codec_name", ""),
        has_audio=has_audio,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
    )


def _duration(data: dict, video: dict) -> float:
    """Thời lượng lấy từ format trước; ảnh động/stream lạ mới phải rơi về luồng video."""
    for source in (data.get("format", {}), video):
        raw = source.get("duration")
        if raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
    return 0.0
