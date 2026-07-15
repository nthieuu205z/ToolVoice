"""Ghép video gốc với track lồng tiếng mới.

Đây là chỗ duy nhất quyết định "xóa hẳn audio gốc": lệnh chỉ map luồng video của
input 0 và luồng audio của input 1, nên audio gốc không bao giờ lọt vào output.
"""

from __future__ import annotations

from pathlib import Path

from .errors import FFmpegError
from .ffmpeg_utils import resolve, run


def build_mux_cmd(video: Path, audio: Path, out: Path, *, ffmpeg: str = "ffmpeg",
                  audio_bitrate: str = "192k") -> list[str]:
    """Dựng lệnh ffmpeg. Hàm thuần, tách riêng để test khẳng định được audio gốc bị loại."""
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-shortest",
        str(out),
    ]


def run_mux(video: Path, audio: Path, out: Path) -> Path:
    """Ghép không encode lại video. Nếu container .mp4 không chứa nổi codec gốc thì lùi về .mkv."""
    ffmpeg = resolve("ffmpeg")
    try:
        run(build_mux_cmd(video, audio, out, ffmpeg=ffmpeg))
        return out
    except FFmpegError:
        fallback = out.with_suffix(".mkv")
        if fallback == out:
            raise
        run(build_mux_cmd(video, audio, fallback, ffmpeg=ffmpeg))
        return fallback
