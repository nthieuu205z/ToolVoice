"""Tách âm thanh từ video ra WAV mono cho bước nhận diện giọng nói."""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import resolve, run
from .models import STT_SAMPLE_RATE


def extract_audio(video_path: Path, out_wav: Path, rate: int = STT_SAMPLE_RATE) -> Path:
    """Xuất luồng âm thanh đầu tiên thành WAV mono 16-bit."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        resolve("ffmpeg"), "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    return out_wav


def slice_audio(src_wav: Path, out_wav: Path, start: float, duration: float) -> Path:
    """Cắt một khoảng của WAV ra file riêng (dùng khi video dài phải chia khúc)."""
    run([
        resolve("ffmpeg"), "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(src_wav),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ])
    return out_wav
