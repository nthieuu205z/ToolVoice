"""Gọi ffmpeg/ffprobe qua subprocess, gói lỗi lại cho dễ hiển thị."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .errors import FFmpegError, FFmpegNotFoundError

# Ghi đè bằng biến môi trường khi ffmpeg không nằm trong PATH.
_ENV_OVERRIDES: dict[str, str | None] = {"ffmpeg": None, "ffprobe": None}


def set_binaries(ffmpeg: str | None, ffprobe: str | None) -> None:
    """Backend gọi hàm này lúc khởi động với giá trị từ .env."""
    _ENV_OVERRIDES["ffmpeg"] = ffmpeg or None
    _ENV_OVERRIDES["ffprobe"] = ffprobe or None
    resolve.cache_clear()


@lru_cache(maxsize=4)
def resolve(name: str) -> str:
    """Tìm đường dẫn ffmpeg/ffprobe: ưu tiên .env, sau đó PATH, sau đó các vị trí brew quen thuộc."""
    override = _ENV_OVERRIDES.get(name)
    if override:
        if not Path(override).is_file():
            raise FFmpegNotFoundError(f"{name} không tồn tại tại đường dẫn đã cấu hình: {override}")
        return override

    found = shutil.which(name)
    if found:
        return found

    for candidate in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
        if Path(candidate).is_file():
            return candidate

    raise FFmpegNotFoundError(f"Không tìm thấy {name}")


def run(args: list[str], *, timeout: int = 3600) -> subprocess.CompletedProcess:
    """Chạy một lệnh ffmpeg/ffprobe đã dựng sẵn; ném FFmpegError kèm stderr khi thất bại."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg quá thời gian: {' '.join(args[:3])}") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise FFmpegError("ffmpeg lỗi:\n" + "\n".join(tail))
    return proc


def run_piped(args: list[str], data: bytes, *, timeout: int = 300) -> bytes:
    """Đẩy `data` vào stdin của ffmpeg và nhận kết quả từ stdout — không cần file tạm."""
    try:
        proc = subprocess.run(args, input=data, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError("ffmpeg quá thời gian khi giải mã âm thanh") from exc

    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"").decode("utf-8", "ignore").strip().splitlines()[-6:]
        raise FFmpegError("ffmpeg không giải mã được âm thanh:\n" + "\n".join(tail))
    return proc.stdout


def ffprobe_json(path: Path) -> dict:
    args = [
        resolve("ffprobe"),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    proc = run(args, timeout=120)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError("ffprobe trả về dữ liệu không đọc được") from exc
