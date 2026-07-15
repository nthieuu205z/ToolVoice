"""Giọng nhân bản từ audio mẫu của người dùng — chỉ có ý nghĩa với VieNeu.

Mỗi giọng là một cặp file trong thư mục cấu hình:

    <id>.wav   audio mẫu đã chuẩn hóa (mono 24 kHz, tối đa 12 giây)
    <id>.json  {"id", "display_name", "created_at"}

VieNeu học giọng từ file mẫu ngay lúc đăng ký (`add_voice`, vài giây) — không có
model nào phải huấn luyện, xóa file là giọng biến mất. Backend gọi `configure()`
lúc khởi động, giống mẫu `ffmpeg_utils.set_binaries`.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Tiền tố phân biệt giọng nhân bản với giọng dựng sẵn trong mọi voice_id.
PREFIX = "clone-"

_dir: Path | None = None


def configure(directory: Path) -> None:
    global _dir
    _dir = directory


def _store() -> Path:
    if _dir is None:
        raise RuntimeError("custom_voices chưa được configure()")
    _dir.mkdir(parents=True, exist_ok=True)
    return _dir


@dataclass(frozen=True)
class CustomVoice:
    id: str
    display_name: str
    created_at: float


def slugify(name: str) -> str:
    """Tên tiếng Việt có dấu → slug an toàn cho URL và tên file."""
    s = name.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s or "giong"


def is_custom(voice_id: str) -> bool:
    return voice_id.startswith(PREFIX)


def sample_path(voice_id: str) -> Path:
    """File audio mẫu mà engine sẽ học giọng từ đó."""
    return _store() / f"{voice_id}.wav"


def list_custom() -> list[CustomVoice]:
    if _dir is None or not _dir.is_dir():
        return []
    voices = []
    for meta in _dir.glob(f"{PREFIX}*.json"):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            voice = CustomVoice(
                id=data["id"],
                display_name=data.get("display_name", data["id"]),
                created_at=float(data.get("created_at", 0.0)),
            )
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Bỏ qua giọng nhân bản hỏng tại %s: %s", meta, exc)
            continue
        if not (meta.parent / f"{voice.id}.wav").is_file():
            continue  # mất file mẫu thì giọng vô dụng — coi như không tồn tại
        voices.append(voice)
    return sorted(voices, key=lambda v: v.created_at)


def get(voice_id: str) -> CustomVoice | None:
    return next((v for v in list_custom() if v.id == voice_id), None)


def unique_id(name: str) -> str:
    """Slug không đụng giọng đã có VÀ không tái sử dụng id cũ.

    Không tái sử dụng là bắt buộc: engine giữ giọng đã đăng ký trong RAM theo id,
    nên xóa rồi tạo lại cùng id với mẫu khác sẽ đọc bằng giọng CŨ tới khi restart.
    """
    base = PREFIX + slugify(name)
    taken = {p.stem for p in _store().glob(f"{PREFIX}*.json")}
    taken |= {p.stem for p in _store().glob(f"{PREFIX}*.tombstone")}
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def register(voice_id: str, display_name: str) -> CustomVoice:
    """Ghi metadata. File mẫu phải đã nằm sẵn ở `sample_path(voice_id)`."""
    voice = CustomVoice(id=voice_id, display_name=display_name, created_at=time.time())
    (_store() / f"{voice_id}.json").write_text(
        json.dumps({"id": voice.id, "display_name": voice.display_name,
                    "created_at": voice.created_at}, ensure_ascii=False),
        encoding="utf-8",
    )
    return voice


def remove(voice_id: str) -> bool:
    if get(voice_id) is None:
        return False
    store = _store()
    (store / f"{voice_id}.json").unlink(missing_ok=True)
    (store / f"{voice_id}.wav").unlink(missing_ok=True)
    # Bia mộ giữ chỗ id để unique_id không cấp lại (xem docstring của unique_id).
    (store / f"{voice_id}.tombstone").touch()
    return True
