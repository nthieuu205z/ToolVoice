"""Tải các model chạy trên máy theo yêu cầu, một lần một, và báo tiến trình.

Tiến trình đo bằng dung lượng thư mục `blobs/` trên đĩa, nên không phụ thuộc vào
thư viện tải bên dưới có phơi ra thanh tiến trình hay không.
"""

from __future__ import annotations

import logging
import threading

from pipeline import model_store
from pipeline.model_store import ModelSpec

log = logging.getLogger(__name__)


class ModelDownloader:
    """Một slot duy nhất cho tất cả model. Bấm tải lần hai khi đang tải sẽ bị từ chối."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._active_key = ""
        self._status = "idle"   # idle | downloading | error
        self._message = ""

    def is_downloading(self) -> bool:
        return self._status == "downloading"

    def start(self, spec: ModelSpec) -> None:
        """Ném RuntimeError nếu đang tải — route đổi thành HTTP 409."""
        with self._lock:
            if self._status == "downloading":
                raise RuntimeError("Đang tải một model rồi")
            self._active_key = spec.key
            self._status = "downloading"
            self._message = "Đang kết nối tới Hugging Face…"

        self._thread = threading.Thread(target=self._run, args=(spec,), daemon=True,
                                        name=f"download-{spec.key}")
        self._thread.start()

    def _run(self, spec: ModelSpec) -> None:
        try:
            model_store.download(spec)
        except Exception as exc:
            log.exception("Tải model %s thất bại", spec.key)
            self._message = f"Tải model thất bại: {exc}"
            self._status = "error"
        else:
            self._message = "Đã tải xong"
            self._status = "idle"

    def snapshot(self, spec: ModelSpec) -> dict:
        """Trạng thái một model, kèm tiến trình nếu chính nó đang được tải."""
        try:
            info = model_store.describe(spec)
        except ValueError as exc:
            return {"key": spec.key, "label": spec.label, "ready": False,
                    "status": "error", "message": str(exc),
                    "percent": 0.0, "downloaded_mb": 0.0, "total_mb": 0.0}

        mine = self._active_key == spec.key
        if mine and self._status == "downloading":
            status = "downloading"
        elif info.ready:
            status = "ready"
        elif mine and self._status == "error":
            status = "error"
        else:
            status = "idle"

        return {
            "key": info.key,
            "label": info.label,
            "ready": info.ready,
            "status": status,
            "message": self._message if mine else "",
            "percent": info.percent,
            "downloaded_mb": round(info.downloaded_bytes / 1e6, 1),
            "total_mb": round(info.total_bytes / 1e6, 1),
        }


downloader = ModelDownloader()
