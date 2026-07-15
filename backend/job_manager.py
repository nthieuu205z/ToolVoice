"""Quản lý công việc: nhiều video cùng lúc, chạy trên một pool luồng có trần.

Pipeline toàn là lời gọi chặn (subprocess + HTTP) nên dùng thread thay vì asyncio.
Trần số job chạy đồng thời (MAX_CONCURRENT_JOBS) thấp có chủ ý: Whisper và VieNeu
đều bị khóa suy luận toàn cục, edge-tts bị trần 2 request đồng thời — job thứ ba
chủ yếu chỉ chen hàng chứ không nhanh thêm. Vượt trần thì xếp hàng ("queued").

Mỗi job ghi trạng thái xuống `workdir/job.json` tại các mốc chuyển trạng thái, nên
đóng trình duyệt hay khởi động lại server vẫn thấy lại danh sách; job đang chạy dở
lúc server chết được đánh dấu lỗi khi khôi phục (thread của nó không sống lại được).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.errors import JobCancelledError, PipelineError
from pipeline.models import MediaInfo, overall_percent
from pipeline.runner import PipelineOptions, run_pipeline

log = logging.getLogger(__name__)

# Trên ngưỡng này thì video coi như hỏng nặng: vẫn tải về được nhưng phải cảnh báo đỏ.
SILENT_RATIO_ALERT = 0.2

# Trạng thái còn "sống": chiếm slot chạy và không được dọn thư mục.
ACTIVE_STATUSES = ("queued", "running", "cancelling")

_INTERRUPTED_MESSAGE = "Máy chủ đã dừng khi video này đang xử lý. Hãy chạy lại video."


@dataclass
class Job:
    id: str
    filename: str
    workdir: Path
    voice_id: str
    status: str = "queued"  # queued | running | cancelling | cancelled | done | error
    stage: str = "extract"
    percent: float = 0.0
    message: str = ""
    video_path: str = ""
    srt_path: str = ""
    attempted_count: int = 0
    spoken_count: int = 0
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # Không giết thread giữa chừng (ffmpeg đang ghi file); pipeline tự đọc cờ này.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def silent_ratio(self) -> float:
        if self.attempted_count <= 0:
            return 0.0
        return (self.attempted_count - self.spoken_count) / self.attempted_count

    @property
    def degraded(self) -> bool:
        return self.status == "done" and self.silent_ratio >= SILENT_RATIO_ALERT

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "stage": self.stage,
            "percent": self.percent,
            "message": self.message,
            "filename": self.filename,
            "voice_id": self.voice_id,
            "created_at": round(self.created_at, 3),
            "warnings": self.warnings,
            "attempted_count": self.attempted_count,
            "spoken_count": self.spoken_count,
            "silent_ratio": round(self.silent_ratio, 3),
            "degraded": self.degraded,
        }


class JobManager:
    """Sổ đăng ký job + pool luồng. Route chỉ đọc snapshot, không đụng thread."""

    def __init__(self, max_workers: int | None = None):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._max_workers = max_workers

    # ─── tra cứu ────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def jobs(self) -> list[dict]:
        """Snapshot mọi job, mới nhất trước — giao diện vẽ thẳng danh sách này."""
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.snapshot() for job in ordered]

    @property
    def current(self) -> Job | None:
        """Job đang sống mới nhất — giữ cho /api/jobs/current cũ tiếp tục chạy."""
        active = [j for j in self._jobs.values() if j.status in ACTIVE_STATUSES]
        return max(active, key=lambda j: j.created_at) if active else None

    def is_busy(self) -> bool:
        with self._lock:
            return any(j.status in ACTIVE_STATUSES for j in self._jobs.values())

    # ─── vòng đời ───────────────────────────────────────────────────

    def start(self, *, filename: str, workdir: Path, voice_id: str, video_path: Path,
              media: MediaInfo, backend_factory, options: PipelineOptions) -> Job:
        """Nhận job mới. Quá trần chạy đồng thời thì job nằm hàng đợi, không từ chối."""
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, workdir=workdir, voice_id=voice_id)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)

        future = self._ensure_executor().submit(
            self._run, job, video_path, media, backend_factory, options
        )
        self._futures[job.id] = future
        return job

    def cancel(self, job_id: str) -> Job:
        """Yêu cầu dừng. Ném RuntimeError nếu job đã kết thúc — route đổi thành HTTP 409."""
        job = self.get(job_id)
        if job is None:
            raise LookupError(job_id)
        with self._lock:
            if job.status not in ACTIVE_STATUSES:
                raise RuntimeError(f"Công việc đã kết thúc ({job.status}), không hủy được.")
            job.cancel_event.set()

            future = self._futures.get(job_id)
            if job.status == "queued" and future is not None and future.cancel():
                # Còn nằm trong hàng đợi, chưa chiếm luồng nào — hủy được ngay lập tức.
                job.message = JobCancelledError.user_message
                job.status = "cancelled"
            else:
                job.status = "cancelling"
                job.message = "Đang dừng…"
        self._persist(job)
        log.info("Job %s: người dùng yêu cầu hủy ở bước %s (→ %s)", job.id, job.stage, job.status)
        return job

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            workers = self._max_workers
            if workers is None:
                from backend.config import settings

                workers = settings.max_concurrent_jobs
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, workers), thread_name_prefix="pipeline"
            )
        return self._executor

    def _run(self, job: Job, video_path: Path, media: MediaInfo, backend_factory,
             options: PipelineOptions) -> None:
        if job.cancel_requested:   # bấm hủy khi còn trong hàng đợi
            job.message = JobCancelledError.user_message
            job.status = "cancelled"
            self._persist(job)
            return

        job.status = "running"
        self._persist(job)

        def progress(stage: str, fraction: float, message: str) -> None:
            # Trong lúc đang dừng thì đừng ghi đè thông điệp "Đang dừng…".
            if job.cancel_requested:
                return
            job.stage = stage
            job.percent = overall_percent(stage, fraction)
            job.message = message

        try:
            result = run_pipeline(backend_factory(), video_path, job.workdir, options,
                                  progress, media, job.cancel_event.is_set)
        except JobCancelledError as exc:
            log.info("Job %s đã dừng theo yêu cầu ở bước %s", job.id, job.stage)
            job.message = exc.user_message
            job.status = "cancelled"
        except PipelineError as exc:
            log.warning("Job %s lỗi ở bước %s: %s", job.id, job.stage, exc)
            job.message = exc.user_message
            job.status = "error"
        except Exception as exc:  # lỗi ngoài dự kiến — vẫn phải hiện được lên UI
            log.exception("Job %s hỏng bất ngờ", job.id)
            job.message = f"Lỗi không lường trước: {exc}"
            job.status = "error"
        else:
            # Gán kết quả TRƯỚC khi lật status: giao diện thấy "done" là dừng đọc ngay,
            # nên nếu lật trước thì cảnh báo và đường dẫn file có thể chưa kịp có mặt.
            job.percent = 100.0
            job.stage = "mux"
            job.message = "Hoàn tất"
            job.video_path = result.video_path
            job.srt_path = result.srt_path
            job.warnings = result.warnings
            job.attempted_count = result.attempted_count
            job.spoken_count = result.spoken_count
            job.status = "done"
        self._persist(job)

    # ─── lưu và khôi phục ───────────────────────────────────────────

    def _persist(self, job: Job) -> None:
        """Ghi trạng thái xuống thư mục của job. Đĩa hỏng không được phép giết pipeline."""
        data = job.snapshot()
        data["video_path"] = job.video_path
        data["srt_path"] = job.srt_path
        try:
            (job.workdir / "job.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Không ghi được job.json cho %s: %s", job.id, exc)

    def restore(self, jobs_dir: Path) -> int:
        """Nạp lại các job từ đĩa lúc khởi động server.

        Job đang chạy dở lúc server chết không thể tiếp tục (thread đã mất) — đánh dấu
        lỗi và nói thẳng, thay vì hiện "running" mãi mãi với một tiến trình ma.
        """
        if not jobs_dir.is_dir():
            return 0

        count = 0
        for meta in jobs_dir.glob("*/job.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job = Job(
                    id=data["job_id"],
                    filename=data.get("filename", ""),
                    workdir=meta.parent,
                    voice_id=data.get("voice_id", ""),
                    status=data.get("status", "error"),
                    stage=data.get("stage", "extract"),
                    percent=float(data.get("percent", 0.0)),
                    message=data.get("message", ""),
                    video_path=data.get("video_path", ""),
                    srt_path=data.get("srt_path", ""),
                    attempted_count=int(data.get("attempted_count", 0)),
                    spoken_count=int(data.get("spoken_count", 0)),
                    warnings=list(data.get("warnings", [])),
                    created_at=float(data.get("created_at", meta.stat().st_mtime)),
                )
            except (OSError, ValueError, KeyError) as exc:
                log.warning("Bỏ qua job.json hỏng tại %s: %s", meta, exc)
                continue

            if job.status == "cancelling":
                job.status = "cancelled"
                job.message = JobCancelledError.user_message
                self._persist(job)
            elif job.status in ("queued", "running"):
                job.status = "error"
                job.message = _INTERRUPTED_MESSAGE
                self._persist(job)

            self._jobs[job.id] = job
            count += 1

        if count:
            log.info("Khôi phục %d job từ %s", count, jobs_dir)
        return count

    def prune(self, jobs_dir: Path, keep: int = 10) -> None:
        """Dọn các thư mục job cũ nhất, không bao giờ đụng vào job đang sống."""
        if not jobs_dir.is_dir():
            return
        active_dirs = {j.workdir.resolve() for j in self._jobs.values()
                       if j.status in ACTIVE_STATUSES}
        by_dir = {j.workdir.resolve(): j.id for j in self._jobs.values()}

        dirs = sorted(
            (d for d in jobs_dir.iterdir() if d.is_dir()),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for stale in dirs[keep:]:
            resolved = stale.resolve()
            if resolved in active_dirs:
                continue
            shutil.rmtree(stale, ignore_errors=True)
            job_id = by_dir.get(resolved)
            if job_id:  # file đã mất thì đừng để job ma trong danh sách
                self._jobs.pop(job_id, None)
                self._futures.pop(job_id, None)


manager = JobManager()
