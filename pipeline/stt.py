"""Bước nhận diện giọng nói: chép lời cho từng vùng tiếng nói mà ffmpeg đã xác định.

Mốc thời gian KHÔNG lấy từ Gemini — nó không đo được (xem pipeline/segmentation.py).
Mỗi vùng được gửi đi riêng nên văn bản trả về chắc chắn thuộc đúng khoảng thời gian đó.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .audio import write_wav
from .errors import JobCancelledError, NoSpeechDetectedError
from .models import CancelFn, GeminiBackend, ProgressFn, Segment, never_cancel, noop_progress
from .segmentation import Region

log = logging.getLogger(__name__)


def transcribe_regions(
    backend: GeminiBackend,
    samples: np.ndarray,
    rate: int,
    regions: list[Region],
    workdir: Path,
    *,
    workers: int = 4,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
    progress_base: int = 0,
    progress_total: int | None = None,
) -> tuple[str, list[Segment], list[str]]:
    """Chép lời song song từng vùng, giữ nguyên thứ tự thời gian.

    Trả về (ngôn ngữ, các lượt thoại, cảnh báo). Đoạn hỏng được thử lại lần hai
    TUẦN TỰ sau khi cơn dồn request đã dịu — đo thật: 8 luồng song song thỉnh
    thoảng dính 429 của Vertex, và một đoạn bị bỏ là mất hẳn lời thoại đoạn đó.

    `progress_base`/`progress_total`: khi video xử lý theo cụm gối đầu, quy số đếm
    về toàn video thay vì cụm hiện tại.
    """
    if not regions:
        raise NoSpeechDetectedError()
    grand_total = progress_total if progress_total is not None else len(regions)

    # Whisper trên GPU chép cả lô trong một lượt gọi — nhanh gấp 9,4 lần so với từng vùng
    # một (xem pipeline/whisper_stt.py::batch_size). Backend nào không gộp được trả về 0.
    if getattr(backend, "stt_batch_size", 0) > 0:
        return _transcribe_theo_lo(
            backend, samples, rate, regions,
            progress=progress, should_cancel=should_cancel,
            progress_base=progress_base, grand_total=grand_total,
        )

    clips_dir = workdir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    results: dict[int, tuple[str, str]] = {}
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_transcribe_one, backend, samples, rate, region, clips_dir, i): i
            for i, region in enumerate(regions)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            if should_cancel():
                # Bỏ những đoạn còn xếp hàng; đoạn đang chạy tự kết thúc trong vài giây.
                pool.shutdown(wait=False, cancel_futures=True)
                raise JobCancelledError()

            index = futures[future]
            progress("transcribe", (progress_base + done) / max(1, grand_total),
                     f"Nhận diện đoạn {progress_base + done}/{grand_total}")
            try:
                results[index] = future.result()
            except Exception as exc:
                log.warning("Không nhận diện được đoạn %d (sẽ thử lại): %s", index, exc)
                failed.append(index)

    lost: list[int] = []
    for index in failed:
        if should_cancel():
            raise JobCancelledError()
        progress("transcribe", (progress_base + len(regions)) / max(1, grand_total),
                 f"Thử lại đoạn nhận diện hỏng (quanh mốc {regions[index].start:.0f}s)")
        try:
            results[index] = _transcribe_one(backend, samples, rate, regions[index], clips_dir, index)
        except Exception as exc:
            log.warning("Đoạn %d hỏng cả lần thử lại: %s", index, exc)
            lost.append(index)

    warnings: list[str] = []
    if lost:
        spots = ", ".join(f"{regions[i].start:.0f}s" for i in lost[:5])
        warnings.append(
            f"{len(lost)} đoạn không nhận diện được (quanh mốc {spots}) — "
            "các đoạn này sẽ thiếu lời thoại. Chạy lại video thường khắc phục được."
        )

    language = next((lang for _, (lang, _) in sorted(results.items()) if lang), "")
    segments = [
        Segment(start=regions[i].start, end=regions[i].end, text=text)
        for i, (_, text) in sorted(results.items())
        if text.strip()
    ]

    progress("transcribe", min(1.0, (progress_base + len(regions)) / max(1, grand_total)),
             f"Đã nhận diện {len(segments)} lượt thoại")
    if not segments:
        raise NoSpeechDetectedError()
    return language, segments, warnings


def _chia_lo(so_luong: int, tran: int) -> list[int]:
    """Chia đều thành các lô không quá `tran` — giống pipeline/tts.py."""
    if so_luong <= 0:
        return []
    so_lo = (so_luong + tran - 1) // tran
    deu, du = divmod(so_luong, so_lo)
    return [deu + (1 if i < du else 0) for i in range(so_lo)]


def _transcribe_theo_lo(
    backend: GeminiBackend, samples: np.ndarray, rate: int, regions: list[Region], *,
    progress: ProgressFn, should_cancel: CancelFn, progress_base: int, grand_total: int,
) -> tuple[str, list[Segment], list[str]]:
    """Chép cả lô một lượt trên GPU. Giữ nguyên hai lời hứa của đường cũ:

    - Hủy giữa chừng còn ăn: kiểm tra trước mỗi lô (lô chạy vài giây, không phải cả video).
    - Một đoạn hỏng không giết cả video: lô nào ném lỗi thì lùi về chép từng vùng trong lô đó.
    """
    ket: dict[int, tuple[str, str]] = {}
    lost: list[int] = []
    xong = 0
    vi_tri = 0

    for co_lo in _chia_lo(len(regions), backend.stt_batch_size):
        if should_cancel():
            raise JobCancelledError()

        lo = regions[vi_tri : vi_tri + co_lo]
        goc = vi_tri
        vi_tri += co_lo

        try:
            ra = backend.transcribe_batch(samples, rate, lo)
        except Exception as exc:
            log.warning("Lô %d đoạn hỏng (%s) — chép lại từng đoạn một", len(lo), exc)
            ra = []
            for r in lo:
                try:
                    ra.append(_transcribe_mot_vung(backend, samples, rate, r))
                except Exception as le:
                    log.warning("Đoạn quanh mốc %.0fs hỏng cả lần thử lại: %s", r.start, le)
                    ra.append(("", ""))

        for offset, cap in enumerate(ra):
            index = goc + offset
            if cap[1].strip():
                ket[index] = cap
            else:
                lost.append(index)

        xong += co_lo
        progress("transcribe", (progress_base + xong) / max(1, grand_total),
                 f"Nhận diện đoạn {progress_base + xong}/{grand_total}")

    warnings: list[str] = []
    # Vùng không ra chữ nào thường là nhạc nền hoặc tiếng động, không phải lỗi — chỉ báo
    # khi số đó lớn bất thường, kẻo mỗi video nào cũng hiện một cảnh báo vô nghĩa.
    if len(lost) > max(3, len(regions) // 10):
        spots = ", ".join(f"{regions[i].start:.0f}s" for i in lost[:5])
        warnings.append(
            f"{len(lost)} đoạn không nhận diện được (quanh mốc {spots}) — "
            "các đoạn này sẽ thiếu lời thoại. Chạy lại video thường khắc phục được."
        )

    language = next((lang for _, (lang, _) in sorted(ket.items()) if lang), "")
    segments = [
        Segment(start=regions[i].start, end=regions[i].end, text=text)
        for i, (_, text) in sorted(ket.items())
    ]

    progress("transcribe", min(1.0, (progress_base + len(regions)) / max(1, grand_total)),
             f"Đã nhận diện {len(segments)} lượt thoại")
    if not segments:
        raise NoSpeechDetectedError()
    return language, segments, warnings


def _transcribe_mot_vung(
    backend: GeminiBackend, samples: np.ndarray, rate: int, region: Region,
) -> tuple[str, str]:
    """Đường lùi khi cả lô hỏng: cắt một vùng ra file tạm rồi chép riêng."""
    import tempfile

    begin = max(0, int(region.start * rate))
    finish = min(len(samples), int(region.end * rate))
    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "region.wav"
        write_wav(clip, samples[begin:finish], rate)
        return backend.transcribe_clip(clip)


def _transcribe_one(
    backend: GeminiBackend, samples: np.ndarray, rate: int,
    region: Region, clips_dir: Path, index: int,
) -> tuple[str, str]:
    """Cắt vùng ra WAV bằng numpy (không gọi ffmpeg) rồi gửi cho Gemini."""
    begin = max(0, int(region.start * rate))
    finish = min(len(samples), int(region.end * rate))
    clip = clips_dir / f"region_{index:04d}.wav"
    write_wav(clip, samples[begin:finish], rate)
    try:
        return backend.transcribe_clip(clip)
    finally:
        clip.unlink(missing_ok=True)
