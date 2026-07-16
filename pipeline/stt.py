"""Bước nhận diện giọng nói: chép lời cho từng vùng tiếng nói mà ffmpeg đã xác định.

Mốc thời gian KHÔNG lấy từ Gemini — nó không đo được (xem pipeline/segmentation.py).
Mỗi vùng được gửi đi riêng nên văn bản trả về chắc chắn thuộc đúng khoảng thời gian đó.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from .audio import write_wav
from .errors import JobCancelledError, NoSpeechDetectedError
from .models import CancelFn, GeminiBackend, ProgressFn, Segment, never_cancel, noop_progress
from .segmentation import Region

log = logging.getLogger(__name__)

# Ký tự kết thúc một câu trọn vẹn (Whisper/Gemini có chấm câu). Mảnh KHÔNG kết bằng
# một trong số này nghĩa là câu còn dang dở, nối tiếp sang vùng sau.
_SENTENCE_END = tuple(".!?…")


def merge_sentence_fragments(segments: list[Segment], max_duration: float) -> list[Segment]:
    """Gộp các mảnh vụn thành CÂU TRỌN theo dấu câu, không vượt `max_duration`.

    Segmentation cắt theo khoảng lặng của audio TIẾNG ANH, nên một câu hay bị chẻ thành
    nhiều mảnh ở chỗ người nói ngừng lấy hơi. Dịch & đọc từng mảnh rời làm cụm tiếng Việt
    bị ngắt giữa chừng ("Hôm ...(nghỉ)... nay") và lệch nhịp so với hình. Gộp lại theo câu
    (mảnh không kết bằng .!?… được nối với mảnh sau) cho tiếng Việt liền mạch, dịch đúng
    ngữ cảnh cả câu. Trần `max_duration` chặn câu dài vô hạn khi thiếu chấm câu.

    Chỉ đụng khi nhận diện chạy TRÊN MÁY có chấm câu (Whisper). Mốc thời gian gộp lại là
    (đầu mảnh đầu, cuối mảnh cuối) — vẫn là mốc thật của ffmpeg.
    """
    if not segments:
        return []
    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_open = not prev.text.strip().rstrip('"”\'’)').endswith(_SENTENCE_END)
        if prev_open and (seg.end - prev.start) <= max_duration:
            merged[-1] = replace(prev, end=seg.end,
                                 text=f"{prev.text.strip()} {seg.text.strip()}".strip())
        else:
            merged.append(seg)
    if len(merged) != len(segments):
        log.info("Gộp mảnh vụn theo câu: %d vùng → %d câu", len(segments), len(merged))
    return merged


def transcribe_regions(
    backend: GeminiBackend,
    samples: np.ndarray,
    rate: int,
    regions: list[Region],
    workdir: Path,
    *,
    workers: int = 4,
    sentence_level: bool = True,
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

    # Whisper/GPU trả về mốc thời gian cấp CÂU: mỗi câu thành một Segment đặt đúng thời điểm
    # câu tiếng Anh, thay vì nhồi cả vùng ffmpeg (2–4 câu) làm một khối rồi đặt tại một mốc.
    # Đây là gốc rễ của việc bám hình sát và hết cảnh "Hôm ...(nghỉ)... nay" (xem
    # merge_sentence_fragments ghép nốt câu bị vùng chẻ đôi). Vùng ffmpeg vẫn ràng buộc
    # Whisper chỉ được chép trong chỗ CÓ tiếng — nó lo THỜI GIAN THÔ, Whisper tinh mốc câu.
    if sentence_level and getattr(backend, "stt_timed", False):
        return _transcribe_timed_theo_lo(
            backend, samples, rate, regions,
            progress=progress, should_cancel=should_cancel,
            progress_base=progress_base, grand_total=grand_total,
        )

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


def _transcribe_timed_theo_lo(
    backend: GeminiBackend, samples: np.ndarray, rate: int, regions: list[Region], *,
    progress: ProgressFn, should_cancel: CancelFn, progress_base: int, grand_total: int,
) -> tuple[str, list[Segment], list[str]]:
    """Chép cả lô lấy mốc thời gian cấp CÂU. Mỗi mảnh câu Whisper trả về → một Segment
    đặt đúng thời điểm câu tiếng Anh (không nhồi cả vùng làm một khối). Giữ hai lời hứa:

    - Hủy giữa chừng còn ăn: kiểm tra trước mỗi lô.
    - Lô hỏng không giết cả video: lùi về chép CẢ VÙNG (mất mốc con nhưng vẫn có lời thoại).
    """
    pieces: list[tuple[float, float, str]] = []
    language = ""
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
            lang, ra = backend.transcribe_batch_timed(samples, rate, lo)
            language = language or lang
            pieces.extend(ra)
        except Exception as exc:
            log.warning("Lô %d vùng lấy mốc câu hỏng (%s) — lùi về chép cả vùng", len(lo), exc)
            for offset, r in enumerate(lo):
                try:
                    lang, text = _transcribe_mot_vung(backend, samples, rate, r)
                    language = language or lang
                    if text.strip():
                        pieces.append((r.start, r.end, text.strip()))
                    else:
                        lost.append(goc + offset)
                except Exception as le:
                    log.warning("Vùng quanh mốc %.0fs hỏng cả lần lùi: %s", r.start, le)
                    lost.append(goc + offset)

        xong += co_lo
        progress("transcribe", (progress_base + xong) / max(1, grand_total),
                 f"Nhận diện đoạn {progress_base + xong}/{grand_total}")

    pieces = [p for p in pieces if p[2].strip()]
    pieces.sort(key=lambda p: p[0])
    segments = [Segment(start=s, end=e, text=t.strip()) for s, e, t in pieces]

    warnings: list[str] = []
    if len(lost) > max(3, len(regions) // 10):
        spots = ", ".join(f"{regions[i].start:.0f}s" for i in lost[:5])
        warnings.append(
            f"{len(lost)} đoạn không nhận diện được (quanh mốc {spots}) — "
            "các đoạn này sẽ thiếu lời thoại. Chạy lại video thường khắc phục được."
        )

    progress("transcribe", min(1.0, (progress_base + len(regions)) / max(1, grand_total)),
             f"Đã nhận diện {len(segments)} câu")
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
