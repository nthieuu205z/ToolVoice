"""Bước tạo giọng đọc: mỗi lượt phát ngôn tiếng Việt → một đoạn PCM vừa khung thời gian gốc."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .audio import duration_of, fit_to_window, pcm_to_array, truncate_with_fade
from .errors import JobCancelledError, QuotaExhaustedError
from .models import CancelFn, GeminiBackend, ProgressFn, Segment, never_cancel, noop_progress

log = logging.getLogger(__name__)

# Chỉ cảnh báo khi lời thoại tràn khung đủ nhiều để tai nghe ra.
_OVERFLOW_WARN_SECONDS = 0.3
# Phần khoảng lặng KHÔNG cho mượn. Đây là chỗ hồi của cả chuỗi: khi một câu tràn cứng
# đẩy các câu sau lùi lại, mỗi khoảng lặng chỉ hấp thụ được phần không bị mượn mất.
# Đo thật: cho mượn cạn kiệt → độ lệch tan 0,03s/bước, 5 câu tràn kéo 133 câu lùi theo.
_BORROW_RESERVE = 0.4

# Chống TTS "chạy hoang": model tự hồi quy thỉnh thoảng bịa thêm lời khi đầu vào quá
# ngắn — đo thật: chữ "Và" (2 ký tự) sinh ra 7,1 giây giọng nói, kéo 15 câu sau lệch
# 3–6 giây. Đọc chậm nhất còn hợp lý là ~8 ký tự/giây (VieNeu thực tế đọc 14–17),
# vượt trần đó là audio bịa — cắt bỏ, từ thật luôn nằm ở phần đầu.
_RUNAWAY_CHARS_PER_SECOND = 8.0
_RUNAWAY_HEADROOM_SECONDS = 1.0
# Sàn cố định: VieNeu tốn ~2–3,5s cho MỌI lượt đọc bất kể dài ngắn (ngữ điệu, hơi thở,
# đuôi câu), gần như không theo số ký tự. Ngưỡng tuyến tính len/8+1 quá chặt với câu
# ngắn — "Thứ hai," (8 ký tự) chỉ được 2,0s trong khi giọng thật ~3s → cắt cụt tiếng thật
# (nghe "ngắt đột ngột"). Đo raw: đọc ngắn bình thường ≤3,5s, chạy hoang ≥6s. Sàn 4s tách
# sạch hai loại — câu ngắn thật sống, còn "Và"→8s vẫn bị cắt.
_RUNAWAY_MIN_SECONDS = 4.0


def synthesize_segments(
    backend: GeminiBackend,
    segments: list[Segment],
    voice_id: str,
    *,
    workers: int = 4,
    max_speedup: float = 1.5,
    total_duration: float | None = None,
    next_utterance_start: float | None = None,
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
    progress_base: int = 0,
    progress_total: int | None = None,
) -> tuple[list[tuple[Segment, np.ndarray]], list[str]]:
    """Tạo giọng đọc song song. Trả về ([(lượt phát ngôn, mẫu âm thanh)], cảnh báo).

    Tiếng Việt dài hơn khung gốc thì trước hết cho MƯỢN khoảng lặng phía sau (tới tận
    mốc câu kế tiếp) — đọc tốc độ tự nhiên tràn vào chỗ im lặng nghe thật hơn hẳn
    giọng bị tăng tốc. Chỉ khi hết cả chỗ mượn mới tăng tốc, tối đa `max_speedup`.

    Một lượt hỏng không giết cả video — nó chỉ để lại khoảng lặng và một cảnh báo.
    Nhưng hết hạn mức theo ngày thì mọi lượt sau đó cũng hỏng, nên được báo riêng.

    `next_utterance_start`: mốc lượt phát ngôn ĐẦU TIÊN của cụm kế tiếp, khi video
    được xử lý theo cụm gối đầu — để lượt cuối cụm không mượn lố sang cụm sau.
    `progress_base`/`progress_total`: quy số đếm về toàn video thay vì cụm hiện tại.
    """
    todo = [(i, seg) for i, seg in enumerate(segments) if seg.text_vi.strip()]
    if not todo:
        return [], ["Không có lời thoại nào để đọc."]

    grand_total = progress_total if progress_total is not None else len(todo)
    results: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    overflowed = 0
    failed = 0
    quota_hit = False

    # Engine GPU đọc cả một lô trong một lượt gọi — nhanh hơn ~20 lần so với từng câu một
    # (xem pipeline/vieneu_batch.py). Backend nào không gộp được thì trả batch_size 0.
    if getattr(backend, "batch_size", 0) > 0:
        return _synthesize_theo_lo(
            backend, segments, todo, voice_id, max_speedup=max_speedup,
            total_duration=total_duration, next_utterance_start=next_utterance_start,
            progress=progress, should_cancel=should_cancel,
            progress_base=progress_base, grand_total=grand_total,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_render_one, backend, seg, voice_id, max_speedup,
                        _allowed_window(segments, i, total_duration,
                                        next_utterance_start)): (i, seg)
            for i, seg in todo
        }
        for done, future in enumerate(as_completed(futures), start=1):
            if should_cancel():
                # Bỏ những lượt còn xếp hàng; lượt đang chạy tự kết thúc trong vài giây.
                pool.shutdown(wait=False, cancel_futures=True)
                raise JobCancelledError()

            index, _ = futures[future]
            progress("synthesize", (progress_base + done) / max(1, grand_total),
                     f"Đang đọc lượt thoại {progress_base + done}/{grand_total}")

            try:
                samples, overflow = future.result()
            except QuotaExhaustedError:
                quota_hit = True
                failed += 1
                continue
            except Exception as exc:
                log.warning("Không đọc được lượt thoại %d: %s", index, exc)
                failed += 1
                continue

            results[index] = samples
            # Phụ đề bám theo thời lượng đọc thật, không theo khung thời gian gốc.
            segments[index].spoken_duration = duration_of(samples)
            if overflow > _OVERFLOW_WARN_SECONDS:
                overflowed += 1

    if quota_hit:
        warnings.append(QuotaExhaustedError.user_message)
    if failed:
        warnings.append(f"{failed}/{len(todo)} lượt thoại không tạo được giọng đọc và đã bị bỏ trống.")
    if overflowed:
        warnings.append(
            f"{overflowed} lượt thoại dài hơn cả khung gốc lẫn khoảng lặng kế tiếp dù đã "
            "tăng tốc — các câu ngay sau bị lùi lại một chút, không mất chữ nào."
        )

    fitted = [(segments[i], results[i]) for i in sorted(results)]
    progress("synthesize", min(1.0, (progress_base + len(todo)) / max(1, grand_total)),
             f"Đã tạo {progress_base + len(fitted)}/{grand_total} lượt thoại")
    return fitted, warnings


def _allowed_window(segments: list[Segment], index: int, total_duration: float | None,
                    next_utterance_start: float | None = None) -> float:
    """Lời thoại được phép kéo dài tới đâu: khung gốc + PHẦN CHO MƯỢN của khoảng lặng.

    Không cho mượn cả khoảng lặng: phần chừa lại (_BORROW_RESERVE) là chỗ để độ lệch
    dồn toa tự tan. Khoảng lặng ngắn hơn mức chừa thì không mượn được gì — câu phải
    vừa khung gốc (tăng tốc) hoặc chịu lùi. Sau lượt cuối, ranh giới lần lượt là:
    mốc cụm kế tiếp (xử lý theo cụm) → hết video → đành dùng khung gốc.
    """
    seg = segments[index]
    if index + 1 < len(segments):
        next_start = segments[index + 1].start
    elif next_utterance_start is not None:
        next_start = next_utterance_start
    else:
        next_start = total_duration
    if next_start is None:
        return seg.duration
    borrowable = max(0.0, (next_start - seg.end) - _BORROW_RESERVE)
    return seg.duration + borrowable


def _fit_one(seg: Segment, pcm: bytes, max_speedup: float,
             allowed_duration: float) -> tuple[np.ndarray, float]:
    """Chống chạy hoang rồi nhét lời thoại vừa khung thời gian. Dùng chung cho cả hai đường."""
    samples = pcm_to_array(pcm)

    sane = max(
        len(seg.text_vi) / _RUNAWAY_CHARS_PER_SECOND + _RUNAWAY_HEADROOM_SECONDS,
        _RUNAWAY_MIN_SECONDS,
    )
    if duration_of(samples) > sane:
        log.warning("TTS chạy hoang: %.1fs audio cho %d ký tự (%.40r) — cắt về %.1fs",
                    duration_of(samples), len(seg.text_vi), seg.text_vi, sane)
        samples = truncate_with_fade(samples, sane)

    return fit_to_window(samples, seg.duration, allowed_duration, max_speedup)


def _render_one(
    backend: GeminiBackend, seg: Segment, voice_id: str, max_speedup: float,
    allowed_duration: float,
) -> tuple[np.ndarray, float]:
    """Gọi TTS đúng MỘT lần — tenacity bên trong GeminiRunner đã lo việc thử lại.

    Gọi lại mù ở đây từng nhân đôi mọi request và làm cạn hạn mức nhanh gấp đôi.
    """
    pcm = backend.synthesize(seg.text_vi, voice_id)
    return _fit_one(seg, pcm, max_speedup, allowed_duration)


def _chia_lo(so_luong: int, tran: int) -> list[int]:
    """Chia `so_luong` việc thành các lô ĐỀU NHAU, không lô nào vượt `tran`.

    Chia đều thay vì cắt thẳng theo trần vì mọi hàng trong lô phải chạy tới khi hàng dài
    nhất đọc xong — một lô cuối lèo tèo 5 câu vẫn tốn gần đủ thời gian của một lô đầy.
    53 câu, trần 32: hai lô 27+26 chứ không phải 32+21.
    """
    if so_luong <= 0:
        return []
    so_lo = (so_luong + tran - 1) // tran
    deu, du = divmod(so_luong, so_lo)
    return [deu + (1 if i < du else 0) for i in range(so_lo)]


def _synthesize_theo_lo(
    backend: GeminiBackend, segments: list[Segment], todo: list[tuple[int, Segment]],
    voice_id: str, *, max_speedup: float, total_duration: float | None,
    next_utterance_start: float | None, progress: ProgressFn, should_cancel: CancelFn,
    progress_base: int, grand_total: int,
) -> tuple[list[tuple[Segment, np.ndarray]], list[str]]:
    """Đọc cả lô một lượt trên GPU. Giữ nguyên hai lời hứa của đường cũ:

    - Hủy giữa chừng vẫn ăn: kiểm tra trước mỗi lô (lô chạy vài giây, không phải cả video).
    - Một lượt hỏng không giết cả video: lô nào ném lỗi thì lùi về đọc từng câu trong lô đó,
      để chỉ đúng câu hỏng bị bỏ trống chứ không mất cả 32 câu cùng nó.
    """
    results: dict[int, np.ndarray] = {}
    warnings: list[str] = []
    overflowed = failed = 0
    xong = 0

    vi_tri = 0
    for co_lo in _chia_lo(len(todo), backend.batch_size):
        if should_cancel():
            raise JobCancelledError()

        lo = todo[vi_tri : vi_tri + co_lo]
        vi_tri += co_lo

        try:
            pcms = backend.synthesize_batch([seg.text_vi for _, seg in lo], voice_id)
        except QuotaExhaustedError:
            warnings.append(QuotaExhaustedError.user_message)
            failed += len(lo)
            continue
        except Exception as exc:
            log.warning("Lô %d lượt thoại hỏng (%s) — đọc lại từng câu một", len(lo), exc)
            pcms = []
            for _, seg in lo:
                try:
                    pcms.append(backend.synthesize(seg.text_vi, voice_id))
                except Exception as le:
                    log.warning("Không đọc được lượt thoại: %s", le)
                    pcms.append(b"")

        for (index, seg), pcm in zip(lo, pcms):
            xong += 1
            if not pcm:
                failed += 1
                continue
            try:
                samples, overflow = _fit_one(
                    seg, pcm, max_speedup,
                    _allowed_window(segments, index, total_duration, next_utterance_start),
                )
            except Exception as exc:
                log.warning("Không dựng được lượt thoại %d: %s", index, exc)
                failed += 1
                continue

            results[index] = samples
            # Phụ đề bám theo thời lượng đọc thật, không theo khung thời gian gốc.
            segments[index].spoken_duration = duration_of(samples)
            if overflow > _OVERFLOW_WARN_SECONDS:
                overflowed += 1

        progress("synthesize", (progress_base + xong) / max(1, grand_total),
                 f"Đang đọc lượt thoại {progress_base + xong}/{grand_total}")

    if failed:
        warnings.append(f"{failed}/{len(todo)} lượt thoại không tạo được giọng đọc và đã bị bỏ trống.")
    if overflowed:
        warnings.append(
            f"{overflowed} lượt thoại dài hơn cả khung gốc lẫn khoảng lặng kế tiếp dù đã "
            "tăng tốc — các câu ngay sau bị lùi lại một chút, không mất chữ nào."
        )

    fitted = [(segments[i], results[i]) for i in sorted(results)]
    progress("synthesize", min(1.0, (progress_base + len(todo)) / max(1, grand_total)),
             f"Đã tạo {progress_base + len(fitted)}/{grand_total} lượt thoại")
    return fitted, warnings
