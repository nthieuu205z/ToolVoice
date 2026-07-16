"""Bước dịch: lời thoại gốc → tiếng Việt, giữ nguyên số dòng và thứ tự.

Số dòng phải khớp tuyệt đối: lệch một dòng là toàn bộ lời thoại phía sau lệch giờ,
nên thà báo lỗi to còn hơn xuất ra video sai tiếng.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .errors import JobCancelledError, TranslationAlignmentError
from .models import CancelFn, GeminiBackend, ProgressFn, Segment, never_cancel, noop_progress

# Dịch theo lô để câu lệnh không vượt giới hạn token đầu ra của model. Đo thật: mỗi lượt
# gọi bị trói bởi SỐ TOKEN XUẤT (một dòng dịch = một dòng ra), nên lô 80 câu tốn ~20s.
# Lô nhỏ hơn (48) tốn ~11s/lô và chạy song song được nhiều lô hơn — tổng nhanh hơn hẳn,
# lại ít rủi ro lệch số dòng.
BATCH_SIZE = 48
# Số dòng gốc ngay trước đưa vào làm ngữ cảnh cho lô. Lấy từ LỜI GỐC (không phải bản dịch)
# nên mỗi lô độc lập, dịch song song được.
CONTEXT_LINES = 3
# Số lô dịch chạy song song. Nút cổ chai là độ trễ mạng/model xuất token, không phải CPU —
# đo thật: 4 lô tuần tự 63,7s → song song còn ~11s. Vertex chịu được vài request đồng thời;
# hạ xuống nếu dùng Developer API free tier (RPM thấp).
TRANSLATE_WORKERS = 6


def translate_segments(
    backend: GeminiBackend,
    segments: list[Segment],
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
    workers: int = TRANSLATE_WORKERS,
) -> list[Segment]:
    """Điền `text_vi` cho từng lời thoại. Trả về chính danh sách đã truyền vào.

    Các lô dịch SONG SONG (ngữ cảnh lấy từ lời gốc nên độc lập nhau). Kết quả ghép lại đúng
    thứ tự segment — lệch một dòng là lệch giờ toàn bộ phía sau. Một lô lệch số dòng vẫn
    ném lỗi to như cũ (không nuốt lỗi để rồi xuất video sai tiếng).
    """
    total = len(segments)
    if total == 0:
        return segments
    if should_cancel():   # hủy trước khi phóng lô nào — không tiêu tốn lượt gọi vô ích
        raise JobCancelledError()

    batches = [(offset, segments[offset : offset + BATCH_SIZE])
               for offset in range(0, total, BATCH_SIZE)]
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_translate_batch, backend, batch, _context_from(segments[:offset])):
                (offset, batch)
            for offset, batch in batches
        }
        for future in as_completed(futures):
            if should_cancel():
                pool.shutdown(wait=False, cancel_futures=True)
                raise JobCancelledError()
            _offset, batch = futures[future]
            translations = future.result()   # lệch số dòng → ném lỗi, dừng cả bước dịch
            for seg, text_vi in zip(batch, translations):
                seg.text_vi = text_vi
            done += len(batch)
            progress("translate", done / total, f"Đang dịch {done}/{total}")

    progress("translate", 1.0, f"Đã dịch {total} lời thoại")
    return segments


def _translate_batch(backend: GeminiBackend, batch: list[Segment], context: str) -> list[str]:
    """Gọi model, kiểm tra khớp số dòng, thử lại một lần trước khi bỏ cuộc."""
    texts = [seg.text for seg in batch]
    durations = [seg.duration for seg in batch]

    for attempt in (1, 2):
        result = backend.translate(texts, durations, context)
        if _is_aligned(result, len(batch)):
            return result
        if attempt == 2:
            raise TranslationAlignmentError(
                f"Cần {len(batch)} dòng dịch nhưng nhận về {len(result)} dòng dùng được"
            )
    raise AssertionError("không tới được")  # pragma: no cover


def _is_aligned(result: list[str], expected: int) -> bool:
    """Đúng số dòng và không dòng nào rỗng — dòng rỗng nghĩa là model bỏ sót index."""
    return len(result) == expected and all(text.strip() for text in result)


def _context_from(previous: list[Segment]) -> str:
    """Ngữ cảnh cho một lô = vài dòng LỜI GỐC ngay trước nó.

    Lấy lời gốc (không phải bản dịch) để lô nào cũng tính được ngữ cảnh TRƯỚC khi dịch —
    nhờ vậy mọi lô chạy song song. Model vẫn thấy nguyên văn nội dung ngay trước để giữ
    mạch và thuật ngữ nhất quán trong khi dịch lô này.
    """
    tail = [seg for seg in previous[-CONTEXT_LINES:] if seg.text.strip()]
    return "\n".join(f"- {seg.text.strip()}" for seg in tail)
