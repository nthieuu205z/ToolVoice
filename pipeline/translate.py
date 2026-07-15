"""Bước dịch: lời thoại gốc → tiếng Việt, giữ nguyên số dòng và thứ tự.

Số dòng phải khớp tuyệt đối: lệch một dòng là toàn bộ lời thoại phía sau lệch giờ,
nên thà báo lỗi to còn hơn xuất ra video sai tiếng.
"""

from __future__ import annotations

from .errors import JobCancelledError, TranslationAlignmentError
from .models import CancelFn, GeminiBackend, ProgressFn, Segment, never_cancel, noop_progress

# Dịch theo lô để câu lệnh không vượt giới hạn token đầu ra của model.
BATCH_SIZE = 80
# Số dòng đã dịch gần nhất đưa vào làm ngữ cảnh cho lô kế tiếp.
CONTEXT_LINES = 3


def translate_segments(
    backend: GeminiBackend,
    segments: list[Segment],
    progress: ProgressFn = noop_progress,
    should_cancel: CancelFn = never_cancel,
) -> list[Segment]:
    """Điền `text_vi` cho từng lời thoại. Trả về chính danh sách đã truyền vào."""
    total = len(segments)
    for offset in range(0, total, BATCH_SIZE):
        if should_cancel():
            raise JobCancelledError()
        batch = segments[offset : offset + BATCH_SIZE]
        progress("translate", offset / total, f"Đang dịch {offset + 1}–{min(offset + len(batch), total)}/{total}")

        context = _context_from(segments[:offset])
        translations = _translate_batch(backend, batch, context)
        for seg, text_vi in zip(batch, translations):
            seg.text_vi = text_vi

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
    tail = [seg for seg in previous[-CONTEXT_LINES:] if seg.text_vi]
    return "\n".join(f"- {seg.text_vi}" for seg in tail)
