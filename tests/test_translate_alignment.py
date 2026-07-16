"""Lệch số dòng dịch = lệch giờ toàn bộ lời thoại phía sau, nên phải bắt bằng test."""

from __future__ import annotations

import pytest

from pipeline.errors import TranslationAlignmentError
from pipeline.models import Segment
from pipeline.translate import translate_segments
from tests.conftest import FakeGemini


def segments(n: int) -> list[Segment]:
    return [Segment(float(i), float(i) + 1.0, f"line {i}") for i in range(n)]


def test_the_prompt_forbids_dropping_content():
    """"Thiếu câu hay chữ" từng có thể xảy ra vì prompt bảo model rút gọn cho vừa khung.

    Giờ khung thời gian chỉ là gợi ý cách diễn đạt; đủ ý là bắt buộc — phần dài ra
    đã có cơ chế mượn khoảng lặng + tăng tốc lo.
    """
    from pipeline.gemini import _TRANSLATE_PROMPT

    assert "không lược bỏ" in _TRANSLATE_PROMPT
    assert "luôn chọn đủ ý" in _TRANSLATE_PROMPT


def test_translations_land_on_matching_segments_in_order():
    backend = FakeGemini(translations=[["một", "hai", "ba"]])
    result = translate_segments(backend, segments(3))
    assert [s.text_vi for s in result] == ["một", "hai", "ba"]


def test_durations_are_passed_so_model_can_budget_line_length():
    backend = FakeGemini(translations=[["một"]])
    translate_segments(backend, [Segment(0.0, 2.5, "hello")])
    _, durations, _ = backend.translate_calls[0]
    assert durations == [2.5]


def test_wrong_line_count_triggers_exactly_one_retry_then_succeeds():
    backend = FakeGemini(translations=[["chỉ một dòng"], ["một", "hai"]])
    result = translate_segments(backend, segments(2))
    assert len(backend.translate_calls) == 2
    assert [s.text_vi for s in result] == ["một", "hai"]


def test_persistent_misalignment_raises_rather_than_desyncing_audio():
    backend = FakeGemini(translations=[["chỉ một dòng"]])
    with pytest.raises(TranslationAlignmentError):
        translate_segments(backend, segments(2))
    assert len(backend.translate_calls) == 2  # thử lại đúng một lần rồi bỏ cuộc


def test_blank_line_counts_as_misalignment():
    """Dòng rỗng nghĩa là model bỏ sót một index — im lặng ở đó sẽ lệch tiếng."""
    backend = FakeGemini(translations=[["một", "  "], ["một", "hai"]])
    translate_segments(backend, segments(2))
    assert len(backend.translate_calls) == 2


def test_long_transcript_is_translated_in_batches():
    from pipeline.translate import BATCH_SIZE

    total = BATCH_SIZE + 10
    backend = FakeGemini()
    translate_segments(backend, segments(total))

    # Các lô chạy SONG SONG nên thứ tự gọi không cố định — kiểm theo tập kích thước.
    assert len(backend.translate_calls) == 2
    assert sorted(len(c[0]) for c in backend.translate_calls) == [10, BATCH_SIZE]


def test_batches_take_the_preceding_SOURCE_lines_as_context_so_they_are_independent():
    """Ngữ cảnh lấy từ lời GỐC (English), không từ bản dịch — nhờ vậy các lô độc lập,
    dịch song song được. Model vẫn thấy nguyên văn nội dung ngay trước để dịch nhất quán.
    """
    from pipeline.translate import BATCH_SIZE

    backend = FakeGemini()
    translate_segments(backend, segments(BATCH_SIZE + 1))
    first = next(c for c in backend.translate_calls if len(c[0]) == BATCH_SIZE)   # lô đầu
    later = next(c for c in backend.translate_calls if len(c[0]) == 1)            # lô sau
    assert first[2] == ""                     # lô đầu: không có gì trước nó
    assert "line" in later[2]                 # lô sau: có lời gốc ngay trước
    assert "[vi]" not in later[2]             # KHÔNG phải bản dịch (nếu là dịch sẽ có "[vi]")


def test_batches_are_translated_in_parallel_and_results_stay_in_order():
    """Song song để nhanh (đo thật: 64s tuần tự → 11s song song), nhưng KẾT QUẢ phải đúng
    thứ tự segment — lệch một dòng là lệch giờ toàn bộ phía sau."""
    import threading
    import time

    from pipeline.translate import BATCH_SIZE

    class Probe(FakeGemini):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.active = 0
            self.peak = 0
            self._lk = threading.Lock()

        def translate(self, texts, durations, context=""):
            with self._lk:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.15)
            try:
                return super().translate(texts, durations, context)
            finally:
                with self._lk:
                    self.active -= 1

    backend = Probe()
    result = translate_segments(backend, segments(BATCH_SIZE * 3), workers=3)

    assert backend.peak >= 2                                        # thật sự chạy song song
    assert [s.text_vi for s in result[:3]] == ["[vi] line 0", "[vi] line 1", "[vi] line 2"]
    assert result[-1].text_vi == f"[vi] line {BATCH_SIZE * 3 - 1}"  # cuối cùng vẫn đúng chỗ
