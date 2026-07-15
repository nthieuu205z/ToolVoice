"""Thời gian đến từ vùng ffmpeg, nội dung đến từ Gemini. Test khóa đúng ranh giới đó."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import NoSpeechDetectedError
from pipeline.models import Segment
from pipeline.segmentation import Region
from pipeline.stt import merge_sentence_fragments, transcribe_regions
from tests.conftest import FakeGemini

RATE = 16000


def audio(seconds: float) -> np.ndarray:
    return np.full(int(RATE * seconds), 1000, dtype="<i2")


def test_merge_sentence_fragments_joins_open_fragments_into_full_sentences():
    """Mảnh không kết bằng dấu câu được nối với mảnh sau; câu trọn thì đứng riêng."""
    segs = [
        Segment(0.0, 2.0, "You're in the right place"),   # dang dở
        Segment(2.3, 4.0, "if you use Claude every day."),  # đóng câu
        Segment(5.0, 7.0, "Let's get started."),           # trọn vẹn
    ]
    merged = merge_sentence_fragments(segs, max_duration=12.0)
    assert len(merged) == 2
    assert merged[0].text == "You're in the right place if you use Claude every day."
    assert merged[0].start == 0.0 and merged[0].end == 4.0   # mốc thật: đầu mảnh đầu → cuối mảnh cuối
    assert merged[1].text == "Let's get started."


def test_merge_respects_max_duration_so_long_sentences_are_not_glued_forever():
    """Không gộp nếu vượt trần: câu dài >max_duration vẫn tách để tránh lượt vô hạn."""
    segs = [Segment(0.0, 8.0, "a long open fragment"), Segment(8.0, 18.0, "that keeps going")]
    merged = merge_sentence_fragments(segs, max_duration=12.0)   # gộp lại 18s > 12s
    assert len(merged) == 2


def test_segment_timing_comes_from_the_region_not_the_model(tmp_path):
    """Gemini không được hỏi giờ — trên Developer API nó bịa ra."""
    backend = FakeGemini(clips=["xin chào"])
    regions = [Region(12.5, 20.0)]

    _, segments, _ = transcribe_regions(backend, audio(30), RATE, regions, tmp_path)

    assert (segments[0].start, segments[0].end) == (12.5, 20.0)
    assert segments[0].text == "xin chào"


def test_regions_keep_their_order_despite_parallel_workers(tmp_path):
    backend = FakeGemini(clips=["một", "hai", "ba"])
    regions = [Region(0, 5), Region(10, 15), Region(20, 25)]

    _, segments, _ = transcribe_regions(backend, audio(30), RATE, regions, tmp_path, workers=3)

    assert [s.start for s in segments] == [0, 10, 20]


def test_language_is_reported(tmp_path):
    backend = FakeGemini(clips=["konnichiwa"], language="ja")
    language, _, _ = transcribe_regions(backend, audio(10), RATE, [Region(0, 5)], tmp_path)
    assert language == "ja"


def test_regions_with_no_speech_are_dropped(tmp_path):
    backend = FakeGemini(clips=["có tiếng", "   ", "cũng có"])
    regions = [Region(0, 5), Region(6, 8), Region(9, 12)]

    _, segments, _ = transcribe_regions(backend, audio(15), RATE, regions, tmp_path)

    assert [s.text for s in segments] == ["có tiếng", "cũng có"]


def test_a_transient_failure_is_recovered_on_the_second_pass(tmp_path):
    """8 luồng song song thỉnh thoảng dính 429 của Vertex — đo thật trên video 19 phút.

    Đoạn hỏng phải được thử lại tuần tự sau khi cơn dồn request dịu, vì mỗi đoạn
    bị bỏ là mất hẳn lời thoại đoạn đó trong video kết quả.
    """
    class FlakyOnce(FakeGemini):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.seen: set[str] = set()

        def transcribe_clip(self, wav_path):
            if wav_path.name not in self.seen:
                self.seen.add(wav_path.name)
                raise RuntimeError("429 thoáng qua")
            return super().transcribe_clip(wav_path)

    _, segments, warnings = transcribe_regions(
        FlakyOnce(clips=["một", "hai"]), audio(20), RATE,
        [Region(0, 5), Region(6, 10)], tmp_path, workers=2,
    )

    assert [s.text for s in segments] == ["một", "hai"]  # không mất đoạn nào
    assert warnings == []


def test_a_region_lost_twice_is_reported_not_silently_dropped(tmp_path):
    class AlwaysBroken(FakeGemini):
        def transcribe_clip(self, wav_path):
            result = super().transcribe_clip(wav_path)
            if wav_path.stem.endswith("0000"):
                raise RuntimeError("hỏng hẳn")
            return result

    _, segments, warnings = transcribe_regions(
        AlwaysBroken(clips=["mất", "còn sống"]), audio(20), RATE,
        [Region(0, 5), Region(6, 10)], tmp_path,
    )

    assert [s.text for s in segments] == ["còn sống"]
    assert any("thiếu lời thoại" in w for w in warnings)  # người dùng phải được biết


def test_no_regions_means_no_speech(tmp_path):
    with pytest.raises(NoSpeechDetectedError):
        transcribe_regions(FakeGemini(), audio(10), RATE, [], tmp_path)


def test_all_regions_silent_means_no_speech(tmp_path):
    with pytest.raises(NoSpeechDetectedError):
        transcribe_regions(FakeGemini(clips=[""]), audio(10), RATE, [Region(0, 5)], tmp_path)


def test_temporary_clip_files_are_cleaned_up(tmp_path):
    backend = FakeGemini(clips=["a"])
    transcribe_regions(backend, audio(10), RATE, [Region(0, 5)], tmp_path)
    assert list((tmp_path / "clips").glob("*.wav")) == []
