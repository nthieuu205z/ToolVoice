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


# ─── tách CÂU từ mốc từng TỪ của Whisper ───

def test_sentences_from_words_splits_on_terminal_punctuation():
    from pipeline.whisper_stt import sentences_from_words
    words = [(0.0, 0.5, "Hello"), (0.6, 1.0, "there."),
             (1.5, 2.0, "How"), (2.1, 2.5, "are"), (2.6, 3.0, "you?")]
    assert sentences_from_words(words) == [
        (0.0, 1.0, "Hello there."), (1.5, 3.0, "How are you?")]


def test_sentences_from_words_keeps_a_trailing_open_fragment():
    """Câu cuối chưa có dấu kết vẫn được trả — merge_sentence_fragments sẽ ghép với vùng sau."""
    from pipeline.whisper_stt import sentences_from_words
    words = [(0.0, 0.5, "Third,"), (0.6, 1.0, "as"), (1.1, 1.5, "a"), (1.6, 2.0, "result")]
    assert sentences_from_words(words) == [(0.0, 2.0, "Third, as a result")]


def test_sentences_from_words_handles_no_words():
    from pipeline.whisper_stt import sentences_from_words
    assert sentences_from_words([]) == []


def test_sentences_from_words_treats_ellipsis_as_a_sentence_end():
    from pipeline.whisper_stt import sentences_from_words
    words = [(0.0, 0.5, "Wait…"), (1.0, 1.5, "Okay.")]
    assert sentences_from_words(words) == [(0.0, 0.5, "Wait…"), (1.0, 1.5, "Okay.")]


# ─── nhận diện cấp CÂU: giữ mốc thời gian của Whisper để đặt câu bám hình ───

class TimedRecognizer:
    """Whisper trên GPU trả về từng mảnh cấp CÂU kèm mốc thời gian thật (đã lọc trong vùng)."""

    stt_batch_size = 32
    stt_timed = True

    def __init__(self, pieces: list[tuple[float, float, str]], language: str = "en",
                 fail_timed: bool = False):
        self.pieces = pieces
        self.language = language
        self.fail_timed = fail_timed
        self.timed_calls = 0
        self.clip_calls = 0

    def transcribe_batch_timed(self, samples, rate, regions):
        self.timed_calls += 1
        if self.fail_timed:
            raise RuntimeError("lô mốc-câu hỏng")
        lo, hi = regions[0].start, regions[-1].end
        inside = [(s, e, t) for s, e, t in self.pieces if s < hi and e > lo]
        return self.language, inside

    def transcribe_clip(self, wav_path):
        self.clip_calls += 1
        return self.language, "cả vùng gộp một câu"


def _timed_backend(rec):
    from pipeline.backends import CompositeBackend
    return CompositeBackend(recognizer=rec, translator=None, synthesizer=None)


def test_timed_recognizer_emits_one_segment_per_sentence_not_per_region(tmp_path):
    """Một vùng ffmpeg 12s chứa nhiều câu → nhiều Segment theo mốc câu, không gộp một khối."""
    rec = TimedRecognizer([(0.0, 5.0, "Xin chào."), (6.0, 11.0, "Bạn khỏe không?")])
    lang, segments, _ = transcribe_regions(_timed_backend(rec), audio(12), RATE,
                                           [Region(0.0, 12.0)], tmp_path)

    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 5.0, "Xin chào."), (6.0, 11.0, "Bạn khỏe không?")]
    assert lang == "en"
    assert rec.clip_calls == 0   # không rơi về đường cũ


def test_timed_segments_keep_time_order_across_regions(tmp_path):
    rec = TimedRecognizer([(0.5, 3.0, "một"), (5.0, 7.0, "hai"), (13.0, 15.0, "ba")])
    _, segments, _ = transcribe_regions(_timed_backend(rec), audio(20), RATE,
                                        [Region(0.0, 8.0), Region(12.0, 16.0)], tmp_path)
    assert [s.text for s in segments] == ["một", "hai", "ba"]
    assert [s.start for s in segments] == [0.5, 5.0, 13.0]


def test_timed_path_falls_back_to_region_transcription_when_the_batch_fails(tmp_path):
    """Lô mốc-câu hỏng không giết cả video: lùi về chép cả vùng (mất mốc con, vẫn có lời)."""
    rec = TimedRecognizer([(0.0, 5.0, "x")], fail_timed=True)
    _, segments, _ = transcribe_regions(_timed_backend(rec), audio(12), RATE,
                                        [Region(0.0, 12.0)], tmp_path)
    assert rec.clip_calls == 1                       # đã lùi về chép cả vùng
    assert [(s.start, s.end) for s in segments] == [(0.0, 12.0)]
    assert segments[0].text == "cả vùng gộp một câu"


def test_timed_path_with_no_pieces_raises_no_speech(tmp_path):
    rec = TimedRecognizer([])
    with pytest.raises(NoSpeechDetectedError):
        transcribe_regions(_timed_backend(rec), audio(12), RATE, [Region(0.0, 12.0)], tmp_path)


def test_composite_backend_forwards_the_timed_capability():
    from pipeline.backends import CompositeBackend
    rec = TimedRecognizer([(0.0, 2.0, "a")])
    b = CompositeBackend(recognizer=rec, translator=None, synthesizer=None)
    assert b.stt_timed is True
    lang, pieces = b.transcribe_batch_timed(audio(3), RATE, [Region(0.0, 3.0)])
    assert lang == "en" and pieces == [(0.0, 2.0, "a")]


def test_composite_backend_reports_no_timed_capability_for_plain_recognizers():
    from pipeline.backends import CompositeBackend
    b = CompositeBackend(recognizer=FakeGemini(), translator=None, synthesizer=None)
    assert b.stt_timed is False


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
