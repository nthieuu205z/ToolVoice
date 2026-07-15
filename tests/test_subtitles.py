from __future__ import annotations

import srt

from pipeline.models import Segment
from pipeline.subtitles import (
    MAX_CUE_CHARS,
    MAX_CUE_SECONDS,
    build_srt,
    segment_to_cues,
    split_into_cues,
    wrap_text,
)


def parse(content: str) -> list[srt.Subtitle]:
    return list(srt.parse(content))


# ─── chẻ một lượt phát ngôn dài thành nhiều phụ đề ───────────────

def test_short_utterance_stays_one_cue():
    assert split_into_cues("Xin chào các bạn.") == ["Xin chào các bạn."]


def test_long_utterance_is_split_at_sentence_boundaries():
    text = ("Đây là câu thứ nhất và nó khá là dài. Đây là câu thứ hai cũng dài không kém. "
            "Còn đây là câu thứ ba để cho đủ bộ ba.")
    cues = split_into_cues(text)
    assert len(cues) > 1
    assert all(len(c) <= MAX_CUE_CHARS for c in cues)
    assert all(c.endswith(".") for c in cues)


def test_a_single_sentence_too_long_falls_back_to_clause_breaks():
    text = "Khi bạn viết chương trình Java, bạn phải khai báo lớp, sau đó khai báo phương thức main, rồi mới viết lệnh"
    cues = split_into_cues(text)
    assert len(cues) > 1
    assert all(len(c) <= MAX_CUE_CHARS for c in cues)


def test_text_with_no_punctuation_is_still_split():
    text = " ".join(["từ"] * 60)
    cues = split_into_cues(text)
    assert all(len(c) <= MAX_CUE_CHARS for c in cues)


def test_splitting_never_loses_words():
    text = ("Câu một khá dài để phải chẻ ra. Câu hai cũng vậy thôi bạn ạ. "
            "Câu ba là câu cuối cùng của đoạn này.")
    joined = " ".join(split_into_cues(text))
    assert joined.split() == text.split()


def test_cue_durations_are_shared_out_by_character_count():
    """Lượt 20s, hai câu dài xấp xỉ bằng nhau → mỗi câu chiếm khoảng 10s."""
    half = "Câu này được viết dài ra cho vượt quá trần tám mươi tư ký tự của một cue phụ đề."
    seg = Segment(0.0, 20.0, "src", f"{half} {half}")
    cues = segment_to_cues(seg)

    assert len(cues) == 2
    assert abs((cues[0][1] - cues[0][0]) - 10.0) < 0.5
    assert cues[0][1] == cues[1][0]  # liền mạch, không hở không chồng


def test_cues_span_exactly_the_utterance_window():
    seg = Segment(5.0, 25.0, "src", "Một câu. Hai câu. Ba câu.")
    cues = segment_to_cues(seg)
    assert cues[0][0] == 5.0
    assert abs(cues[-1][1] - 25.0) < 0.001


def test_cues_follow_the_actual_placement_not_the_theoretical_anchor():
    """Lượt bị lùi lại (tránh đè lượt trước) thì phụ đề phải lùi theo tiếng nói thật."""
    seg = Segment(2.0, 5.0, "src", "Câu bị lùi lại một chút.")
    seg.spoken_duration = 2.0
    seg.placed_start = 4.0

    cues = segment_to_cues(seg)
    assert cues[0][0] == 4.0


def test_cues_follow_the_spoken_duration_not_the_original_window():
    """Giọng Việt đọc xong sớm hơn khung gốc; trải cue theo khung sẽ đẩy cue cuối vào im lặng."""
    seg = Segment(0.0, 30.0, "src", "Một câu. Hai câu.")
    seg.spoken_duration = 12.0

    cues = segment_to_cues(seg)

    assert abs(cues[-1][1] - 12.0) < 0.001  # kết thúc cùng lúc giọng đọc dứt


def test_cues_fall_back_to_the_window_when_speech_was_never_generated():
    """Lượt hỏng không có giọng đọc — phụ đề vẫn phải hiện, theo khung gốc."""
    seg = Segment(0.0, 8.0, "src", "Một câu.")
    assert seg.spoken_duration == 0.0
    assert abs(segment_to_cues(seg)[-1][1] - 8.0) < 0.001


def test_wrap_keeps_short_text_on_one_line():
    assert wrap_text("Xin chào các bạn") == "Xin chào các bạn"


def test_wrap_splits_long_text_into_two_lines():
    text = "Đây là một câu thoại rất dài cần được ngắt thành hai dòng cho dễ đọc"
    assert wrap_text(text).count("\n") == 1


def test_wrap_never_drops_words():
    text = " ".join(f"từ{i}" for i in range(40))
    wrapped = wrap_text(text)
    assert wrapped.replace("\n", " ").split() == text.split()


def test_short_cue_is_extended_to_minimum_duration():
    subs = parse(build_srt([Segment(0.0, 0.2, "hi", "chào")]))
    assert subs[0].end.total_seconds() >= 1.0


def test_long_cue_is_clamped_to_maximum_duration():
    subs = parse(build_srt([Segment(0.0, 30.0, "x", "dài")]))
    assert subs[0].end.total_seconds() <= MAX_CUE_SECONDS


def test_a_thirty_second_utterance_becomes_several_readable_cues():
    """Lượt thoại dài không được nhồi hết vào một phụ đề duy nhất."""
    text = " ".join(f"Đây là câu số {i} trong một lượt thoại dài." for i in range(6))
    subs = parse(build_srt([Segment(0.0, 30.0, "src", text)]))

    assert len(subs) >= 3
    for sub in subs:
        assert len(sub.content.replace("\n", " ")) <= MAX_CUE_CHARS
    joined = " ".join(sub.content.replace("\n", " ") for sub in subs)
    assert joined.split() == text.split()  # không mất chữ nào


def test_cues_never_overlap_after_minimum_duration_padding():
    """Cue đầu bị kéo dài tới 1s sẽ đè lên cue thứ hai bắt đầu ở 0.5s nếu không cắt."""
    subs = parse(build_srt([
        Segment(0.0, 0.2, "a", "một"),
        Segment(0.5, 2.0, "b", "hai"),
    ]))
    assert subs[0].end <= subs[1].start


def test_segments_are_sorted_and_renumbered():
    subs = parse(build_srt([
        Segment(5.0, 6.0, "b", "sau"),
        Segment(1.0, 2.0, "a", "trước"),
    ]))
    assert [s.index for s in subs] == [1, 2]
    assert subs[0].content == "trước"


def test_empty_translation_falls_back_to_source_text():
    subs = parse(build_srt([Segment(0.0, 2.0, "hello", "")]))
    assert subs[0].content == "hello"


def test_blank_segments_are_dropped():
    assert build_srt([Segment(0.0, 2.0, "   ", "   ")]) == ""
