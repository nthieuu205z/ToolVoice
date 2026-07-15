"""Mốc thời gian phải đến từ ffmpeg. Đây là các hàm thuần quyết định khung của mỗi lượt thoại."""

from __future__ import annotations

import numpy as np

from pipeline.segmentation import (
    MIN_SPEECH_DURATION,
    Region,
    invert_silences,
    merge_regions,
    parse_silences,
    split_long_regions,
)

STDERR = """
[silencedetect @ 0x1] silence_start: 13.37
[silencedetect @ 0x1] silence_end: 13.79 | silence_duration: 0.42
[silencedetect @ 0x1] silence_start: 25.00
[silencedetect @ 0x1] silence_end: 26.27 | silence_duration: 1.27
"""


def test_parse_silences_reads_start_end_pairs():
    assert parse_silences(STDERR) == [(13.37, 13.79), (25.0, 26.27)]


def test_parse_silences_ignores_trailing_unclosed_start():
    """ffmpeg kết thúc file khi đang trong khoảng lặng → thừa một silence_start."""
    assert len(parse_silences(STDERR + "silence_start: 99.0\n")) == 2


def test_silences_invert_into_speech_regions():
    regions = invert_silences([(13.37, 13.79), (25.0, 26.27)], total_duration=40.0)
    assert [(round(r.start, 2), round(r.end, 2)) for r in regions] == [
        (0.0, 13.37), (13.79, 25.0), (26.27, 40.0)
    ]


def test_leading_silence_is_not_speech():
    regions = invert_silences([(0.0, 0.79)], total_duration=10.0)
    assert regions == [Region(0.79, 10.0)]


def test_audio_that_is_entirely_silent_yields_no_regions():
    assert invert_silences([(0.0, 10.0)], total_duration=10.0) == []


def test_slivers_between_silences_are_discarded():
    """Mẩu 0.1s kẹp giữa hai khoảng lặng là tiếng động, không phải lời nói."""
    regions = invert_silences([(0.0, 5.0), (5.1, 10.0)], total_duration=10.0)
    assert regions == []


def test_adjacent_regions_merge_when_the_pause_is_a_breath():
    merged = merge_regions([Region(0, 5), Region(5.3, 9)], max_gap=0.5, max_duration=30)
    assert merged == [Region(0, 9)]


def test_regions_stay_separate_across_a_real_pause():
    merged = merge_regions([Region(0, 5), Region(6.0, 9)], max_gap=0.5, max_duration=30)
    assert merged == [Region(0, 5), Region(6.0, 9)]


def test_merging_stops_at_the_duration_ceiling():
    """Trần thời lượng giữ cho một lượt không nuốt cả video khi người nói liên tục."""
    regions = [Region(i * 5.0, i * 5.0 + 4.9) for i in range(10)]
    merged = merge_regions(regions, max_gap=0.5, max_duration=20.0)
    assert all(r.duration <= 20.0 for r in merged)
    assert len(merged) > 1


def test_merge_of_empty_input_is_empty():
    assert merge_regions([]) == []


def _speech_with_quiet_middle(rate: int = 16000) -> np.ndarray:
    """40 giây: ồn – im ở giây 20 – ồn."""
    loud = np.full(rate * 20, 8000, dtype="<i2")
    quiet = np.zeros(rate, dtype="<i2")
    return np.concatenate([loud, quiet, loud])


def test_long_region_is_split_at_its_quietest_point():
    rate = 16000
    samples = _speech_with_quiet_middle(rate)
    parts = split_long_regions([Region(0.0, 41.0)], samples, rate, max_duration=30.0)

    assert len(parts) == 2
    cut = parts[0].end
    assert 20.0 <= cut <= 21.0, f"điểm cắt {cut} phải rơi vào khoảng lặng ở giữa"


def test_splitting_recurses_until_every_part_fits():
    rate = 8000
    samples = np.full(rate * 100, 5000, dtype="<i2")
    parts = split_long_regions([Region(0.0, 100.0)], samples, rate, max_duration=30.0)
    assert parts
    assert all(p.duration <= 30.0 for p in parts)


def test_split_preserves_total_coverage():
    rate = 8000
    samples = np.full(rate * 100, 5000, dtype="<i2")
    parts = split_long_regions([Region(0.0, 100.0)], samples, rate, max_duration=30.0)
    assert parts[0].start == 0.0
    assert parts[-1].end == 100.0
    for prev, nxt in zip(parts, parts[1:]):
        assert prev.end == nxt.start


def test_short_region_is_left_alone():
    samples = np.zeros(16000 * 5, dtype="<i2")
    regions = [Region(0.0, 5.0)]
    assert split_long_regions(regions, samples, 16000, max_duration=30.0) == regions


def test_split_never_produces_slivers():
    rate = 8000
    samples = np.full(int(rate * 30.4), 5000, dtype="<i2")
    parts = split_long_regions([Region(0.0, 30.4)], samples, rate, max_duration=30.0)
    assert all(p.duration >= MIN_SPEECH_DURATION for p in parts)
