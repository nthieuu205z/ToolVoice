from __future__ import annotations

import numpy as np
import pytest

from pipeline.assembly import INTER_UTTERANCE_GAP, build_audio_track, plan_placement
from pipeline.audio import (
    duration_of,
    finalize_track,
    mix_into,
    pcm_to_array,
    read_wav,
    silent_track,
)
from pipeline.models import Segment, TTS_SAMPLE_RATE

RATE = TTS_SAMPLE_RATE


def tone(duration: float, value: int = 1000) -> np.ndarray:
    return np.full(int(duration * RATE), value, dtype="<i2")


def _seg(start: float, end: float) -> Segment:
    return Segment(start, end, "src", "vi")


# ─── xếp chỗ: không bao giờ để hai lượt thoại đè nhau ───

def test_placement_keeps_original_anchors_when_nothing_overlaps():
    fitted = [(_seg(0.0, 2.0), tone(1.0)), (_seg(5.0, 7.0), tone(1.0))]
    placed = plan_placement(fitted)
    assert [start for start, _ in placed] == [0.0, 5.0]
    assert fitted[1][0].placed_start == 5.0


def test_placement_pushes_the_next_line_back_instead_of_overlapping():
    """Lượt trước tràn khung → lượt sau lùi lại chờ, không bao giờ hai giọng cùng lúc."""
    fitted = [(_seg(0.0, 2.0), tone(5.0)), (_seg(2.0, 4.0), tone(1.0))]
    placed = plan_placement(fitted)

    assert placed[1][0] == pytest.approx(5.0 + INTER_UTTERANCE_GAP)
    assert fitted[1][0].placed_start == pytest.approx(5.0 + INTER_UTTERANCE_GAP)


def test_the_delay_dissolves_at_the_next_silence():
    """Bị lùi không có nghĩa lùi mãi: gặp khoảng lặng đủ dài là về đúng mốc gốc."""
    fitted = [
        (_seg(0.0, 2.0), tone(5.0)),    # tràn tới giây 5
        (_seg(2.0, 4.0), tone(1.0)),    # bị lùi tới ~5.12
        (_seg(20.0, 22.0), tone(1.0)),  # xa hẳn — không liên can
    ]
    placed = plan_placement(fitted)
    assert placed[2][0] == 20.0


def test_a_chain_of_overflows_cascades_without_ever_overlapping():
    fitted = [
        (_seg(0.0, 1.0), tone(3.0)),
        (_seg(1.0, 2.0), tone(3.0)),
        (_seg(2.0, 3.0), tone(3.0)),
    ]
    placed = plan_placement(fitted)
    for (start_a, samples_a), (start_b, _) in zip(placed, placed[1:]):
        assert start_b >= start_a + duration_of(samples_a)  # kết thúc trước, bắt đầu sau


def test_pcm_to_array_drops_trailing_odd_byte():
    assert len(pcm_to_array(b"\x01\x02\x03")) == 1


def test_track_length_matches_video_duration(tmp_path):
    out = build_audio_track([(0.0, tone(1.0))], total_duration=5.0, out_wav=tmp_path / "a.wav")
    samples, rate = read_wav(out)
    assert rate == RATE
    assert abs(duration_of(samples, rate) - 5.0) < 0.001


def test_segment_lands_at_its_start_timestamp(tmp_path):
    out = build_audio_track([(2.0, tone(1.0))], total_duration=4.0, out_wav=tmp_path / "a.wav")
    samples, _ = read_wav(out)
    assert samples[int(0.5 * RATE)] == 0        # trước lời thoại: im lặng
    assert samples[int(2.5 * RATE)] == 1000     # trong lời thoại
    assert samples[int(3.5 * RATE)] == 0        # sau lời thoại: im lặng


def test_gaps_between_segments_stay_silent(tmp_path):
    out = build_audio_track(
        [(0.0, tone(0.5)), (2.0, tone(0.5))], total_duration=3.0, out_wav=tmp_path / "a.wav"
    )
    samples, _ = read_wav(out)
    assert samples[int(1.0 * RATE)] == 0


def test_overflowing_segment_mixes_instead_of_truncating_the_next():
    """Lời thoại tràn khung phải cộng vào lời kế tiếp, không được xén mất nó."""
    track = silent_track(3.0)
    mix_into(track, tone(2.0, 1000), 0.0)   # tràn tới giây thứ 2
    mix_into(track, tone(1.0, 500), 1.0)    # bắt đầu từ giây 1
    final = finalize_track(track)
    assert final[int(1.5 * RATE)] == 1500


def test_mixing_clips_at_int16_bounds_instead_of_wrapping():
    track = silent_track(1.0)
    for _ in range(4):
        mix_into(track, tone(1.0, 30000), 0.0)
    assert finalize_track(track).max() == 32767


def test_segment_past_track_end_is_ignored():
    track = silent_track(1.0)
    mix_into(track, tone(1.0), 5.0)
    assert finalize_track(track).max() == 0


def test_segment_is_truncated_at_track_end():
    track = silent_track(1.0)
    mix_into(track, tone(2.0), 0.5)
    assert len(track) == RATE


def test_empty_segment_list_produces_silent_track_of_right_length(tmp_path):
    out = build_audio_track([], total_duration=2.0, out_wav=tmp_path / "a.wav")
    samples, _ = read_wav(out)
    assert len(samples) == 2 * RATE
    assert samples.max() == 0
