"""VieNeu là model ngẫu nhiên: ~1–2% lượt đọc rút phải "lá bài xấu" — audio dài bình
thường nhưng có lỗ hổng im lặng dài ở giữa (đo thật: lỗ 4,72s), nghe như "ngắt đột ngột".
Bộ chống chạy hoang cũ mù với ca này vì tổng độ dài vẫn hợp lý. Ở đây ta phát hiện lỗ
hổng bất thường rồi đọc lại (VieNeu chạy local, đọc lại không tốn gì)."""

from __future__ import annotations

import numpy as np

from pipeline.audio import longest_internal_silence, pcm_to_array
from pipeline.models import Segment, TTS_SAMPLE_RATE
from pipeline.tts import synthesize_segments
from tests.conftest import FakeGemini, sine_pcm, sine_pcm_with_hole

R = TTS_SAMPLE_RATE


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * R), dtype="<i2")


def segments(*specs: tuple[float, float, str]) -> list[Segment]:
    return [Segment(start, end, "src", vi) for start, end, vi in specs]


# ─── phát hiện lỗ hổng ───

def test_longest_internal_silence_measures_a_hole_between_speech():
    clip = np.concatenate([pcm_to_array(sine_pcm(1.0)), _silence(2.5), pcm_to_array(sine_pcm(1.0))])
    assert 2.2 < longest_internal_silence(clip) < 2.8


def test_leading_and_trailing_silence_are_not_counted_as_a_hole():
    clip = np.concatenate([_silence(1.0), pcm_to_array(sine_pcm(2.0)), _silence(1.5)])
    assert longest_internal_silence(clip) < 0.3


def test_continuous_speech_has_no_internal_hole():
    assert longest_internal_silence(pcm_to_array(sine_pcm(3.0))) < 0.3


def test_a_short_natural_pause_between_sentences_is_not_flagged():
    """Nghỉ 0,4s giữa hai câu là bình thường — không được coi là lỗ hổng."""
    clip = np.concatenate([pcm_to_array(sine_pcm(1.5)), _silence(0.4), pcm_to_array(sine_pcm(1.5))])
    assert longest_internal_silence(clip) < 1.3


# ─── đọc lại lượt xấu ───

def test_a_segment_with_an_abnormal_hole_is_resynthesized_and_the_clean_take_is_kept():
    backend = FakeGemini(tts_duration=3.0, hole_first_for={"lỗ hổng"})
    fitted, _ = synthesize_segments(
        backend, segments((0.0, 10.0, "lỗ hổng")), "Kore", workers=1,
        total_duration=30.0, resynthesize_bad=True,
    )
    assert backend.synthesize_calls.count(("lỗ hổng", "Kore")) == 2   # đọc lại đúng 1 lần
    assert longest_internal_silence(fitted[0][1]) < 1.3               # bản giữ lại đã sạch


def test_clean_segments_are_never_resynthesized():
    backend = FakeGemini(tts_duration=3.0)
    synthesize_segments(
        backend, segments((0.0, 10.0, "ổn")), "Kore", workers=1,
        total_duration=30.0, resynthesize_bad=True,
    )
    assert backend.synthesize_calls.count(("ổn", "Kore")) == 1


def test_resynthesis_is_off_by_default_so_metered_backends_are_not_billed_twice():
    """Gemini TTS tính tiền theo lượt — không đọc lại trừ khi runner bật (backend miễn phí)."""
    backend = FakeGemini(tts_duration=3.0, hole_first_for={"lỗ hổng"})
    synthesize_segments(
        backend, segments((0.0, 10.0, "lỗ hổng")), "Kore", workers=1, total_duration=30.0,
    )
    assert backend.synthesize_calls.count(("lỗ hổng", "Kore")) == 1
