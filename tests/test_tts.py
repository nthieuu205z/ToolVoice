from __future__ import annotations

from pipeline.audio import duration_of, fit_to_slot, fit_to_window, parse_pcm_rate, pcm_to_array
from pipeline.errors import QuotaExhaustedError
from pipeline.models import Segment
from pipeline.tts import synthesize_segments
from tests.conftest import FakeGemini, sine_pcm


def segments(*specs: tuple[float, float, str]) -> list[Segment]:
    return [Segment(start, end, "src", vi) for start, end, vi in specs]


def test_each_segment_is_synthesized_with_the_chosen_voice():
    backend = FakeGemini(tts_duration=0.5)
    fitted, warnings = synthesize_segments(backend, segments((0.0, 2.0, "chào")), "Kore", workers=1)

    assert backend.synthesize_calls == [("chào", "Kore")]
    assert [seg.start for seg, _ in fitted] == [0.0]
    assert warnings == []


def test_untranslated_segments_are_skipped():
    backend = FakeGemini(tts_duration=0.5)
    fitted, _ = synthesize_segments(
        backend, segments((0.0, 2.0, "chào"), (3.0, 4.0, "  ")), "Kore", workers=1
    )
    assert len(backend.synthesize_calls) == 1
    assert len(fitted) == 1


def test_results_are_ordered_by_segment_even_when_workers_finish_out_of_order():
    backend = FakeGemini(tts_duration=0.5)
    fitted, _ = synthesize_segments(
        backend, segments((0.0, 2.0, "a"), (5.0, 7.0, "b"), (9.0, 11.0, "c")), "Kore", workers=3
    )
    assert [seg.start for seg, _ in fitted] == [0.0, 5.0, 9.0]


def test_failed_segment_is_not_retried_blindly_and_is_left_silent():
    """Gọi lại mù ở tầng này từng nhân đôi mọi request và làm cạn hạn mức nhanh gấp đôi.

    Việc thử lại thuộc về tenacity bên trong GeminiRunner, không phải ở đây.
    """
    backend = FakeGemini(tts_duration=0.5, fail_tts_for={"hỏng"})
    fitted, warnings = synthesize_segments(
        backend, segments((0.0, 2.0, "hỏng"), (3.0, 5.0, "tốt")), "Kore", workers=1
    )

    assert len(fitted) == 1                                       # câu tốt vẫn ra
    assert backend.synthesize_calls.count(("hỏng", "Kore")) == 1  # đúng MỘT lần
    assert any("không tạo được giọng đọc" in w for w in warnings)


def test_daily_quota_exhaustion_is_reported_as_its_own_warning():
    backend = FakeGemini(tts_duration=0.5, tts_error=QuotaExhaustedError())
    fitted, warnings = synthesize_segments(
        backend, segments((0.0, 2.0, "a"), (3.0, 5.0, "b")), "Kore", workers=1
    )

    assert fitted == []
    assert any("hết hạn mức" in w.lower() for w in warnings)


def test_quota_exhaustion_does_not_raise_so_partial_output_survives():
    """Giữ lại những câu đã đọc được + file .srt, thay vì vứt cả job."""
    backend = FakeGemini(tts_duration=0.5, tts_error=QuotaExhaustedError())
    fitted, warnings = synthesize_segments(backend, segments((0.0, 2.0, "a")), "Kore", workers=1)
    assert warnings  # không ném exception


def test_no_translated_lines_returns_a_warning_not_a_crash():
    fitted, warnings = synthesize_segments(FakeGemini(), segments((0.0, 2.0, "")), "Kore")
    assert fitted == []
    assert warnings


# ─── mượn khoảng lặng trước, tăng tốc sau ───

def test_long_audio_gets_a_gentle_speedup_then_borrows_the_silence():
    """Tiếng Việt dài hơn khung gốc: tăng tốc NHẸ (≤1,15 — tai không nhận ra) đưa về
    sát khung, phần còn dư tràn tự nhiên vào khoảng lặng phía sau. Không bao giờ
    nhảy thẳng lên 1,5× khi khoảng lặng còn rộng."""
    backend = FakeGemini(tts_duration=3.0)   # dài hơn khung 2 giây
    segs = segments((0.0, 2.0, "Một câu đủ dài để guard chống chạy hoang không đụng vào nó chút nào."), (6.0, 8.0, "sau"))
    fitted, warnings = synthesize_segments(backend, segs, "Kore", workers=1, total_duration=10.0)

    by_start = {seg.start: samples for seg, samples in fitted}
    spoken = duration_of(by_start[0.0])
    assert abs(spoken - 3.0 / 1.15) < 0.1    # đúng nấc nhẹ, không phải 1,5×
    assert spoken > 2.0                       # phần dư mượn khoảng lặng, không ép vào khung
    assert warnings == []


def test_speedup_kicks_in_only_when_even_the_silence_is_not_enough():
    """Hết cả chỗ mượn mới tăng tốc; vẫn dư thì báo rõ là KHÔNG mất chữ."""
    backend = FakeGemini(tts_duration=8.0)
    segs = segments((0.0, 2.0, "Một câu rất dài, đủ để tám giây âm thanh vẫn nằm dưới trần chống chạy hoang, vì trần được tính theo số ký tự của chính câu này."), (4.0, 5.0, ""))   # câu kế chặn cửa sổ ở giây 4
    fitted, warnings = synthesize_segments(backend, segs, "Kore", workers=1, total_duration=20.0)

    assert duration_of(fitted[0][1]) < 8.0 * 0.7            # đã tăng tốc ~1,5×
    assert any("không mất chữ" in w for w in warnings)


def test_lower_max_speedup_reads_a_dense_utterance_more_gently():
    """Trần tăng tốc quyết định mức 'nói nhanh' tối đa: cap thấp → đọc êm hơn, đổi lại tràn nhiều hơn.

    Câu dày (audio 6s), khung 3s và KHÔNG có khoảng lặng để mượn (window = khung): cap 1,5×
    nén về ~4,0s (nhanh), cap 1,3× chỉ về ~4,6s (êm hơn) và phần dư tràn ra để lùi câu sau.
    """
    from pipeline.audio import fit_to_window

    samples = pcm_to_array(sine_pcm(6.0))
    at_15, ov_15 = fit_to_window(samples, slot_duration=3.0, window_duration=3.0, max_speedup=1.5)
    at_13, ov_13 = fit_to_window(samples, slot_duration=3.0, window_duration=3.0, max_speedup=1.3)

    assert abs(duration_of(at_15) - 6.0 / 1.5) < 0.2      # ~4,0s
    assert abs(duration_of(at_13) - 6.0 / 1.3) < 0.2      # ~4,6s — chậm hơn, ít "nhanh" hơn
    assert duration_of(at_13) > duration_of(at_15)        # cap thấp => đọc êm hơn
    assert ov_13 > ov_15                                  # đánh đổi: tràn nhiều hơn (drift)


# ─── lấp nhẹ để bám hình (chống "nói nhanh hơn hình, xong sớm") ───

def test_short_audio_is_gently_slowed_to_track_the_picture_when_fill_enabled():
    """Tiếng Việt đọc nhanh hơn tiếng Anh nên hay xong sớm hơn khung. Kéo giãn NHẸ
    (tối đa tới sàn 0,9×, tai không nhận ra) để bám hình thay vì để im lặng cụt lủn."""
    samples = pcm_to_array(sine_pcm(4.0))
    filled, overflow = fit_to_window(samples, slot_duration=6.0, window_duration=6.0,
                                     max_speedup=1.3, fill_slowdown=0.9)
    assert abs(duration_of(filled) - 4.0 / 0.9) < 0.15   # 4s ở 0,9× ≈ 4,44s
    assert overflow == 0.0


def test_fill_is_disabled_by_default_so_short_audio_is_left_untouched():
    """Mặc định giữ hành vi cũ: không kéo chậm, để im lặng — chỉ bật khi runner truyền sàn."""
    samples = pcm_to_array(sine_pcm(4.0))
    filled, _ = fit_to_window(samples, slot_duration=6.0, window_duration=6.0, max_speedup=1.3)
    assert len(filled) == len(samples)


def test_fill_never_slows_below_the_floor_so_the_voice_stays_natural():
    """Câu rất ngắn so với khung: chỉ kéo chậm tối đa tới sàn, không kéo tới lấp kín khung."""
    samples = pcm_to_array(sine_pcm(2.0))
    filled, _ = fit_to_window(samples, slot_duration=10.0, window_duration=10.0,
                              max_speedup=1.3, fill_slowdown=0.9)
    assert abs(duration_of(filled) - 2.0 / 0.9) < 0.1    # đúng sàn 0,9×, không hơn


def test_audio_that_already_fills_its_slot_is_not_slowed():
    """Chỉ lấp khi audio NGẮN hơn khung; audio vừa/đủ khung không bị đụng."""
    samples = pcm_to_array(sine_pcm(6.0))
    filled, _ = fit_to_window(samples, slot_duration=6.0, window_duration=6.0,
                              max_speedup=1.3, fill_slowdown=0.9)
    assert len(filled) == len(samples)


def test_the_last_segment_may_run_to_the_end_of_the_video():
    backend = FakeGemini(tts_duration=3.0)
    fitted, warnings = synthesize_segments(
        backend, segments((0.0, 2.0, "Một câu đủ dài để guard chống chạy hoang không đụng vào nó chút nào.")), "Kore", workers=1, total_duration=30.0
    )
    assert duration_of(fitted[0][1]) > 2.0   # không bị ép cứng vào khung 2 giây
    assert warnings == []


def test_no_borrowing_when_the_gap_is_too_small_to_leave_recovery_room():
    """Khoảng lặng dưới mức chừa (0,4s) thì không mượn — đó là chỗ hồi của cả chuỗi.

    Đo thật: cho mượn cạn kiệt khoảng lặng → độ lệch dồn toa chỉ tan 0,03s mỗi bước,
    5 câu tràn cứng kéo 133 câu lùi theo.
    """
    backend = FakeGemini(tts_duration=3.0)
    segs = segments((0.0, 2.0, "Một câu đủ dài để guard chống chạy hoang không đụng vào nó chút nào."), (2.2, 4.0, "b"))   # khoảng lặng 0,2s < 0,4s
    fitted, _ = synthesize_segments(backend, segs, "Kore", workers=1, total_duration=10.0)

    assert duration_of(fitted[0][1]) <= 2.05   # ép hẳn về khung, không lấn sang câu sau


def test_without_total_duration_the_old_slot_behaviour_remains():
    """CLI/test không truyền tổng thời lượng → khung gốc như trước, không đoán mò."""
    backend = FakeGemini(tts_duration=3.0)
    fitted, _ = synthesize_segments(backend, segments((0.0, 2.0, "Một câu đủ dài để guard chống chạy hoang không đụng vào nó chút nào.")), "Kore", workers=1)
    assert duration_of(fitted[0][1]) <= 2.1   # bị ép về khung 2 giây


def test_the_chunk_boundary_caps_the_last_segments_window():
    """Xử lý theo cụm: lượt cuối cụm không được mượn lố sang mốc đầu cụm kế tiếp."""
    text = "Một câu đủ dài để guard chống chạy hoang không đụng vào nó chút nào."
    backend = FakeGemini(tts_duration=3.0)
    fitted, _ = synthesize_segments(
        backend, segments((0.0, 2.0, text)), "Kore", workers=1,
        total_duration=60.0, next_utterance_start=2.4,   # cụm sau bắt đầu ngay sát
    )
    # khoảng lặng 0,4s = đúng mức chừa → không mượn được gì, ép về khung 2s
    assert duration_of(fitted[0][1]) <= 2.05


# ─── chống TTS chạy hoang ───

def test_runaway_tts_audio_is_capped_by_text_length():
    """Đo thật: chữ "Và" (2 ký tự) sinh 7,1 giây giọng bịa, kéo 15 câu sau lệch 3–6s.

    Chạy hoang vẫn phải bị chặn: 7s → cắt về sàn 4s rồi ép khung còn ngắn hơn nữa.
    Sàn 4s tha cho câu ngắn đọc bình thường (≤3,5s) nhưng vẫn cắt audio bịa 6–8s.
    """
    backend = FakeGemini(tts_duration=7.0)
    segs = segments((0.0, 0.5, "Và"), (1.5, 3.0, "câu bình thường phía sau"))
    fitted, _ = synthesize_segments(backend, segs, "Kore", workers=1, total_duration=10.0)

    by_start = {seg.start: samples for seg, samples in fitted}
    assert duration_of(by_start[0.0]) < 3.0   # 7s bịa → cắt về sàn 4s rồi ép khung


def test_normal_speech_is_never_touched_by_the_runaway_guard():
    text = "Một câu dài bình thường, khoảng sáu mươi ký tự gì đó là vừa."
    backend = FakeGemini(tts_duration=3.5)    # ~17 ký tự/giây — tốc độ đọc thật
    fitted, warnings = synthesize_segments(
        backend, segments((0.0, 4.0, text)), "Kore", workers=1, total_duration=30.0
    )
    assert abs(duration_of(fitted[0][1]) - 3.5) < 0.05   # nguyên vẹn
    assert warnings == []


def test_short_normal_speech_survives_the_runaway_guard():
    """Câu NGẮN đọc bình thường (2,5–3,5s) KHÔNG được bị cắt.

    VieNeu có chi phí cố định ~2–3,5s mỗi lần đọc, gần như không theo độ dài text.
    Ngưỡng len/8+1 tuyến tính quá chặt với câu ngắn: "Thứ hai," (8 ký tự) chỉ cho
    2,0s trong khi giọng thật ~3s → cắt cụt tiếng thật (nghe 'ngắt đột ngột').
    """
    backend = FakeGemini(tts_duration=3.0)   # đọc ngắn bình thường
    # Khung rộng để không có tăng tốc xen vào — cô lập đúng hành vi bộ chống chạy hoang.
    fitted, warnings = synthesize_segments(
        backend, segments((0.0, 5.0, "Thứ hai,")), "Kore", workers=1, total_duration=30.0
    )
    assert abs(duration_of(fitted[0][1]) - 3.0) < 0.1   # nguyên vẹn, KHÔNG bị cắt về 2,0s
    assert warnings == []


def test_audio_shorter_than_its_slot_is_left_untouched():
    """Không kéo chậm giọng để lấp khoảng trống — im lặng nghe tự nhiên hơn."""
    samples = pcm_to_array(sine_pcm(1.0))
    fitted, overflow = fit_to_slot(samples, target_duration=3.0, max_speedup=1.5)
    assert len(fitted) == len(samples)
    assert overflow == 0.0


def test_zero_length_slot_is_left_untouched():
    samples = pcm_to_array(sine_pcm(1.0))
    fitted, overflow = fit_to_slot(samples, target_duration=0.0, max_speedup=1.5)
    assert len(fitted) == len(samples)
    assert overflow == 0.0


def test_empty_audio_is_handled():
    fitted, overflow = fit_to_slot(pcm_to_array(b""), target_duration=2.0, max_speedup=1.5)
    assert len(fitted) == 0
    assert overflow == 0.0


def test_sample_rate_is_read_from_the_mime_type_not_assumed():
    """Đoán sai tần số → toàn bộ giọng đọc sai cao độ và lệch thời gian."""
    assert parse_pcm_rate("audio/L16;codec=pcm;rate=24000") == 24000
    assert parse_pcm_rate("audio/L16;codec=pcm;rate=16000") == 16000


def test_missing_or_rateless_mime_type_returns_none():
    assert parse_pcm_rate(None) is None
    assert parse_pcm_rate("audio/L16;codec=pcm") is None
