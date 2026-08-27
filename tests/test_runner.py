"""Cảnh báo hạn mức TTS chỉ dành cho nhà cung cấp có trần theo ngày (Gemini).

VieNeu/edge chạy không giới hạn — video dài trăm câu mà vẫn dọa "vượt hạn mức
miễn phí" là báo sai, người dùng sẽ tưởng kết quả bị bỏ trống.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import runner
from pipeline.models import MediaInfo, Segment
from pipeline.runner import PipelineOptions, run_pipeline
from pipeline.segmentation import Region

MEDIA = MediaInfo(duration=5.0, video_codec="h264", has_audio=True)


@pytest.fixture
def stub_stages(monkeypatch, tmp_path):
    """Cắt hết ffmpeg lẫn nhà cung cấp: chỉ còn logic điều phối của run_pipeline."""
    segments = [Segment(float(i), float(i) + 1.0, "src", f"câu {i}") for i in range(3)]

    monkeypatch.setattr(runner, "extract_audio", lambda video, dest: dest)
    monkeypatch.setattr(runner, "read_wav",
                        lambda path: (np.zeros(16000, dtype="<i2"), 16000))
    monkeypatch.setattr(runner, "plan_utterances", lambda *a, **k: [Region(0.0, 1.0)])
    monkeypatch.setattr(runner, "transcribe_regions", lambda *a, **k: ("en", segments, []))
    monkeypatch.setattr(runner, "translate_segments", lambda b, s, p, c, **k: segments)
    monkeypatch.setattr(runner, "synthesize_segments", lambda *a, **k: ([], []))
    monkeypatch.setattr(runner, "build_srt", lambda segs: "")
    monkeypatch.setattr(runner, "build_audio_track", lambda fitted, dur, dest: dest)
    monkeypatch.setattr(runner, "run_mux", lambda video, audio, dest: dest)
    return tmp_path


def _options(*, metered: bool) -> PipelineOptions:
    # budget=2 nhưng có 3 lượt thoại — vượt trần trong cả hai test.
    return PipelineOptions(voice_id="v", tts_daily_budget=2, tts_is_metered=metered)


def test_gemini_tts_over_budget_warns(stub_stages):
    result = run_pipeline(None, stub_stages / "in.mp4", stub_stages,
                          _options(metered=True), media=MEDIA)
    assert any("hạn mức" in w for w in result.warnings)


def test_local_tts_never_warns_about_gemini_quota(stub_stages):
    """VieNeu vượt 'budget' thoải mái — trần đó không phải của nó."""
    result = run_pipeline(None, stub_stages / "in.mp4", stub_stages,
                          _options(metered=False), media=MEDIA)
    assert result.warnings == []


def test_metering_is_off_by_default():
    """CLI/route nào quên truyền cờ thì thà im lặng còn hơn dọa nhầm."""
    assert PipelineOptions(voice_id="v").tts_is_metered is False


# ─── đọc lại lượt xấu: chỉ engine CÓ khuyết tật lỗ hổng im lặng mới cần ───

def _capture_resynthesize(monkeypatch) -> dict:
    """Bắt cờ resynthesize_bad mà runner thật sự truyền xuống bước đọc giọng."""
    seen: dict = {}

    def fake_synth(*a, **k):
        seen["flag"] = k.get("resynthesize_bad")
        return [], []

    monkeypatch.setattr(runner, "synthesize_segments", fake_synth)
    return seen


def _run(stub_stages, options) -> None:
    run_pipeline(None, stub_stages / "in.mp4", stub_stages, options, media=MEDIA)


def test_omnivoice_skips_the_costly_resynthesis(stub_stages, monkeypatch):
    """OmniVoice tự vá lỗ hổng bằng postprocess; đọc lại là một single-synth ~5,8s KHÔNG gộp
    lô — ở video dài đó là khoản phí thuần túy vô ích."""
    seen = _capture_resynthesize(monkeypatch)
    _run(stub_stages, PipelineOptions(voice_id="v", resynthesize_holes=False))
    assert seen["flag"] is False


def test_vieneu_and_edge_still_resynthesize(stub_stages, monkeypatch):
    """Khuyết tật lỗ hổng im lặng là có thật ở VieNeu — không được tắt nhầm của chúng."""
    seen = _capture_resynthesize(monkeypatch)
    _run(stub_stages, PipelineOptions(voice_id="v"))
    assert seen["flag"] is True


def test_metered_backend_never_resynthesizes(stub_stages, monkeypatch):
    """Gemini tính tiền theo lượt: đọc lại vì CHẤT LƯỢNG là tiêu tiền, dù engine có khuyết tật."""
    seen = _capture_resynthesize(monkeypatch)
    _run(stub_stages, PipelineOptions(voice_id="v", tts_is_metered=True))
    assert seen["flag"] is False


def test_resynthesis_is_on_by_default():
    """Route/CLI quên truyền cờ thì giữ hành vi cũ (đọc lại), không im lặng đổi chất lượng."""
    assert PipelineOptions(voice_id="v").resynthesize_holes is True
