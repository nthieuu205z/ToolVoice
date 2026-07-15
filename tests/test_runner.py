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
    monkeypatch.setattr(runner, "translate_segments", lambda b, s, p, c: segments)
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
