"""edge-tts bóp tần suất và thỉnh thoảng từ chối — thử lại là bắt buộc, không phải tùy chọn.

Đo thực tế: 6 luồng song song → 25/32 thành công; 2 luồng kèm thử lại → 10/10.
"""

from __future__ import annotations

import sys
import types

import pytest

from pipeline.edge_speech import EdgeSynthesizer
from pipeline.errors import SpeechServiceError


class _FakeCommunicate:
    """Giả lập edge_tts.Communicate: hỏng `fail_times` lần đầu rồi mới trả audio."""

    calls: list[tuple[str, str]] = []
    fail_times = 0
    empty_times = 0
    audio = b"mp3-bytes"

    def __init__(self, text, voice, **kwargs):
        type(self).calls.append((text, voice))
        self._text = text

    async def stream(self):
        cls = type(self)
        index = len(cls.calls)
        if index <= cls.fail_times:
            raise RuntimeError("NoAudioReceived")
        if index <= cls.fail_times + cls.empty_times:
            return
        yield {"type": "audio", "data": cls.audio}
        yield {"type": "WordBoundary", "data": b"ignored"}


@pytest.fixture(autouse=True)
def fake_edge_tts(monkeypatch):
    _FakeCommunicate.calls = []
    _FakeCommunicate.fail_times = 0
    _FakeCommunicate.empty_times = 0

    module = types.ModuleType("edge_tts")
    module.Communicate = _FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)

    # Không chờ backoff thật trong test.
    monkeypatch.setattr("pipeline.edge_speech._BACKOFF_SECONDS", 0.0)
    # decode_to_pcm gọi ffmpeg — thay bằng hàm nhận diện được.
    monkeypatch.setattr("pipeline.edge_speech.decode_to_pcm", lambda data, rate: b"pcm:" + data)
    return module


def test_a_clean_call_returns_decoded_pcm():
    assert EdgeSynthesizer().synthesize("chào", "vi-VN-HoaiMyNeural") == b"pcm:mp3-bytes"
    assert _FakeCommunicate.calls == [("chào", "vi-VN-HoaiMyNeural")]


def test_only_audio_chunks_are_kept():
    """Stream còn trả WordBoundary — gộp nhầm vào là hỏng file mp3."""
    assert b"ignored" not in EdgeSynthesizer().synthesize("chào", "v")


def test_a_throttled_call_is_retried_until_it_succeeds():
    _FakeCommunicate.fail_times = 2
    assert EdgeSynthesizer(attempts=5).synthesize("chào", "v") == b"pcm:mp3-bytes"
    assert len(_FakeCommunicate.calls) == 3


def test_an_empty_response_counts_as_failure_and_is_retried():
    """Dịch vụ có lúc trả stream rỗng mà không ném lỗi — im lặng là hỏng ngầm."""
    _FakeCommunicate.empty_times = 1
    assert EdgeSynthesizer(attempts=3).synthesize("chào", "v") == b"pcm:mp3-bytes"
    assert len(_FakeCommunicate.calls) == 2


def test_giving_up_raises_a_message_that_names_the_way_out():
    _FakeCommunicate.fail_times = 99
    with pytest.raises(SpeechServiceError) as excinfo:
        EdgeSynthesizer(attempts=3).synthesize("chào", "v")

    assert len(_FakeCommunicate.calls) == 3
    assert "TTS_PROVIDER=gemini" in excinfo.value.user_message


def test_attempts_is_never_below_one():
    _FakeCommunicate.fail_times = 99
    with pytest.raises(SpeechServiceError):
        EdgeSynthesizer(attempts=0).synthesize("chào", "v")
    assert len(_FakeCommunicate.calls) == 1
