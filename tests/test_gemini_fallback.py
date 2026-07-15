"""Khóa lại các cạm bẫy của Developer API (khóa AI Studio) đã đo được ngoài đời thật.

Không cạm bẫy nào ném exception theo cách thông thường, nên nếu không có test này thì
mọi hồi quy đều biểu hiện thành "video im lặng" chứ không phải lỗi đỏ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from pipeline.errors import GeminiAPIError, QuotaExhaustedError
from pipeline.gemini import (
    _MAX_HONOURED_DELAY,
    _TTS_INSTRUCTION,
    GeminiRunner,
    _is_retryable,
    _quota_scope,
    _retry_delay_seconds,
    _wait_strategy,
)

PER_DAY_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel", "quotaValue": "100"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "19113s"},
]}}

PER_MINUTE_429 = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
     "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel", "quotaValue": "10"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "31s"},
]}}


class _Inline:
    def __init__(self, data, mime_type="audio/L16;codec=pcm;rate=24000"):
        self.data = data
        self.mime_type = mime_type


class _Part:
    def __init__(self, data=None):
        self.inline_data = _Inline(data) if data else None
        self.text = None


class _Candidate:
    def __init__(self, parts, finish_reason="STOP"):
        self.content = type("C", (), {"parts": parts})() if parts is not None else None
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, parts, finish_reason="STOP"):
        self.candidates = [_Candidate(parts, finish_reason)]
        self.text = ""
        self.parsed = None


class _Client:
    def __init__(self, models):
        self.models = models


@pytest.fixture
def runner() -> GeminiRunner:
    return GeminiRunner(api_key="dummy", stt_model="m", translate_model="m", tts_model="m")


# ─── phân loại 429: theo ngày (vô vọng) vs theo phút (đợi là qua) ───

def test_per_day_quota_is_recognised_as_hopeless():
    exc = genai_errors.ClientError(429, PER_DAY_429)
    assert _quota_scope(exc) == "day"


def test_per_minute_quota_is_recognised_as_worth_retrying():
    exc = genai_errors.ClientError(429, PER_MINUTE_429)
    assert _quota_scope(exc) == "minute"


def test_per_day_quota_is_not_retried():
    """Retry vô nghĩa: server bảo đợi 19113 giây."""
    assert _is_retryable(genai_errors.ClientError(429, PER_DAY_429)) is False


def test_per_minute_quota_is_still_retried():
    assert _is_retryable(genai_errors.ClientError(429, PER_MINUTE_429)) is True


def test_server_errors_are_still_retried():
    assert _is_retryable(genai_errors.ServerError(503, {"error": {"code": 503}})) is True


def test_google_supplied_retry_delay_is_honoured_over_fixed_backoff():
    """Backoff cứng bỏ cuộc sau ~14 giây; 429-theo-phút bảo đợi lâu hơn thế."""
    exc = genai_errors.ClientError(429, PER_MINUTE_429)
    assert _retry_delay_seconds(exc) == 31.0

    state = SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc), attempt_number=1)
    assert 31.0 <= _wait_strategy(state) <= _MAX_HONOURED_DELAY


def test_an_absurd_retry_delay_is_capped_not_obeyed():
    exc = genai_errors.ClientError(429, PER_DAY_429)  # bảo đợi 19113 giây
    state = SimpleNamespace(outcome=SimpleNamespace(exception=lambda: exc), attempt_number=1)
    assert _wait_strategy(state) == _MAX_HONOURED_DELAY


def test_tts_text_confusion_is_retried():
    """"Model tried to generate text" xảy ra ngẫu nhiên với câu hỏi — lần sau thường đọc được."""
    body = {"error": {"code": 400, "message":
            "Model tried to generate text, but it should only be used for TTS."}}
    assert _is_retryable(genai_errors.ClientError(400, body)) is True


def test_other_400_errors_are_not_retried():
    body = {"error": {"code": 400, "message": "Invalid voice name"}}
    assert _is_retryable(genai_errors.ClientError(400, body)) is False


def test_per_day_quota_is_reported_but_never_retried(runner):
    """Google bảo đợi 4 tiếng — thử lại là đốt 27 phút vô ích như đã xảy ra thật."""
    calls = []

    class _Models:
        def generate_content(self, **kwargs):
            calls.append(1)
            raise genai_errors.ClientError(429, PER_DAY_429)

    runner._client = _Client(_Models())

    with pytest.raises(QuotaExhaustedError):
        runner.synthesize("chào", "Kore")

    assert len(calls) == 1  # đúng một lần thử, không retry
    assert runner._quota_exhausted is True


def test_later_segments_still_get_their_one_attempt_after_a_day_429(runner):
    """Hạn mức của Google rò rỉ: đo thật thấy 3/8 request vẫn qua sau khi đã nhận 429/ngày."""
    calls = []

    class _LeakyQuota:
        def generate_content(self, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise genai_errors.ClientError(429, PER_DAY_429)
            return _Response(parts=[_Part(b"\x07")])

    runner._client = _Client(_LeakyQuota())

    with pytest.raises(QuotaExhaustedError):
        runner.synthesize("lượt một", "Kore")

    # Lượt hai vẫn được thử — và lọt qua.
    assert runner.synthesize("lượt hai", "Kore") == b"\x07"
    assert len(calls) == 2


# ─── cạm bẫy: HTTP 200, finish_reason=OTHER, không có audio ───

class _LanguageCodeTrap:
    """gemini-2.5-flash-preview-tts: có language_code → 200 kèm 0 part, KHÔNG ném lỗi."""

    def __init__(self):
        self.sent = []

    def generate_content(self, *, model, contents, config):
        lang = config.speech_config.language_code
        self.sent.append(lang)
        if lang is not None:
            return _Response(parts=[], finish_reason="OTHER")
        return _Response(parts=[_Part(b"\x01\x02")])


def test_silent_rejection_of_language_code_triggers_a_retry_without_it(runner):
    models = _LanguageCodeTrap()
    runner._client = _Client(models)

    audio = runner.synthesize("chào", "Kore")

    assert audio == b"\x01\x02"
    assert models.sent == ["vi-VN", None]
    assert runner._tts_language_code_rejected is True


def test_the_rejection_is_remembered_for_later_segments(runner):
    models = _LanguageCodeTrap()
    runner._client = _Client(models)

    runner.synthesize("câu một", "Kore")
    runner.synthesize("câu hai", "Kore")

    assert models.sent == ["vi-VN", None, None]  # câu hai không thử lại language_code


def test_explicit_400_rejection_still_falls_back(runner):
    """Model khác từ chối bằng lỗi 400 thay vì im lặng — nhánh cũ phải còn sống."""
    sent = []

    class _Models:
        def generate_content(self, *, model, contents, config):
            sent.append(config.speech_config.language_code)
            if config.speech_config.language_code is not None:
                raise genai_errors.ClientError(400, {"error": {"message": "bad language_code"}})
            return _Response(parts=[_Part(b"\x03")])

    runner._client = _Client(_Models())
    assert runner.synthesize("chào", "Kore") == b"\x03"
    assert sent == ["vi-VN", None]


def test_no_audio_even_without_language_code_raises_rather_than_going_silent(runner):
    class _Models:
        def generate_content(self, **kwargs):
            return _Response(parts=[], finish_reason="OTHER")

    runner._client = _Client(_Models())
    with pytest.raises(GeminiAPIError, match="không trả về âm thanh"):
        runner.synthesize("chào", "Kore")


def test_error_message_names_the_model_and_finish_reason(runner):
    class _Models:
        def generate_content(self, **kwargs):
            return _Response(parts=[], finish_reason="SAFETY")

    runner._client = _Client(_Models())
    with pytest.raises(GeminiAPIError) as excinfo:
        runner.synthesize("chào", "Kore")
    assert "SAFETY" in str(excinfo.value)


# ─── STT: tắt thinking (chép lời không cần suy luận) ───

class _TranscriptResponse:
    def __init__(self):
        from pipeline.gemini import _ClipTranscript

        self.parsed = _ClipTranscript(language="en", text="hello")
        self.candidates = []
        self.text = ""


def test_transcription_disables_thinking_by_default(runner, tmp_path):
    """Đo trên gemini-3.5-flash qua Vertex: 4,2–18,9s/request mặc định → 2,3s khi tắt."""
    seen = []

    class _Models:
        def generate_content(self, *, model, contents, config):
            seen.append(config.thinking_config)
            return _TranscriptResponse()

    runner._client = _Client(_Models())
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF fake wav")

    assert runner.transcribe_clip(clip) == ("en", "hello")
    assert seen[0] is not None and seen[0].thinking_budget == 0


def test_a_model_that_rejects_thinking_config_falls_back_and_remembers(runner, tmp_path):
    """Đổi GEMINI_STT_MODEL sang model không có thinking thì phải tự lùi, không chết."""
    seen = []

    class _Models:
        def generate_content(self, *, model, contents, config):
            seen.append(config.thinking_config)
            if config.thinking_config is not None:
                raise genai_errors.ClientError(
                    400, {"error": {"message": "thinking is not supported"}})
            return _TranscriptResponse()

    runner._client = _Client(_Models())
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF fake wav")

    assert runner.transcribe_clip(clip) == ("en", "hello")
    assert runner.transcribe_clip(clip) == ("en", "hello")

    # Lần đầu: thử có thinking rồi lùi; lần hai: nhớ luôn, không thử lại nữa.
    assert seen[0] is not None
    assert seen[1] is None and seen[2] is None
    assert runner._stt_thinking_rejected is True


# ─── _parts phòng thủ ───

def test_parts_of_a_response_without_candidates_is_empty(runner):
    assert GeminiRunner._parts(type("R", (), {})()) == []


def test_parts_of_a_response_with_empty_candidates_is_empty(runner):
    assert GeminiRunner._parts(type("R", (), {"candidates": []})()) == []


def test_parts_of_a_candidate_without_content_is_empty(runner):
    assert GeminiRunner._parts(_Response(parts=None)) == []


def test_has_audio_distinguishes_real_audio_from_empty_parts():
    assert GeminiRunner._has_audio(_Response(parts=[_Part(b"\x01")])) is True
    assert GeminiRunner._has_audio(_Response(parts=[_Part(None)])) is False
    assert GeminiRunner._has_audio(_Response(parts=[])) is False


# ─── lời thoại phải được ra lệnh đọc, không đưa trần ───

def test_speech_request_wraps_the_line_in_a_read_aloud_instruction(runner):
    """Đưa câu hỏi trần vào TTS thì model tưởng là câu lệnh và định trả lời (lỗi 400 thật)."""
    sent = []

    class _Models:
        def generate_content(self, *, model, contents, config):
            sent.append(contents)
            return _Response(parts=[_Part(b"\x01")])

    runner._client = _Client(_Models())
    runner.synthesize("Bạn viết tên lớp như thế nào?", "Kore")

    assert sent[0].startswith(_TTS_INSTRUCTION)
    assert sent[0].endswith("Bạn viết tên lớp như thế nào?")


def test_only_audio_modality_is_requested(runner):
    class _Models:
        def generate_content(self, *, model, contents, config):
            assert config.response_modalities == ["AUDIO"]
            return _Response(parts=[_Part(b"\x01")])

    runner._client = _Client(_Models())
    runner.synthesize("chào", "Kore")
