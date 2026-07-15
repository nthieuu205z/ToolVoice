"""Cùng một khóa API, hai endpoint khác nhau — chọn nhầm là 403 hoặc 401.

Đo thực tế với khóa nằm trong project Google Cloud:
- Developer API (generativelanguage.googleapis.com) → 403 API_KEY_SERVICE_BLOCKED
- Vertex AI Express (aiplatform.googleapis.com)     → chạy tốt
Và ngược lại với khóa từ aistudio.google.com.
"""

from __future__ import annotations

import pytest

from pipeline.backends import LazyGemini, ProviderConfig, build_backend
from pipeline.gemini import GeminiRunner


def _config(backend: str) -> ProviderConfig:
    return ProviderConfig(
        stt_provider="whisper", tts_provider="edge",
        gemini_api_key="k", gemini_backend=backend,
        gemini_stt_model="m", gemini_translate_model="m", gemini_tts_model="m",
    )


@pytest.fixture
def spy_client(monkeypatch):
    calls = []

    class _Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("pipeline.gemini.genai.Client", _Client)
    return calls


def test_developer_api_is_the_default(spy_client):
    GeminiRunner(api_key="k", stt_model="m", translate_model="m", tts_model="m")
    assert spy_client == [{"api_key": "k"}]  # không có vertexai


def test_vertex_sends_the_key_to_the_aiplatform_endpoint(spy_client):
    """Khóa từ Google Cloud bị Developer API trả 403; phải đi qua Vertex Express."""
    GeminiRunner(api_key="k", stt_model="m", translate_model="m", tts_model="m", use_vertex=True)
    assert spy_client == [{"vertexai": True, "api_key": "k"}]


def test_the_backend_setting_reaches_the_client(spy_client):
    build_backend(_config("vertex"))._translator._get()
    assert spy_client[0]["vertexai"] is True


def test_developer_setting_never_turns_on_vertex(spy_client):
    build_backend(_config("developer"))._translator._get()
    assert "vertexai" not in spy_client[0]


def test_an_unknown_backend_falls_back_to_developer(spy_client):
    """Gõ sai `GEMINI_BACKEND` thì đi đường mặc định, không nổ."""
    build_backend(_config("gõ-sai"))._translator._get()
    assert "vertexai" not in spy_client[0]


def test_the_client_is_built_only_once(spy_client):
    lazy = LazyGemini(_config("vertex"))
    lazy._get()
    lazy._get()
    assert len(spy_client) == 1
