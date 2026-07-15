"""Mỗi bước đi tới đúng nhà cung cấp, và Gemini chỉ được dựng khi thật sự cần."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.backends import CompositeBackend, LazyGemini, ProviderConfig, build_backend

FREE = ProviderConfig(
    stt_provider="whisper", tts_provider="edge",
    gemini_api_key="", gemini_stt_model="m", gemini_translate_model="m", gemini_tts_model="m",
)
ALL_GEMINI = ProviderConfig(
    stt_provider="gemini", tts_provider="gemini",
    gemini_api_key="k", gemini_stt_model="m", gemini_translate_model="m", gemini_tts_model="m",
)


class _Spy:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def transcribe_clip(self, wav_path):
        self.calls.append("stt")
        return "en", f"{self.name}-text"

    def translate(self, texts, durations, context=""):
        self.calls.append("translate")
        return [f"{self.name}-vi"] * len(texts)

    def synthesize(self, text, voice_id):
        self.calls.append("tts")
        return b"\x01"


def test_each_step_reaches_its_own_provider():
    stt, tr, tts = _Spy("whisper"), _Spy("gemini"), _Spy("edge")
    backend = CompositeBackend(stt, tr, tts)

    backend.transcribe_clip(Path("a.wav"))
    backend.translate(["hi"], [1.0])
    backend.synthesize("chào", "v")

    assert stt.calls == ["stt"]
    assert tr.calls == ["translate"]
    assert tts.calls == ["tts"]


def test_gemini_is_not_constructed_until_a_gemini_call_happens():
    """Không có khóa API vẫn phải tạo được file nghe thử bằng edge-tts."""
    lazy = LazyGemini(FREE)
    assert lazy._runner is None  # dựng backend không đụng tới Gemini


def test_free_stack_uses_whisper_and_edge(monkeypatch):
    built = {}

    class _FakeWhisper:
        def __init__(self, model, compute):
            built["whisper"] = model

    class _FakeEdge:
        def __init__(self, attempts):
            built["edge"] = attempts

    monkeypatch.setattr("pipeline.whisper_stt.WhisperTranscriber", _FakeWhisper)
    monkeypatch.setattr("pipeline.edge_speech.EdgeSynthesizer", _FakeEdge)

    backend = build_backend(FREE)

    assert built == {"whisper": "small", "edge": 5}
    assert isinstance(backend._translator, LazyGemini)


def test_gemini_stack_shares_one_runner_across_all_three_steps():
    backend = build_backend(ALL_GEMINI)
    assert backend._recognizer is backend._translator is backend._synthesizer


def test_translation_always_goes_to_gemini_even_on_the_free_stack(monkeypatch):
    monkeypatch.setattr("pipeline.whisper_stt.WhisperTranscriber", lambda *a: object())
    monkeypatch.setattr("pipeline.edge_speech.EdgeSynthesizer", lambda *a: object())
    backend = build_backend(FREE)
    assert isinstance(backend._translator, LazyGemini)
