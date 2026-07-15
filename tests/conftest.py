"""Bản Gemini giả — test không bao giờ gọi API thật, không tốn token."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from pipeline.models import TTS_SAMPLE_RATE


def sine_pcm(duration: float, freq: float = 220.0, rate: int = TTS_SAMPLE_RATE) -> bytes:
    """PCM 16-bit mono, dùng làm âm thanh giả cho TTS."""
    count = int(duration * rate)
    samples = (int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(count))
    return struct.pack(f"<{count}h", *samples)


class FakeGemini:
    """Bám đúng giao thức GeminiBackend, có thể lập trình để mô phỏng lỗi."""

    def __init__(
        self,
        clips: list[str] | None = None,
        language: str = "en",
        *,
        translations: list[list[str]] | None = None,
        tts_duration: float | None = None,
        fail_tts_for: set[str] | None = None,
        tts_error: Exception | None = None,
    ):
        # Văn bản trả về cho từng vùng, theo thứ tự được gọi. Thiếu thì lặp phần tử cuối.
        self.clips = clips
        self.language = language
        # Mỗi phần tử là kết quả cho một lần gọi translate — dùng để test lệch số dòng.
        self.translations = translations
        self.tts_duration = tts_duration
        self.fail_tts_for = fail_tts_for or set()
        self.tts_error = tts_error

        self.transcribe_calls: list[Path] = []
        self.translate_calls: list[tuple[list[str], list[float], str]] = []
        self.synthesize_calls: list[tuple[str, str]] = []

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        """Văn bản gán theo CHỈ SỐ VÙNG lấy từ tên file, không theo thứ tự gọi.

        Các vùng chạy song song nên thứ tự gọi là ngẫu nhiên; gán theo thứ tự gọi sẽ
        khiến test đổi kết quả mỗi lần chạy và che mất lỗi ghép sai vùng.
        """
        path = Path(wav_path)
        self.transcribe_calls.append(path)
        index = int(path.stem.rsplit("_", 1)[-1])

        if self.clips is None:
            return self.language, f"clip {index}"
        if not self.clips:
            return self.language, ""
        return self.language, self.clips[min(index, len(self.clips) - 1)]

    def translate(self, texts: list[str], durations: list[float], context: str = "") -> list[str]:
        self.translate_calls.append((texts, durations, context))
        if self.translations:
            index = min(len(self.translate_calls) - 1, len(self.translations) - 1)
            return self.translations[index]
        return [f"[vi] {t}" for t in texts]

    def synthesize(self, text: str, voice_id: str) -> bytes:
        self.synthesize_calls.append((text, voice_id))
        if self.tts_error is not None:
            raise self.tts_error
        if text in self.fail_tts_for:
            raise RuntimeError(f"TTS hỏng với: {text}")
        return sine_pcm(self.tts_duration if self.tts_duration is not None else 1.0)


@pytest.fixture
def fake_gemini() -> FakeGemini:
    return FakeGemini()


@pytest.fixture(autouse=True)
def _isolated_custom_voices(tmp_path_factory, monkeypatch):
    """Mọi test dùng kho giọng nhân bản RIÊNG — không thấy và không đụng giọng thật."""
    from pipeline import custom_voices

    monkeypatch.setattr(custom_voices, "_dir", tmp_path_factory.mktemp("voices"))
