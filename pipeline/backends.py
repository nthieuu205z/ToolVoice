"""Ghép ba nhà cung cấp rời thành một backend mà pipeline mong đợi.

Mỗi bước đổi độc lập qua .env, nên khi một dịch vụ hỏng vẫn còn đường lùi:

    STT_PROVIDER=whisper|gemini    nhận diện giọng nói
    TTS_PROVIDER=edge|gemini       tạo giọng đọc
    (dịch luôn dùng Gemini — chỉ tốn 1–4 lượt gọi cho cả video)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderConfig:
    stt_provider: str
    tts_provider: str
    gemini_api_key: str
    gemini_stt_model: str
    gemini_translate_model: str
    gemini_tts_model: str
    gemini_backend: str = "developer"   # developer (AI Studio) | vertex (Google Cloud)
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"
    edge_tts_attempts: int = 5
    vieneu_watermark: bool = False


class LazyGemini:
    """Chỉ dựng GeminiRunner khi thật sự có lời gọi tới Gemini.

    Nếu dựng sớm thì chạy `generate_voice_previews.py` với edge-tts cũng đòi khóa API.
    """

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._runner = None

    def _get(self):
        if self._runner is None:
            from .gemini import GeminiRunner

            self._runner = GeminiRunner(
                api_key=self._config.gemini_api_key,
                stt_model=self._config.gemini_stt_model,
                translate_model=self._config.gemini_translate_model,
                tts_model=self._config.gemini_tts_model,
                use_vertex=self._config.gemini_backend == "vertex",
            )
        return self._runner

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        return self._get().transcribe_clip(wav_path)

    def translate(self, texts: list[str], durations: list[float], context: str = "") -> list[str]:
        return self._get().translate(texts, durations, context)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        return self._get().synthesize(text, voice_id)


class CompositeBackend:
    """Bám giao thức GeminiBackend, nhưng mỗi phương thức đi tới một nhà cung cấp khác nhau."""

    def __init__(self, recognizer, translator, synthesizer):
        self._recognizer = recognizer
        self._translator = translator
        self._synthesizer = synthesizer

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        return self._recognizer.transcribe_clip(wav_path)

    def translate(self, texts: list[str], durations: list[float], context: str = "") -> list[str]:
        return self._translator.translate(texts, durations, context)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        return self._synthesizer.synthesize(text, voice_id)

    # ── gộp lô ──────────────────────────────────────────────────────────
    # Cả hai model chạy trên máy (Whisper và VieNeu) đều nhanh hơn cả chục lần khi gộp lô.
    # Vỏ bọc này PHẢI chuyển tiếp, nếu không stt.py/tts.py không thấy `*_batch_size` và
    # LẶNG LẼ lùi về xử lý từng câu một — mất sạch phần tăng tốc mà không báo lỗi gì.
    # Đã từng xảy ra đúng như vậy: xem tests/test_backend_batch_forwarding.py.

    @property
    def batch_size(self) -> int:
        """0 = nhà cung cấp giọng đọc này không gộp lô được (edge-tts, Gemini, VieNeu/CPU)."""
        return getattr(self._synthesizer, "batch_size", 0)

    def synthesize_batch(self, texts: list[str], voice_id: str) -> list[bytes]:
        return self._synthesizer.synthesize_batch(texts, voice_id)

    @property
    def stt_batch_size(self) -> int:
        """0 = nhà cung cấp nhận diện này không gộp lô được (Gemini, Whisper/CPU)."""
        return getattr(self._recognizer, "stt_batch_size", 0)

    def transcribe_batch(self, samples, rate: int, regions: list) -> list[tuple[str, str]]:
        return self._recognizer.transcribe_batch(samples, rate, regions)

    @property
    def stt_timed(self) -> bool:
        """True = nhận diện này trả về mốc thời gian cấp CÂU (Whisper/GPU), để đặt câu bám hình."""
        return getattr(self._recognizer, "stt_timed", False)

    def transcribe_batch_timed(self, samples, rate: int, regions: list):
        return self._recognizer.transcribe_batch_timed(samples, rate, regions)


def build_backend(config: ProviderConfig) -> CompositeBackend:
    """Import muộn từng nhà cung cấp để không kéo phụ thuộc nặng khi không dùng tới."""
    gemini = LazyGemini(config)

    if config.stt_provider == "whisper":
        from .whisper_stt import WhisperTranscriber

        recognizer = WhisperTranscriber(config.whisper_model, config.whisper_compute_type)
    else:
        recognizer = gemini

    if config.tts_provider == "edge":
        from .edge_speech import EdgeSynthesizer

        synthesizer = EdgeSynthesizer(config.edge_tts_attempts)
    elif config.tts_provider == "vieneu":
        from .vieneu_speech import VieNeuSynthesizer

        synthesizer = VieNeuSynthesizer(watermark=config.vieneu_watermark)
    else:
        synthesizer = gemini

    return CompositeBackend(recognizer, gemini, synthesizer)
