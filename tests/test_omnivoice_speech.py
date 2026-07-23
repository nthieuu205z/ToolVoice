"""OmniVoice làm backend TTS: nhân bản zero-shot từ giọng người dùng đã lưu.

OmniVoice KHÔNG có giọng dựng sẵn — mỗi lượt đọc cần ref_audio (clip mẫu) + ref_text
(lời của clip đó). Ta lấy clip từ custom_voices/ và để Whisper chép ref_text một lần.
Các test dùng model giả + transcriber giả nên không tải model 612 MB, không cần GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline import custom_voices
from pipeline.audio import pcm_to_array, write_wav
from pipeline.errors import SpeechServiceError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.omnivoice_speech import OmniVoiceSynthesizer


class FakeOmni:
    """Bám API OmniVoice.generate(text=[...], ref_audio=[...], ref_text=[...], **cfg) -> list @24kHz."""

    def __init__(self, audio: np.ndarray | None = None, error: Exception | None = None):
        self.audio = audio if audio is not None else np.full(2400, 0.3, dtype=np.float32)
        self.error = error
        self.calls: list[dict] = []

    def generate(self, text, ref_audio, ref_text, **kwargs):
        self.calls.append({"text": text, "ref_audio": ref_audio, "ref_text": ref_text, "kwargs": kwargs})
        if self.error is not None:
            raise self.error
        n = len(text) if isinstance(text, (list, tuple)) else 1
        return [self.audio for _ in range(n)]


class FakeTranscriber:
    """Callable (Path) -> str, đứng thay Whisper cho phần ref_text."""

    def __init__(self, text: str = "lời tham chiếu", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list[Path] = []

    def __call__(self, path: Path) -> str:
        self.calls.append(Path(path))
        if self.error is not None:
            raise self.error
        return self.text


def _make_clone(voice_id: str = "clone-test", display: str = "Giọng Test") -> str:
    """Tạo một giọng nhân bản có file mẫu trên đĩa trong kho tạm của test."""
    write_wav(custom_voices.sample_path(voice_id),
              np.zeros(2400, dtype="<i2"), TTS_SAMPLE_RATE)
    custom_voices.register(voice_id, display)
    return voice_id


def _sidecar(voice_id: str) -> Path:
    return custom_voices.sample_path(voice_id).with_name(f"{voice_id}.reftext.txt")


def test_synthesize_returns_pcm16_and_calls_generate():
    vid = _make_clone()
    audio = (0.5 * np.sin(
        2 * np.pi * 220 * np.arange(TTS_SAMPLE_RATE // 2) / TTS_SAMPLE_RATE
    )).astype(np.float32)  # 0,5 giây @24kHz
    model = FakeOmni(audio=audio)
    trans = FakeTranscriber(text="câu mẫu")
    synth = OmniVoiceSynthesizer(model=model, transcriber=trans)

    pcm = synth.synthesize("Xin chào", vid)

    arr = pcm_to_array(pcm)
    # 24kHz vào = 24kHz ra (TTS_SAMPLE_RATE), không nội suy → giữ nguyên số mẫu.
    assert len(arr) == len(audio)
    assert arr.dtype == np.dtype("<i2")
    assert int(np.abs(arr).max()) > 1000          # có tiếng thật, không phải im lặng

    assert len(model.calls) == 1
    call = model.calls[0]
    # API OmniVoice nhận list (đường gộp lô dùng chung) — một câu là list một phần tử.
    assert call["text"] == ["Xin chào"]
    assert call["ref_text"] == ["câu mẫu"]
    assert Path(call["ref_audio"][0]) == custom_voices.sample_path(vid)
    # Đặt sẵn tham số tốt: tiếng Việt + số bước sinh.
    assert call["kwargs"]["language"] == ["Vietnamese"]
    assert call["kwargs"]["num_step"] == 32


def test_ref_text_computed_once_and_persisted():
    vid = _make_clone()
    model = FakeOmni()
    trans = FakeTranscriber(text="mẫu")
    synth = OmniVoiceSynthesizer(model=model, transcriber=trans)

    synth.synthesize("a", vid)
    synth.synthesize("b", vid)

    assert len(trans.calls) == 1                   # chép lời đúng MỘT lần
    assert _sidecar(vid).read_text(encoding="utf-8").strip() == "mẫu"


def test_ref_text_reused_from_sidecar_without_transcribing():
    vid = _make_clone()
    _sidecar(vid).write_text("có sẵn", encoding="utf-8")
    model = FakeOmni()
    trans = FakeTranscriber(error=AssertionError("không được gọi transcriber khi đã có sidecar"))
    synth = OmniVoiceSynthesizer(model=model, transcriber=trans)

    synth.synthesize("x", vid)

    assert trans.calls == []
    assert model.calls[0]["ref_text"] == ["có sẵn"]


def test_rejects_non_custom_voice():
    synth = OmniVoiceSynthesizer(model=FakeOmni(), transcriber=FakeTranscriber())
    with pytest.raises(SpeechServiceError):
        synth.synthesize("x", "truc-ly")           # giọng dựng sẵn VieNeu, không phải clone-


def test_missing_sample_raises():
    synth = OmniVoiceSynthesizer(model=FakeOmni(), transcriber=FakeTranscriber())
    with pytest.raises(SpeechServiceError):
        synth.synthesize("x", "clone-khong-co-file")


def test_empty_audio_raises():
    vid = _make_clone()
    model = FakeOmni(audio=np.zeros(0, dtype=np.float32))
    synth = OmniVoiceSynthesizer(model=model, transcriber=FakeTranscriber(text="m"))
    with pytest.raises(SpeechServiceError):
        synth.synthesize("x", vid)


def test_generate_error_wrapped():
    vid = _make_clone()
    model = FakeOmni(error=RuntimeError("boom"))
    synth = OmniVoiceSynthesizer(model=model, transcriber=FakeTranscriber(text="m"))
    with pytest.raises(SpeechServiceError):
        synth.synthesize("x", vid)


def test_synthesize_batch_generates_all_in_one_call():
    vid = _make_clone()
    model = FakeOmni()
    synth = OmniVoiceSynthesizer(model=model, transcriber=FakeTranscriber(text="m"))
    texts = ["câu một", "câu hai", "câu ba"]

    pcms = synth.synthesize_batch(texts, vid)

    assert len(pcms) == 3
    assert all(isinstance(p, (bytes, bytearray)) and p for p in pcms)
    assert len(model.calls) == 1                     # MỘT lượt gọi generate cho cả lô
    assert model.calls[0]["text"] == texts
    assert model.calls[0]["ref_audio"] == [str(custom_voices.sample_path(vid))] * 3


def test_batch_size_uses_override_on_gpu(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_uses_gpu", lambda: True)
    synth = OmniVoiceSynthesizer(model=FakeOmni(), batch_size=5)
    assert synth.batch_size == 5


def test_batch_size_zero_on_cpu(monkeypatch):
    import pipeline.omnivoice_speech as ov

    monkeypatch.setattr(ov, "_uses_gpu", lambda: False)
    synth = OmniVoiceSynthesizer(model=FakeOmni(), batch_size=5)
    assert synth.batch_size == 0


def test_forget_clone_removes_reftext_sidecar():
    from pipeline.omnivoice_speech import forget_clone

    vid = _make_clone()
    sidecar = _sidecar(vid)
    sidecar.write_text("lời cũ", encoding="utf-8")
    assert sidecar.exists()

    forget_clone(vid)
    assert not sidecar.exists()


def test_voices_for_omnivoice_lists_only_clones():
    from pipeline.voices import voices_for

    _make_clone("clone-alpha", "Alpha")
    ids = [v.id for v in voices_for("omnivoice")]
    assert "clone-alpha" in ids
    assert "truc-ly" not in ids                    # không rò giọng dựng sẵn VieNeu


def test_build_backend_uses_omnivoice():
    from pipeline.backends import ProviderConfig, build_backend

    cfg = ProviderConfig(
        stt_provider="whisper", tts_provider="omnivoice",
        gemini_api_key="", gemini_stt_model="", gemini_translate_model="",
        gemini_tts_model="",
    )
    backend = build_backend(cfg)
    assert isinstance(backend._synthesizer, OmniVoiceSynthesizer)
