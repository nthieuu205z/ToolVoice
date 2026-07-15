"""VieNeu-TTS trả float32 48 kHz; pipeline cần PCM 16-bit 24 kHz. Test khoá chỗ nối đó."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from pipeline.errors import SpeechServiceError
from pipeline.models import TTS_SAMPLE_RATE
from pipeline.vieneu_speech import VieNeuSynthesizer, _to_pcm16
from pipeline.voices import VIENEU_VOICES, native_id


# ─── đổi định dạng âm thanh ───

def test_forty_eight_kilohertz_is_halved_to_the_pipeline_rate():
    audio = np.zeros(48_000, dtype=np.float32)   # 1 giây ở 48 kHz
    pcm = _to_pcm16(audio, 48_000)
    assert len(pcm) == TTS_SAMPLE_RATE * 2       # 1 giây ở 24 kHz, 2 byte mỗi mẫu


def test_a_matching_rate_is_passed_through_untouched():
    audio = np.zeros(TTS_SAMPLE_RATE, dtype=np.float32)
    assert len(_to_pcm16(audio, TTS_SAMPLE_RATE)) == TTS_SAMPLE_RATE * 2


def test_float_amplitude_maps_onto_the_int16_range():
    pcm = _to_pcm16(np.array([1.0, -1.0, 0.0], dtype=np.float32), TTS_SAMPLE_RATE)
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767, 0]


def test_values_beyond_one_are_clipped_not_wrapped():
    """Không kẹp thì 1.5 tràn số âm và tạo tiếng nổ."""
    pcm = _to_pcm16(np.array([1.5, -1.5], dtype=np.float32), TTS_SAMPLE_RATE)
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32767]


def test_stereo_is_folded_down_to_mono():
    audio = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert len(_to_pcm16(audio, TTS_SAMPLE_RATE)) == 4  # 2 mẫu × 2 byte


def test_empty_audio_yields_empty_pcm():
    assert _to_pcm16(np.array([], dtype=np.float32), 48_000) == b""


def test_resampling_preserves_the_waveform_shape():
    t = np.linspace(0, 1, 48_000, endpoint=False)
    audio = np.sin(2 * np.pi * 5 * t).astype(np.float32)   # sóng sin 5 Hz
    samples = np.frombuffer(_to_pcm16(audio, 48_000), dtype="<i2").astype(np.float64) / 32767

    expected = np.sin(2 * np.pi * 5 * np.linspace(0, 1, TTS_SAMPLE_RATE, endpoint=False))
    assert np.abs(samples - expected).max() < 0.01


# ─── ánh xạ giọng ───

def test_url_safe_slug_maps_to_the_vietnamese_name_vieneu_expects():
    assert native_id("truc-ly", "vieneu") == "Trúc Ly"
    assert native_id("pham-tuyen", "vieneu") == "Phạm Tuyên"


def test_every_vieneu_slug_is_url_safe():
    for voice in VIENEU_VOICES:
        assert voice.id.replace("-", "").isalnum()
        assert voice.id.islower()


def test_all_three_regions_are_offered():
    labels = " ".join(v.display_name for v in VIENEU_VOICES)
    assert "giọng Bắc" in labels and "giọng Trung" in labels and "giọng Nam" in labels


# ─── gọi engine ───

class _FakeEngine:
    sample_rate = 48_000

    def __init__(self, audio=None, styles=None):
        self.audio = audio if audio is not None else np.zeros(4800, dtype=np.float32)
        self.styles = styles or {}
        self.calls = []

    def infer(self, text, *, voice, style, apply_watermark):
        self.calls.append({"text": text, "voice": voice, "style": style, "watermark": apply_watermark})
        return self.audio

    def get_preset_voice(self, voice):
        return {"style": self.styles.get(voice, "tu_nhien")}


@pytest.fixture
def engine(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr("pipeline.vieneu_speech._get_engine", lambda: fake)
    return fake


def test_the_slug_is_translated_before_reaching_the_engine(engine):
    VieNeuSynthesizer().synthesize("chào", "truc-ly")
    assert engine.calls[0]["voice"] == "Trúc Ly"


def test_each_voice_keeps_its_own_designed_style(monkeypatch):
    """Ép một phong cách chung sẽ xóa mất chất "kể chuyện" của Ngọc Linh."""
    fake = _FakeEngine(styles={"Ngọc Linh": "ke_chuyen"})
    monkeypatch.setattr("pipeline.vieneu_speech._get_engine", lambda: fake)

    VieNeuSynthesizer().synthesize("chào", "ngoc-linh")
    assert fake.calls[0]["style"] == "ke_chuyen"


def test_the_watermark_is_off_unless_asked_for(engine):
    VieNeuSynthesizer().synthesize("chào", "truc-ly")
    assert engine.calls[0]["watermark"] is False

    VieNeuSynthesizer(watermark=True).synthesize("chào", "truc-ly")
    assert engine.calls[1]["watermark"] is True


def test_silent_output_raises_instead_of_going_unnoticed(monkeypatch):
    fake = _FakeEngine(audio=np.array([], dtype=np.float32))
    monkeypatch.setattr("pipeline.vieneu_speech._get_engine", lambda: fake)

    with pytest.raises(SpeechServiceError, match="0 byte"):
        VieNeuSynthesizer().synthesize("chào", "truc-ly")


def test_an_engine_crash_names_the_way_out(monkeypatch):
    class _Boom(_FakeEngine):
        def infer(self, *a, **k):
            raise RuntimeError("onnx nổ")

    monkeypatch.setattr("pipeline.vieneu_speech._get_engine", lambda: _Boom())
    with pytest.raises(SpeechServiceError) as excinfo:
        VieNeuSynthesizer().synthesize("chào", "truc-ly")
    assert "TTS_PROVIDER=edge" in excinfo.value.user_message


# ─── nạp sẵn lúc khởi động ───

def test_prewarm_loads_the_engine_in_the_background(monkeypatch):
    import threading

    from pipeline import model_store, vieneu_speech

    monkeypatch.setattr(model_store, "is_ready", lambda spec: True)
    loaded = threading.Event()
    monkeypatch.setattr(vieneu_speech, "_get_engine", lambda: loaded.set())

    vieneu_speech.prewarm()
    assert loaded.wait(2)


def test_prewarm_never_triggers_a_download(monkeypatch):
    """Model chưa tải thì thôi — việc tải 609 MB là do người dùng bấm, không phải lúc boot."""
    import threading

    from pipeline import model_store, vieneu_speech

    monkeypatch.setattr(model_store, "is_ready", lambda spec: False)
    loaded = threading.Event()
    monkeypatch.setattr(vieneu_speech, "_get_engine", lambda: loaded.set())

    vieneu_speech.prewarm()
    assert not loaded.wait(0.3)


def test_a_missing_package_tells_you_how_to_install_it(monkeypatch):
    monkeypatch.setattr("pipeline.vieneu_speech._engine", None)
    monkeypatch.setitem(sys.modules, "vieneu", None)  # import → ImportError

    from pipeline import vieneu_speech

    with pytest.raises(SpeechServiceError) as excinfo:
        vieneu_speech._get_engine()
    assert ".[vieneu]" in excinfo.value.user_message
