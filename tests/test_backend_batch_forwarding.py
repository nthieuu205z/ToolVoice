"""CompositeBackend phải chuyển tiếp khả năng gộp lô của nhà cung cấp giọng đọc.

Bài học đắt: gộp lô của VieNeu đã chạy đúng, test đơn vị xanh, benchmark nhanh gấp 20 lần
— nhưng trong app THẬT nó không hề chạy. Vì pipeline nói chuyện với CompositeBackend, mà
vỏ bọc đó chỉ phơi ra synthesize(). tts.py hỏi `backend.batch_size`, không thấy, và lặng lẽ
lùi về đọc từng câu một. Không lỗi, không cảnh báo, chỉ là chậm y như cũ.

Benchmark hồi đó truyền thẳng VieNeuSynthesizer vào synthesize_segments nên đi vòng qua
đúng cái vỏ này — nó đo một đường code mà người dùng không bao giờ chạy tới.
"""

from __future__ import annotations

import numpy as np

from pipeline.backends import CompositeBackend
from pipeline.models import TTS_SAMPLE_RATE, Segment
from pipeline.tts import synthesize_segments


class SynthCoLo:
    """Nhà cung cấp giọng đọc có gộp lô (như VieNeu trên GPU)."""

    batch_size = 32

    def __init__(self):
        self.lo_da_goi: list[list[str]] = []
        self.le_da_goi: list[str] = []

    def synthesize_batch(self, texts, voice_id):
        self.lo_da_goi.append(list(texts))
        return [np.zeros(TTS_SAMPLE_RATE, dtype="<i2").tobytes() for _ in texts]

    def synthesize(self, text, voice_id):
        self.le_da_goi.append(text)
        return np.zeros(TTS_SAMPLE_RATE, dtype="<i2").tobytes()


class SynthKhongLo:
    """Nhà cung cấp không gộp lô được (edge-tts, Gemini, VieNeu trên CPU)."""

    def __init__(self):
        self.le_da_goi: list[str] = []

    def synthesize(self, text, voice_id):
        self.le_da_goi.append(text)
        return np.zeros(TTS_SAMPLE_RATE, dtype="<i2").tobytes()


def _backend(synth):
    return CompositeBackend(recognizer=None, translator=None, synthesizer=synth)


def _segs():
    return [Segment(start=i * 3.0, end=i * 3.0 + 2.0, text="x", text_vi=c)
            for i, c in enumerate(["a", "b", "c"])]


def test_vo_boc_phoi_ra_batch_size_cua_nha_cung_cap():
    assert _backend(SynthCoLo()).batch_size == 32


def test_vo_boc_bao_0_khi_nha_cung_cap_khong_gop_lo_duoc():
    """edge-tts/Gemini không có batch_size — vỏ bọc phải trả 0, không được nổ AttributeError."""
    assert _backend(SynthKhongLo()).batch_size == 0


def test_pipeline_that_su_di_duong_gop_lo_qua_vo_boc():
    """Đây là bài test đáng lẽ phải có từ đầu: đi qua ĐÚNG đối tượng mà app truyền vào."""
    synth = SynthCoLo()
    segs = _segs()

    synthesize_segments(_backend(synth), segs, "v", total_duration=30.0)

    assert synth.lo_da_goi == [["a", "b", "c"]]   # một lượt gọi cho cả ba
    assert synth.le_da_goi == []                  # KHÔNG được rơi về đọc từng câu


def test_nha_cung_cap_khong_gop_lo_van_chay_duong_cu():
    synth = SynthKhongLo()
    segs = _segs()

    fitted, _ = synthesize_segments(_backend(synth), segs, "v", workers=1, total_duration=30.0)

    assert sorted(synth.le_da_goi) == ["a", "b", "c"]
    assert len(fitted) == 3


# ─── nhận diện lời cũng gộp lô, và cũng phải được chuyển tiếp ────────────

class NhanDienCoLo:
    stt_batch_size = 32

    def __init__(self):
        self.lo_da_goi: list[int] = []

    def transcribe_batch(self, samples, rate, regions):
        self.lo_da_goi.append(len(regions))
        return [("en", f"loi thoai {i}") for i in range(len(regions))]

    def transcribe_clip(self, path):
        raise AssertionError("khong duoc goi khi da co gop lo")


class NhanDienKhongLo:
    def transcribe_clip(self, path):
        return "en", "loi thoai"


def test_vo_boc_phoi_ra_stt_batch_size():
    b = CompositeBackend(recognizer=NhanDienCoLo(), translator=None, synthesizer=None)
    assert b.stt_batch_size == 32


def test_vo_boc_bao_0_khi_nhan_dien_khong_gop_lo_duoc():
    """Gemini không có stt_batch_size — phải trả 0, không được nổ AttributeError."""
    b = CompositeBackend(recognizer=NhanDienKhongLo(), translator=None, synthesizer=None)
    assert b.stt_batch_size == 0


def test_pipeline_that_su_di_duong_gop_lo_khi_nhan_dien():
    import numpy as np

    from pipeline.segmentation import Region
    from pipeline.stt import transcribe_regions

    nd = NhanDienCoLo()
    b = CompositeBackend(recognizer=nd, translator=None, synthesizer=None)
    regions = [Region(i * 5.0, i * 5.0 + 4.0) for i in range(3)]
    samples = np.zeros(16000 * 20, dtype="<i2")

    lang, segs, _ = transcribe_regions(b, samples, 16000, regions, workdir=None)

    assert nd.lo_da_goi == [3]          # một lượt gọi cho cả ba vùng
    assert lang == "en"
    assert len(segs) == 3
