"""Đường gộp lô của tts.py phải giữ nguyên hai lời hứa của đường cũ.

Gộp lô làm tăng tốc ~20 lần, nhưng nó gom nhiều lượt thoại vào một lượt gọi — nên hai
thứ dễ mất nhất là: (1) hủy giữa chừng còn ăn không, (2) một câu hỏng có kéo cả lô chết
theo không. Hai cái đó chính là nội dung của file này.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import JobCancelledError
from pipeline.models import TTS_SAMPLE_RATE, Segment
from pipeline.tts import _chia_lo, synthesize_segments


def segments(*specs: tuple[float, float, str]) -> list[Segment]:
    return [Segment(start=s, end=e, text="x", text_vi=vi) for s, e, vi in specs]


def pcm(seconds: float = 1.0) -> bytes:
    return np.zeros(int(TTS_SAMPLE_RATE * seconds), dtype="<i2").tobytes()


class LoBackend:
    """Backend giả có gộp lô, đúng giao thức mà tts.py trông đợi."""

    def __init__(self, batch_size: int = 32, hong_lo: bool = False, hong_cau: str = ""):
        self.batch_size = batch_size
        self._hong_lo = hong_lo
        self._hong_cau = hong_cau
        self.lo_da_goi: list[list[str]] = []
        self.le_da_goi: list[str] = []

    def synthesize_batch(self, texts: list[str], voice_id: str) -> list[bytes]:
        self.lo_da_goi.append(list(texts))
        if self._hong_lo:
            raise RuntimeError("GPU hết bộ nhớ")
        return [pcm() for _ in texts]

    def synthesize(self, text: str, voice_id: str) -> bytes:
        self.le_da_goi.append(text)
        if text == self._hong_cau:
            raise RuntimeError("câu này hỏng")
        return pcm()


def test_backend_co_gop_lo_thi_khong_goi_tung_cau_mot():
    backend = LoBackend(batch_size=32)
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"), (6.0, 8.0, "c"))

    fitted, warnings = synthesize_segments(backend, segs, "v", total_duration=20.0)

    assert len(fitted) == 3
    assert not warnings
    assert backend.lo_da_goi == [["a", "b", "c"]]   # đúng MỘT lượt gọi cho cả ba
    assert backend.le_da_goi == []


def test_lo_hong_thi_lui_ve_tung_cau_de_chi_mat_dung_cau_hong():
    """Nếu cả lô chết mà ta bỏ luôn cả lô thì một câu xấu làm mất 32 câu tốt."""
    backend = LoBackend(batch_size=32, hong_lo=True, hong_cau="b")
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"), (6.0, 8.0, "c"))

    fitted, warnings = synthesize_segments(backend, segs, "v", total_duration=20.0)

    assert backend.le_da_goi == ["a", "b", "c"]     # đã lùi về đọc lẻ
    assert [s.text_vi for s, _ in fitted] == ["a", "c"]   # chỉ "b" bị bỏ trống
    assert any("1/3" in w for w in warnings)


def test_huy_giua_chung_van_an():
    backend = LoBackend(batch_size=1)
    segs = segments((0.0, 2.0, "a"), (3.0, 5.0, "b"))

    with pytest.raises(JobCancelledError):
        synthesize_segments(backend, segs, "v", total_duration=20.0,
                            should_cancel=lambda: True)
    assert backend.lo_da_goi == []                  # dừng TRƯỚC khi tốn một lượt GPU nào


def test_lo_duoc_chia_deu_chu_khong_cat_thang_theo_tran():
    """Mọi hàng trong lô chạy tới khi hàng dài nhất xong, nên một lô cuối lèo tèo rất phí:
    53 câu trần 32 phải ra 27+26, không phải 32+21."""
    assert _chia_lo(53, 32) == [27, 26]
    assert _chia_lo(32, 32) == [32]
    assert _chia_lo(33, 32) == [17, 16]
    assert _chia_lo(5, 32) == [5]
    assert _chia_lo(0, 32) == []
    for n in (1, 7, 40, 100, 129):
        lo = _chia_lo(n, 32)
        assert sum(lo) == n and max(lo) <= 32
        assert max(lo) - min(lo) <= 1               # thật sự đều


def test_thoi_luong_doc_that_duoc_ghi_lai_cho_phu_de():
    backend = LoBackend(batch_size=32)
    segs = segments((0.0, 5.0, "a"))

    synthesize_segments(backend, segs, "v", total_duration=20.0)

    assert segs[0].spoken_duration is not None
    assert segs[0].spoken_duration > 0
