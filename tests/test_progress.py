"""Phần trăm phải tăng đều và không bao giờ lùi khi chuyển bước."""

from __future__ import annotations

import re

import pytest

from pipeline.models import STAGES, overall_percent


def test_first_stage_starts_at_zero():
    assert overall_percent("extract", 0.0) == 0.0


def test_last_stage_ends_at_one_hundred():
    assert overall_percent("mux", 1.0) == 100.0


def test_percent_never_decreases_across_stages():
    values = [overall_percent(stage, f) for stage in STAGES for f in (0.0, 0.5, 1.0)]
    assert values == sorted(values)


def test_stage_end_equals_next_stage_start():
    for stage, nxt in zip(STAGES, STAGES[1:]):
        assert overall_percent(stage, 1.0) == overall_percent(nxt, 0.0)


@pytest.mark.parametrize("fraction", [-1.0, 2.0])
def test_out_of_range_fractions_are_clamped(fraction):
    assert 0.0 <= overall_percent("translate", fraction) <= 100.0


def test_stage_names_match_the_frontend_data_attributes():
    """app.js và index.html dựa vào đúng bảy tên này, theo đúng thứ tự này."""
    assert STAGES == ["extract", "transcribe", "translate", "synthesize",
                      "subtitle", "assemble", "mux"]


def test_subtitles_are_built_after_speech():
    """Cue phụ đề bám theo thời lượng đọc thật, nên phải chạy sau bước tạo giọng đọc."""
    assert STAGES.index("subtitle") > STAGES.index("synthesize")


def test_frontend_stage_labels_match_the_backend_order():
    """app.js hiển thị nhãn từng bước trên thẻ job — thiếu bước nào là thẻ hiện tên thô."""
    from pathlib import Path

    js = Path("web/static/app.js").read_text(encoding="utf-8")
    # Frontend mới dùng metadata có nhãn tiếng Việt thay vì map trạng thái cũ.
    order = re.findall(r"^\s{2}(\w+): \{ label: \"(?:Tách|Nhận|Dịch|Tạo|Căn|Đặt|Ghép|Xuất)", js, flags=re.MULTILINE)
    assert order == STAGES
