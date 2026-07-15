"""Dựng track lồng tiếng đầy đủ độ dài video từ các đoạn lời thoại rời."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .audio import duration_of, finalize_track, mix_into, silent_track, write_wav
from .models import Segment, TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

# Khe thở tối thiểu giữa hai lượt thoại khi lượt trước tràn khung.
INTER_UTTERANCE_GAP = 0.12


def plan_placement(
    fitted: list[tuple[Segment, np.ndarray]],
    rate: int = TTS_SAMPLE_RATE,
    inter_gap: float = INTER_UTTERANCE_GAP,
) -> list[tuple[float, np.ndarray]]:
    """Chốt mốc phát thật cho từng lượt: đúng mốc gốc, TRỪ KHI lượt trước còn đang đọc.

    Hai giọng đè lên nhau là tệ nhất — không nghe được lời nào. Lượt sau lùi lại vừa
    đủ để chờ lượt trước dứt, và phần lệch tự tan ở khoảng lặng kế tiếp (lượt sau nữa
    lại về đúng mốc gốc của nó). Không cắt, không bỏ chữ nào.

    Ghi `placed_start` vào từng Segment để bước phụ đề bám theo mốc phát thật.
    """
    placed: list[tuple[float, np.ndarray]] = []
    previous_end: float | None = None
    for seg, samples in fitted:
        start = seg.start
        if previous_end is not None:
            start = max(start, previous_end + inter_gap)
        seg.placed_start = start
        placed.append((start, samples))
        previous_end = start + duration_of(samples, rate)
    return placed


def build_audio_track(
    fitted: list[tuple[float, np.ndarray]],
    total_duration: float,
    out_wav: Path,
    rate: int = TTS_SAMPLE_RATE,
) -> Path:
    """Đặt từng đoạn lên nền im lặng dài đúng bằng video, rồi ghi ra WAV."""
    if not fitted:
        log.warning("Không có lượt thoại nào — track lồng tiếng sẽ im lặng hoàn toàn")

    track = silent_track(total_duration, rate)
    for start, samples in fitted:
        mix_into(track, samples, start, rate)
    return write_wav(out_wav, finalize_track(track), rate)
