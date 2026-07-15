"""Xác định các vùng có tiếng nói bằng ffmpeg — nguồn thời gian đáng tin duy nhất.

Vì sao module này tồn tại: Gemini trên Developer API KHÔNG đo được mốc thời gian
(tham số `audio_timestamp` chỉ có trên Vertex). Khi bị hỏi giờ, nó bịa ra các mốc
đều đặn ~12 giây, khoảng lặng luôn bằng 0, và có lần trả về mốc 200s cho một đoạn
audio chỉ dài 120s. Đo thực tế trên 120 giây đầu của một video thật:

    ffmpeg:  0.79–13.37  13.79–25.00  26.27–45.24  45.70–54.42
    Gemini:  0.00–13.90  13.90–25.90  25.90–37.90  ...  (và một mốc kết thúc ở 200.0)

Nên: ffmpeg lo THỜI GIAN, Gemini lo NỘI DUNG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffmpeg_utils import resolve, run

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")

# Ngưỡng coi là im lặng. -32dB bắt được khoảng nghỉ giữa câu mà không cắt vào hơi thở.
SILENCE_NOISE_DB = -32
SILENCE_MIN_DURATION = 0.35

# Vùng ngắn hơn ngưỡng này gần như chắc chắn là tiếng động, không phải lời nói.
MIN_SPEECH_DURATION = 0.35
# Nối hai vùng cách nhau dưới ngưỡng này — nghỉ lấy hơi, không phải hết câu.
DEFAULT_MAX_GAP = 0.5
# Trần thời lượng một lượt phát ngôn. Đây là hạt đồng bộ: cả khối tiếng Việt của một
# lượt được đặt tại MỘT mốc, nên sai lệch hình–tiếng tích lũy bên trong lượt đó.
# Đo với trần 30s: câu cuối của khối có thể vang lên sớm hơn hình cả chục giây.
# 12s giữ được ngữ điệu liền câu mà sai lệch tối đa chỉ còn vài giây.
DEFAULT_MAX_DURATION = 12.0


@dataclass(frozen=True)
class Region:
    """Một lượt phát ngôn, mốc thời gian lấy từ ffmpeg nên là thật."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_silences(stderr: str) -> list[tuple[float, float]]:
    """Đọc các cặp silence_start/silence_end từ stderr của ffmpeg silencedetect."""
    starts = [float(m) for m in _SILENCE_START.findall(stderr)]
    ends = [float(m) for m in _SILENCE_END.findall(stderr)]
    # ffmpeg có thể kết thúc file khi đang trong một khoảng im lặng → thừa một start.
    return [(s, e) for s, e in zip(starts, ends) if e > s]


def detect_speech_regions(
    wav_path: Path,
    total_duration: float,
    *,
    noise_db: int = SILENCE_NOISE_DB,
    min_silence: float = SILENCE_MIN_DURATION,
) -> list[Region]:
    proc = run([
        resolve("ffmpeg"), "-hide_banner", "-nostats",
        "-i", str(wav_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ])
    silences = parse_silences(proc.stderr or "")
    return invert_silences(silences, total_duration)


def invert_silences(
    silences: list[tuple[float, float]],
    total_duration: float,
    min_speech: float = MIN_SPEECH_DURATION,
) -> list[Region]:
    """Khoảng lặng → vùng có tiếng nói (phần bù)."""
    regions: list[Region] = []
    cursor = 0.0
    for start, end in sorted(silences):
        if start - cursor >= min_speech:
            regions.append(Region(cursor, start))
        cursor = max(cursor, end)
    if total_duration - cursor >= min_speech:
        regions.append(Region(cursor, total_duration))
    return regions


def merge_regions(
    regions: list[Region],
    max_gap: float = DEFAULT_MAX_GAP,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[Region]:
    """Gộp các vùng sát nhau thành một lượt phát ngôn, không vượt `max_duration`.

    Ít lượt hơn = ít lần gọi TTS hơn, mà quota free tier chỉ cho 100 lần mỗi ngày.
    """
    if not regions:
        return []

    merged = [regions[0]]
    for region in regions[1:]:
        current = merged[-1]
        gap = region.start - current.end
        if gap < max_gap and (region.end - current.start) <= max_duration:
            merged[-1] = Region(current.start, region.end)
        else:
            merged.append(region)
    return merged


def split_long_regions(
    regions: list[Region],
    samples: np.ndarray,
    rate: int,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[Region]:
    """Cắt các vùng quá dài tại điểm im nhất gần giữa, đệ quy cho tới khi vừa trần."""
    result: list[Region] = []
    for region in regions:
        result.extend(_split(region, samples, rate, max_duration))
    return result


def _split(region: Region, samples: np.ndarray, rate: int, max_duration: float) -> list[Region]:
    if region.duration <= max_duration:
        return [region]

    cut = _quietest_point(region, samples, rate)
    if cut is None:
        # Không tìm được chỗ nghỉ nào — cắt cứng ở giữa còn hơn để một lượt dài vô hạn.
        cut = region.start + region.duration / 2

    left = Region(region.start, cut)
    right = Region(cut, region.end)
    if left.duration < MIN_SPEECH_DURATION or right.duration < MIN_SPEECH_DURATION:
        return [region]
    return _split(left, samples, rate, max_duration) + _split(right, samples, rate, max_duration)


def _quietest_point(region: Region, samples: np.ndarray, rate: int,
                    frame_ms: int = 50) -> float | None:
    """Khung 50ms có năng lượng thấp nhất trong 60% ở giữa vùng.

    Chỉ xét phần giữa để điểm cắt không rơi sát mép, sinh ra mảnh vụn.
    """
    begin = int(region.start * rate)
    finish = min(len(samples), int(region.end * rate))
    span = finish - begin
    frame = int(rate * frame_ms / 1000)
    if span <= 2 * frame:
        return None

    margin = int(span * 0.2)
    window = samples[begin + margin : finish - margin]
    frames = len(window) // frame
    if frames < 1:
        return None

    block = window[: frames * frame].astype(np.int32).reshape(frames, frame)
    energy = np.abs(block).mean(axis=1)
    quietest = int(np.argmin(energy))
    return region.start + (margin + quietest * frame + frame / 2) / rate


def plan_utterances(
    wav_path: Path,
    samples: np.ndarray,
    rate: int,
    total_duration: float,
    *,
    max_gap: float = DEFAULT_MAX_GAP,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[Region]:
    """Toàn bộ quy trình: dò im lặng → đảo thành vùng nói → gộp → cắt vùng quá dài."""
    regions = detect_speech_regions(wav_path, total_duration)
    regions = merge_regions(regions, max_gap, max_duration)
    return split_long_regions(regions, samples, rate, max_duration)
