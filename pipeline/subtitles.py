"""Dựng file .srt từ các lượt phát ngôn đã dịch.

Một lượt phát ngôn có thể dài tới 30 giây, quá dài cho một phụ đề. Ở đây nó được
chẻ thành nhiều cue theo câu, thời lượng chia theo tỉ lệ số ký tự — xấp xỉ đủ tốt
vì tốc độ đọc gần như đều.
"""

from __future__ import annotations

import re
import textwrap
from datetime import timedelta

import srt

from .models import Segment

LINE_WIDTH = 42
MAX_LINES = 2
MAX_CUE_CHARS = LINE_WIDTH * MAX_LINES
MIN_CUE_SECONDS = 1.0
MAX_CUE_SECONDS = 7.0
# Khe hở giữa hai phụ đề liền nhau, đủ để trình phát không nhập nhèm hai cue.
CUE_GAP = 0.04

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")


def wrap_text(text: str, width: int = LINE_WIDTH, max_lines: int = MAX_LINES) -> str:
    """Ngắt dòng cho dễ đọc; câu quá dài thì nới rộng dòng thay vì tràn thành nhiều dòng."""
    text = " ".join(text.split())
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width)
    if len(lines) > max_lines:
        wider = max(width, -(-len(text) // max_lines))
        lines = textwrap.wrap(text, width=wider)
    return "\n".join(lines[:max_lines] if len(lines) <= max_lines else _merge(lines, max_lines))


def _merge(lines: list[str], max_lines: int) -> list[str]:
    """Gộp phần thừa vào dòng cuối để không mất chữ khi câu quá dài."""
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :])
    return [*head, tail]


def split_into_cues(text: str, max_chars: int = MAX_CUE_CHARS) -> list[str]:
    """Chẻ một lượt phát ngôn thành các mẩu đủ ngắn để hiện lên màn hình.

    Ưu tiên cắt ở hết câu, rồi tới hết mệnh đề, cuối cùng mới cắt theo số từ.
    """
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces = _split_by(_SENTENCE_END, text, max_chars)
    return [p for p in pieces if p]


def _split_by(pattern: re.Pattern, text: str, max_chars: int) -> list[str]:
    """Cắt theo `pattern` rồi gom lại thành mẩu ≤ max_chars; mẩu nào còn dài thì chẻ tiếp."""
    parts = pattern.split(text)
    if len(parts) == 1:
        return _split_deeper(text, max_chars)

    cues: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer} {part}".strip()
        if buffer and len(candidate) > max_chars:
            cues.extend(_ensure_short(buffer, max_chars))
            buffer = part
        else:
            buffer = candidate
    cues.extend(_ensure_short(buffer, max_chars))
    return cues


def _ensure_short(text: str, max_chars: int) -> list[str]:
    return [text] if len(text) <= max_chars else _split_deeper(text, max_chars)


def _split_deeper(text: str, max_chars: int) -> list[str]:
    """Hết câu không cắt được thì thử mệnh đề, rồi mới đành cắt theo từ."""
    if _CLAUSE_END.search(text):
        return _split_by(_CLAUSE_END, text, max_chars)
    return textwrap.wrap(text, width=max_chars) or [text]


def segment_to_cues(seg: Segment) -> list[tuple[float, float, str]]:
    """Chia thời lượng của lượt phát ngôn cho các cue theo tỉ lệ số ký tự.

    Trải theo `cue_span` (thời lượng giọng đọc thật) chứ không theo khung gốc: giọng
    Việt thường đọc xong sớm hơn, trải theo khung sẽ đẩy các cue cuối vào chỗ im lặng.
    """
    text = (seg.text_vi or seg.text).strip()
    cues = split_into_cues(text)
    if not cues:
        return []

    span_total = seg.cue_span
    total_chars = sum(len(c) for c in cues)
    result: list[tuple[float, float, str]] = []
    # Bám mốc phát THẬT: lượt bị lùi (tránh đè lượt trước) thì phụ đề lùi theo.
    cursor = seg.cue_start
    for cue in cues:
        span = span_total * len(cue) / total_chars if total_chars else span_total
        result.append((cursor, cursor + span, cue))
        cursor += span
    return result


def build_srt(segments: list[Segment]) -> str:
    """Sinh nội dung .srt: kẹp thời lượng hiển thị và cắt phần chồng lấn giữa hai cue."""
    raw: list[tuple[float, float, str]] = []
    for seg in segments:
        raw.extend(segment_to_cues(seg))

    cues: list[tuple[float, float, str]] = []
    for start, end, text in raw:
        wrapped = wrap_text(text)
        if not wrapped:
            continue
        start = max(0.0, start)
        end = max(end, start + MIN_CUE_SECONDS)
        end = min(end, start + MAX_CUE_SECONDS)
        cues.append((start, end, wrapped))

    cues.sort(key=lambda c: c[0])

    subtitles = []
    for i, (start, end, text) in enumerate(cues):
        if i + 1 < len(cues):
            end = min(end, cues[i + 1][0] - CUE_GAP)
        if end <= start:
            end = start + 0.2
        subtitles.append(
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text,
            )
        )
    return srt.compose(subtitles)
