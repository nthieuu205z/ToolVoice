"""Chốt chặn cho yêu cầu cốt lõi: audio gốc phải bị loại hoàn toàn khỏi video xuất ra."""

from __future__ import annotations

from pathlib import Path

from pipeline.mux import build_mux_cmd

CMD = build_mux_cmd(Path("in.mkv"), Path("dub.wav"), Path("out.mp4"), ffmpeg="ffmpeg")


def test_maps_video_from_input_0_and_audio_from_input_1():
    assert "-map" in CMD
    pairs = [(CMD[i], CMD[i + 1]) for i, a in enumerate(CMD) if a == "-map"]
    assert pairs == [("-map", "0:v:0"), ("-map", "1:a:0")]


def test_never_maps_original_audio():
    """Nếu ai đó thêm `-map 0:a`, audio tiếng nước ngoài sẽ lọt vào output."""
    assert not any(arg.startswith("0:a") for arg in CMD)


def test_copies_video_without_reencoding():
    assert CMD[CMD.index("-c:v") + 1] == "copy"


def test_encodes_new_audio_as_aac():
    assert CMD[CMD.index("-c:a") + 1] == "aac"


def test_inputs_are_ordered_video_then_audio():
    inputs = [CMD[i + 1] for i, a in enumerate(CMD) if a == "-i"]
    assert inputs == ["in.mkv", "dub.wav"]
