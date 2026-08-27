#!/usr/bin/env python
"""Chạy pipeline trực tiếp, không qua web — dùng để kiểm thử nhanh trên một clip ngắn.

    ./.venv/bin/python scripts/run_pipeline_cli.py video.mp4 --voice Charon
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from pipeline.backends import build_backend  # noqa: E402
from pipeline.errors import PipelineError  # noqa: E402
from pipeline.ffmpeg_utils import set_binaries  # noqa: E402
from pipeline.models import overall_percent  # noqa: E402
from pipeline.runner import PipelineOptions, run_pipeline  # noqa: E402
from pipeline.voices import default_voice, voices_for  # noqa: E402


def main() -> int:
    voices = voices_for(settings.tts_provider)
    parser = argparse.ArgumentParser(description="Lồng tiếng Việt cho một video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--voice", default=default_voice(settings.tts_provider),
                        choices=[v.id for v in voices])
    parser.add_argument("--outdir", type=Path, default=Path("jobs/cli"))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)

    if not settings.gemini_api_key:
        print("Chưa có GEMINI_API_KEY — tạo .env từ .env.example rồi điền khóa.", file=sys.stderr)
        return 1
    if not args.video.is_file():
        print(f"Không tìm thấy file: {args.video}", file=sys.stderr)
        return 1

    def progress(stage: str, fraction: float, message: str) -> None:
        print(f"[{overall_percent(stage, fraction):5.1f}%] {stage:<11} {message}")

    print(f"nhận diện: {settings.stt_provider}   dịch: gemini   giọng đọc: {settings.tts_provider}\n")
    backend = build_backend(settings.provider_config)
    options = PipelineOptions(
        voice_id=args.voice,
        max_utterance_seconds=settings.max_utterance_seconds,
        max_utterance_gap=settings.max_utterance_gap,
        stt_workers=settings.stt_workers,
        tts_workers=settings.tts_workers,
        tts_max_speedup=settings.tts_max_speedup,
        tts_daily_budget=settings.tts_daily_budget,
        tts_is_metered=settings.tts_provider == "gemini",
        # OmniVoice tự vá lỗ hổng (postprocess) → bỏ bước đọc-lại tốn kém; VieNeu/edge vẫn cần.
        resynthesize_holes=settings.tts_provider != "omnivoice",
    )

    try:
        result = run_pipeline(backend, args.video, args.outdir, options, progress)
    except PipelineError as exc:
        print(f"\nLỗi: {exc.user_message}", file=sys.stderr)
        return 1

    print(f"\nVideo:  {result.video_path}")
    print(f"Phụ đề: {result.srt_path}")
    print(f"Ngôn ngữ gốc: {result.language or 'không rõ'} · {result.segment_count} lượt thoại")
    print(f"Đã đọc: {result.spoken_count}/{result.attempted_count} lượt")
    for warning in result.warnings:
        print(f"Lưu ý: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
