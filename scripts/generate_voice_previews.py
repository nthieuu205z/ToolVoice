#!/usr/bin/env python
"""Tạo các file nghe thử giọng đọc cho giao diện. Chỉ cần chạy một lần.

    ./.venv/bin/python scripts/generate_voice_previews.py

Ghi ra web/static/previews/<VoiceId>.wav bằng đúng nhà cung cấp đang cấu hình
(TTS_PROVIDER trong .env). Dùng --force để tạo lại các file đã có.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from pipeline.audio import pcm_to_array, write_wav  # noqa: E402
from pipeline.backends import build_backend  # noqa: E402
from pipeline.ffmpeg_utils import set_binaries  # noqa: E402
from pipeline.voices import voices_for  # noqa: E402

PREVIEW_TEXT = "Xin chào, đây là giọng đọc tiếng Việt dùng để lồng tiếng cho video của bạn."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="tạo lại cả file đã có")
    args = parser.parse_args()

    set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)
    if settings.tts_provider == "gemini" and not settings.gemini_api_key:
        print("Chưa có GEMINI_API_KEY — tạo .env từ .env.example rồi điền khóa.", file=sys.stderr)
        return 1

    backend = build_backend(settings.provider_config)
    settings.previews_dir.mkdir(parents=True, exist_ok=True)
    voices = voices_for(settings.tts_provider)
    print(f"Nhà cung cấp giọng đọc: {settings.tts_provider} ({len(voices)} giọng)\n")

    failed = 0
    for voice in voices:
        dest = settings.previews_dir / f"{voice.id}.wav"
        if dest.is_file() and not args.force:
            print(f"  bỏ qua {voice.id} (đã có)")
            continue
        try:
            pcm = backend.synthesize(PREVIEW_TEXT, voice.id)
            write_wav(dest, pcm_to_array(pcm))
            print(f"  đã tạo {voice.id}.wav")
        except Exception as exc:
            failed += 1
            print(f"  lỗi với giọng {voice.id}: {exc}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
