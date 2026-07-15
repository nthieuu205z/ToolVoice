#!/usr/bin/env python
"""Kiểm tra môi trường trước khi chạy app: ffmpeg, và từng nhà cung cấp đang cấu hình.

    ./.venv/bin/python scripts/check_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import settings  # noqa: E402
from pipeline.errors import FFmpegNotFoundError  # noqa: E402
from pipeline.ffmpeg_utils import resolve, set_binaries  # noqa: E402
from pipeline.voices import voices_for  # noqa: E402

OK, BAD, INFO = "✓", "✗", "·"


def check_ffmpeg() -> bool:
    set_binaries(settings.ffmpeg_bin, settings.ffprobe_bin)
    try:
        print(f"{OK} ffmpeg   {resolve('ffmpeg')}")
        print(f"{OK} ffprobe  {resolve('ffprobe')}")
        return True
    except FFmpegNotFoundError as exc:
        print(f"{BAD} {exc.user_message}")
        return False


def check_stt() -> bool:
    if settings.stt_provider != "whisper":
        print(f"{INFO} Nhận diện giọng nói: Gemini ({settings.gemini_stt_model}) — tốn hạn mức")
        return True
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print(f"{BAD} Thiếu faster-whisper. Chạy: uv pip install -e '.[dev]'")
        return False
    print(f"{OK} Nhận diện giọng nói: Whisper '{settings.whisper_model}' chạy trên máy, miễn phí")
    print(f"{INFO}   lần chạy đầu sẽ tải model về (~460 MB với bản 'small')")
    return True


def check_tts() -> bool:
    provider = settings.tts_provider
    voices = voices_for(provider)

    if provider == "gemini":
        print(f"{INFO} Giọng đọc: Gemini ({settings.gemini_tts_model}) — trần ~100 lượt/ngày")
        print(f"{INFO}   {len(voices)} giọng khả dụng")
        return True

    if provider == "vieneu":
        try:
            import vieneu  # noqa: F401
        except ImportError:
            print(f"{BAD} Thiếu VieNeu-TTS. Chạy: uv pip install -e '.[vieneu]'")
            return False
        print(f"{OK} Giọng đọc: VieNeu-TTS, miễn phí, chạy offline, Apache-2.0")
        print(f"{INFO}   {len(voices)} giọng bản địa (Bắc/Trung/Nam)")
        print(f"{INFO}   chậm hơn edge-tts ~8 lần; lần đầu tải model ~610 MB")
        return True

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print(f"{BAD} Thiếu edge-tts. Chạy: uv pip install -e '.[dev]'")
        return False
    print(f"{OK} Giọng đọc: edge-tts, miễn phí, không cần API key, không trần theo ngày")
    print(f"{INFO}   {len(voices)} giọng (2 giọng Việt bản địa + 12 giọng đa ngôn ngữ)")
    return True


def check_translate() -> bool:
    """Bước dịch luôn cần Gemini, nhưng chỉ 1–4 lượt gọi cho cả video."""
    if not settings.gemini_api_key:
        print(f"{BAD} Chưa có GEMINI_API_KEY (bước dịch cần nó) — tạo .env từ .env.example.")
        return False

    from google import genai

    vertex = settings.gemini_backend == "vertex"
    endpoint = "Vertex AI Express" if vertex else "Developer API (AI Studio)"
    client = genai.Client(vertexai=True, api_key=settings.gemini_api_key) if vertex \
        else genai.Client(api_key=settings.gemini_api_key)

    # Vertex Express KHÔNG hỗ trợ models.list(); cách duy nhất kiểm tra là gọi thật một lượt.
    model = settings.gemini_translate_model
    try:
        client.models.generate_content(model=model, contents="ok")
    except Exception as exc:
        code = getattr(exc, "code", None)
        print(f"{BAD} Dịch: gọi {endpoint} thất bại với model '{model}' (code={code})")
        _explain(code, vertex)
        return False

    print(f"{OK} Dịch: Gemini qua {endpoint}")
    print(f"{OK}   GEMINI_TRANSLATE_MODEL = {model}")
    return True


def _explain(code, vertex: bool) -> None:
    if code == 403:
        print("      Khóa bị chặn hoặc API chưa bật trên project.")
        if not vertex:
            print("      Thử đặt GEMINI_BACKEND=vertex trong .env (khóa từ Google Cloud),")
            print("      hoặc bật 'Generative Language API' cho project.")
    elif code == 404:
        print(f"      Model này không có trên {'Vertex' if vertex else 'Developer API'}.")
        print("      Vertex Express có: gemini-3.5-flash, gemini-2.5-flash, gemini-2.5-pro.")
    elif code == 401:
        print("      Endpoint này không nhận khóa API. Đổi GEMINI_BACKEND=developer,")
        print("      hoặc dùng service account (GOOGLE_APPLICATION_CREDENTIALS).")


if __name__ == "__main__":
    print("── Kiểm tra môi trường ToolVietSub ──\n")
    results = [check_ffmpeg(), check_stt(), check_tts(), check_translate()]
    print()
    if all(results):
        print("Tất cả sẵn sàng. Chạy app:  ./.venv/bin/uvicorn backend.main:app --port 8000")
    else:
        print("Còn thiếu vài thứ ở trên — sửa xong rồi chạy lại script này.")
        sys.exit(1)
