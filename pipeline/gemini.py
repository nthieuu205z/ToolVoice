"""Bản hiện thực GeminiBackend — nơi duy nhất trong pipeline gọi API thật.

Các bước xử lý chỉ phụ thuộc vào giao thức GeminiBackend (pipeline/models.py), nên
test tiêm bản giả và không tốn một token nào.

Hai cạm bẫy của Developer API (khóa AI Studio) đã đo được và xử lý ở đây:
- `audio_timestamp` chỉ tồn tại trên Vertex → tự dò rồi bỏ.
- Một số model TTS từ chối `language_code` bằng cách trả HTTP 200 với 0 part và
  finish_reason=OTHER, KHÔNG ném exception → phải phát hiện bằng "không có audio".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from .audio import parse_pcm_rate, resample_pcm
from .errors import GeminiAPIError, QuotaExhaustedError
from .models import TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

# Lỗi tạm thời phía Google — thử lại có ích.
_RETRYABLE_CODES = {408, 500, 502, 503, 504}
# Hạn mức theo phút thì đợi là qua; theo ngày thì đợi vô nghĩa (retryDelay ~19113s).
_RETRY_DELAY_IS_HOPELESS = 120.0
# Đợi theo lời Google, nhưng đừng treo job quá lâu vì một câu.
_MAX_HONOURED_DELAY = 65.0
# Model TTS thỉnh thoảng tưởng lời thoại là câu lệnh (hay gặp với câu hỏi) và định trả lời.
_TTS_TEXT_CONFUSION = "should only be used for TTS"


def _error_body(exc: genai_errors.APIError) -> dict:
    body = getattr(exc, "details", None)
    return body.get("error", {}) if isinstance(body, dict) else {}


def _retry_delay_seconds(exc: genai_errors.APIError) -> float | None:
    """Khoảng đợi do chính Google chỉ định trong RetryInfo."""
    for detail in _error_body(exc).get("details", []):
        if str(detail.get("@type", "")).endswith("RetryInfo"):
            try:
                return float(str(detail.get("retryDelay", "")).rstrip("s"))
            except ValueError:
                return None
    return None


def _quota_scope(exc: genai_errors.APIError) -> str:
    """429 vì hạn mức theo NGÀY hay theo PHÚT? Đợi qua phút thì được, qua ngày thì không."""
    for detail in _error_body(exc).get("details", []):
        if str(detail.get("@type", "")).endswith("QuotaFailure"):
            for violation in detail.get("violations", []):
                if "PerDay" in str(violation.get("quotaId", "")):
                    return "day"

    delay = _retry_delay_seconds(exc)
    return "day" if delay is not None and delay > _RETRY_DELAY_IS_HOPELESS else "minute"


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 429:
            return _quota_scope(exc) != "day"
        if exc.code == 400:
            # "Model tried to generate text" là ngẫu nhiên — lần sau thường đọc bình thường.
            return _TTS_TEXT_CONFUSION in str(_error_body(exc).get("message", ""))
        return exc.code in _RETRYABLE_CODES
    return isinstance(exc, (ConnectionError, TimeoutError))


_backoff = wait_exponential(multiplier=2, min=2, max=30)


def _wait_strategy(retry_state) -> float:
    """Ưu tiên khoảng đợi Google chỉ định; nếu không có thì lùi theo hàm mũ.

    Backoff cứng bỏ cuộc sau ~14 giây, trong khi 429-theo-phút bảo đợi lâu hơn thế —
    đó là lý do vài lượt thoại rơi vào im lặng dù hạn mức ngày vẫn còn.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, genai_errors.APIError) and exc.code == 429:
        delay = _retry_delay_seconds(exc)
        if delay is not None:
            return min(delay + 1.0, _MAX_HONOURED_DELAY)
    return _backoff(retry_state)


_retry = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=_wait_strategy,
    reraise=True,
)


# ─── Lược đồ JSON bắt Gemini trả về đúng cấu trúc ────────────────────

class _ClipTranscript(BaseModel):
    language: str
    text: str


class _TranslatedLine(BaseModel):
    index: int
    text_vi: str


class _Translation(BaseModel):
    lines: list[_TranslatedLine]


_STT_PROMPT = """\
Chép nguyên văn lời thoại trong đoạn audio này.

- `language`: mã ISO 639-1 của ngôn ngữ đang nói (ví dụ "en", "ja", "zh", "ko").
- `text`: nguyên văn bằng ngôn ngữ gốc, KHÔNG dịch, không tóm tắt, không thêm bớt.
  Bỏ qua nhạc nền và tiếng động. Nếu không có lời nói nào, trả về chuỗi rỗng.
- Không thêm mốc thời gian, không thêm chú thích.
"""

_TRANSLATE_PROMPT = """\
Bạn là biên dịch viên lồng tiếng phim chuyên nghiệp. Hãy dịch từng dòng thoại sang
TIẾNG VIỆT tự nhiên như người Việt nói chuyện.

Quy tắc bắt buộc:
- Trả về ĐÚNG {count} dòng, giữ nguyên `index` của từng dòng đầu vào.
- Dịch ĐẦY ĐỦ mọi ý của câu gốc. TUYỆT ĐỐI không lược bỏ thông tin, không tóm tắt,
  không gộp ý — thiếu ý là lỗi nặng hơn dài dòng.
- `duration_seconds` là khung thời gian của câu gốc (giọng đọc chạy khoảng 16 ký tự
  mỗi giây, `max_chars` là mức vừa khung). Hãy CHỌN CÁCH DIỄN ĐẠT gọn và khẩu ngữ
  tự nhiên để đọc kịp; nhưng khi phải chọn giữa đủ ý và ngắn, luôn chọn đủ ý.
- Giữ nhất quán tên riêng, xưng hô và thuật ngữ xuyên suốt toàn bộ video.
- Chỉ trả về lời thoại đã dịch, không chú thích, không dấu ngoặc mô tả.
{context}
Các dòng cần dịch (JSON):
{payload}
"""

# Đo thực nghiệm: 560 ký tự tiếng Việt → 33.7 giây giọng đọc.
CHARS_PER_SECOND = 16.6

# Không đưa lời thoại trần vào TTS: gặp câu hỏi, model tưởng là câu lệnh và định trả lời
# ("Model tried to generate text, but it should only be used for TTS"). Đã kiểm chứng
# rằng câu lệnh này KHÔNG bị đọc thành tiếng, chỉ nội dung phía sau mới được đọc.
_TTS_INSTRUCTION = (
    "Đọc to nguyên văn đoạn văn bản sau bằng giọng tự nhiên. "
    "Không trả lời, không bình luận, không thêm bớt:\n\n"
)


class GeminiRunner:
    """Bọc google-genai: chuẩn hóa cấu hình, thử lại, và bóc dữ liệu ra kiểu của pipeline."""

    def __init__(
        self,
        api_key: str,
        *,
        stt_model: str,
        translate_model: str,
        tts_model: str,
        tts_language_code: str = "vi-VN",
        use_vertex: bool = False,
    ):
        # Vertex AI Express Mode nhận cùng một khóa API nhưng đi tới endpoint khác
        # (aiplatform.googleapis.com thay vì generativelanguage.googleapis.com).
        # Dùng khi khóa của bạn nằm trong project Google Cloud chưa bật Generative Language API.
        self._client = genai.Client(vertexai=True, api_key=api_key) if use_vertex \
            else genai.Client(api_key=api_key)
        self._use_vertex = use_vertex
        self._stt_model = stt_model
        self._translate_model = translate_model
        self._tts_model = tts_model
        self._tts_language_code = tts_language_code

        # Cờ tự dò, nhớ một lần rồi thôi (dùng chung giữa các luồng TTS).
        self._tts_language_code_rejected = False
        # Model nào không nhận thinking_config thì nhớ luôn (dùng chung giữa các luồng STT).
        self._stt_thinking_rejected = False
        # Chỉ để báo cáo — không dùng để chặn các lượt sau (xem docstring của synthesize).
        self._quota_exhausted = False
        self._warned_rate: int | None = None

    # ─── nhận diện giọng nói ────────────────────────────────────────

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        """Chép lời thoại của MỘT đoạn audio ngắn. Mốc thời gian do ffmpeg quyết định.

        Gửi audio thẳng trong request (inline) thay vì qua Files API: mỗi đoạn chỉ vài
        trăm KB, nên tiết kiệm được hai vòng HTTP upload + delete cho mỗi đoạn.
        """
        audio = types.Part.from_bytes(data=wav_path.read_bytes(), mime_type="audio/wav")
        response = self._generate_transcript(audio)
        parsed = self._parse(response, _ClipTranscript)
        return parsed.language or "", parsed.text.strip()

    @_retry
    def _generate_transcript(self, audio: types.Part) -> types.GenerateContentResponse:
        thinking_off = not self._stt_thinking_rejected
        try:
            return self._transcript_call(audio, thinking_off=thinking_off)
        except genai_errors.APIError as exc:
            if exc.code == 400 and thinking_off:
                # Model không nhận thinking_config (đổi model qua .env chẳng hạn) — nhớ và thử lại.
                log.info("Model STT từ chối thinking_config — thử lại không kèm trường này")
                self._stt_thinking_rejected = True
                try:
                    return self._transcript_call(audio, thinking_off=False)
                except genai_errors.APIError as retry_exc:
                    raise self._wrap(retry_exc, "nhận diện giọng nói")
            raise self._wrap(exc, "nhận diện giọng nói")

    def _transcript_call(self, audio: types.Part, *, thinking_off: bool):
        config_kwargs: dict = dict(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=_ClipTranscript,
        )
        if thinking_off:
            # Chép lời không cần suy luận. Đo trên gemini-3.5-flash qua Vertex:
            # 4,2–18,9 giây/request khi để mặc định → 2,3–2,7 giây khi tắt thinking.
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return self._client.models.generate_content(
            model=self._stt_model,
            contents=[_STT_PROMPT, audio],
            config=types.GenerateContentConfig(**config_kwargs),
        )

    # ─── dịch ───────────────────────────────────────────────────────

    def translate(self, texts: list[str], durations: list[float], context: str = "") -> list[str]:
        payload = json.dumps(
            [
                {
                    "index": i,
                    "duration_seconds": round(d, 1),
                    "max_chars": max(20, int(d * CHARS_PER_SECOND)),
                    "text": t,
                }
                for i, (t, d) in enumerate(zip(texts, durations))
            ],
            ensure_ascii=False,
        )
        prompt = _TRANSLATE_PROMPT.format(
            count=len(texts),
            context=f"\nNgữ cảnh các dòng ngay trước đó (đã dịch):\n{context}\n" if context else "",
            payload=payload,
        )
        parsed = self._parse(self._generate_translation(prompt), _Translation)
        by_index = {line.index: line.text_vi.strip() for line in parsed.lines}
        return [by_index.get(i, "") for i in range(len(texts))]

    @_retry
    def _generate_translation(self, prompt: str) -> types.GenerateContentResponse:
        try:
            return self._client.models.generate_content(
                model=self._translate_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                    response_schema=_Translation,
                ),
            )
        except genai_errors.APIError as exc:
            raise self._wrap(exc, "dịch lời thoại")

    # ─── tạo giọng đọc ──────────────────────────────────────────────

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Mỗi lượt thoại được một cơ hội, kể cả khi hạn mức ngày đã báo cạn.

        Hạn mức của Google rò rỉ: đo thực tế thấy 3/8 request vẫn qua sau khi đã nhận 429
        theo ngày. Chặn hết từ lỗi đầu tiên là vứt oan những lượt lẽ ra đọc được. Cái phải
        bỏ là RETRY (429 theo ngày bảo đợi hơn 4 tiếng), không phải bản thân lần thử.
        """
        response = self._generate_speech(text, voice_id)
        for part in self._parts(response):
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                return self._normalize_rate(inline.data, parse_pcm_rate(inline.mime_type))

        raise GeminiAPIError(
            f"Gemini TTS không trả về âm thanh (model={self._tts_model}, "
            f"finish_reason={self._finish_reason(response)}, "
            f"language_code={'có' if not self._tts_language_code_rejected else 'không'})"
        )

    @_retry
    def _generate_speech(self, text: str, voice_id: str) -> types.GenerateContentResponse:
        with_language = not self._tts_language_code_rejected
        try:
            response = self._speech_call(text, voice_id, with_language=with_language)
        except genai_errors.APIError as exc:
            if exc.code == 400 and with_language:
                self._reject_language_code("model trả lỗi 400")
                try:
                    return self._speech_call(text, voice_id, with_language=False)
                except genai_errors.APIError as retry_exc:
                    raise self._wrap(retry_exc, "tạo giọng đọc")
            raise self._wrap(exc, "tạo giọng đọc")

        # Cạm bẫy thật: HTTP 200, finish_reason=OTHER, không part nào — không có exception nào để bắt.
        if with_language and not self._has_audio(response):
            self._reject_language_code(f"model trả 200 nhưng không có audio "
                                       f"(finish_reason={self._finish_reason(response)})")
            return self._speech_call(text, voice_id, with_language=False)
        return response

    def _reject_language_code(self, reason: str) -> None:
        if not self._tts_language_code_rejected:
            log.info("Model TTS từ chối language_code (%s) — thử lại không kèm trường này", reason)
        self._tts_language_code_rejected = True

    def _speech_call(self, text: str, voice_id: str, *, with_language: bool):
        speech_kwargs: dict = {
            "voice_config": types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_id)
            )
        }
        if with_language and self._tts_language_code:
            speech_kwargs["language_code"] = self._tts_language_code

        return self._client.models.generate_content(
            model=self._tts_model,
            contents=_TTS_INSTRUCTION + text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(**speech_kwargs),
            ),
        )

    def _normalize_rate(self, pcm: bytes, rate: int | None) -> bytes:
        """Model có thể đổi tần số lấy mẫu; ép về TTS_SAMPLE_RATE thay vì tin vào mặc định."""
        if rate is None:
            if self._warned_rate != -1:
                log.warning("Phản hồi TTS không ghi tần số lấy mẫu, đang giả định %d Hz", TTS_SAMPLE_RATE)
                self._warned_rate = -1
            return pcm
        if rate == TTS_SAMPLE_RATE:
            return pcm
        if rate != self._warned_rate:
            log.warning("Gemini TTS trả về %d Hz, đang chuyển về %d Hz", rate, TTS_SAMPLE_RATE)
            self._warned_rate = rate
        return resample_pcm(pcm, rate, TTS_SAMPLE_RATE)

    # ─── tiện ích ───────────────────────────────────────────────────

    @staticmethod
    def _parts(response: types.GenerateContentResponse):
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return []
        content = getattr(candidates[0], "content", None)
        return (getattr(content, "parts", None) or []) if content else []

    @classmethod
    def _has_audio(cls, response: types.GenerateContentResponse) -> bool:
        return any(
            getattr(part, "inline_data", None) is not None and part.inline_data.data
            for part in cls._parts(response)
        )

    @staticmethod
    def _finish_reason(response: types.GenerateContentResponse):
        candidates = getattr(response, "candidates", None) or []
        return getattr(candidates[0], "finish_reason", None) if candidates else None

    @staticmethod
    def _parse(response: types.GenerateContentResponse, schema: type[BaseModel]):
        """`response.parsed` là đối tượng đã dựng sẵn; rơi về JSON thô khi SDK không dựng được."""
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        raw = (getattr(response, "text", None) or "").strip()
        if not raw:
            raise GeminiAPIError("Gemini trả về phản hồi rỗng")
        try:
            return schema.model_validate_json(raw)
        except Exception as exc:
            raise GeminiAPIError(f"Gemini trả về JSON không đúng cấu trúc: {raw[:200]}") from exc

    def _wrap(self, exc: genai_errors.APIError, action: str) -> Exception:
        """Lỗi tạm thời trả nguyên trạng cho tenacity; lỗi vĩnh viễn đổi thành thông điệp tiếng Việt."""
        if exc.code == 429 and _quota_scope(exc) == "day":
            self._quota_exhausted = True
            return QuotaExhaustedError()
        if _is_retryable(exc):
            return exc
        if exc.code in (401, 403):
            return GeminiAPIError(
                str(exc),
                user_message="Khóa Gemini API không hợp lệ hoặc không có quyền. Kiểm tra GEMINI_API_KEY trong .env.",
            )
        if exc.code == 404:
            return GeminiAPIError(
                str(exc),
                user_message=f"Không tìm thấy model Gemini khi {action}. "
                             f"Chạy `python scripts/check_setup.py` để xem model khả dụng.",
            )
        return GeminiAPIError(str(exc), user_message=f"Gemini báo lỗi khi {action}: {exc}")
