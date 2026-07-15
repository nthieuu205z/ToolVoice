"""Giọng đọc tiếng Việt bằng edge-tts — miễn phí, không cần API key, không trần theo ngày.

Đo thực nghiệm trên dịch vụ này:
- Bắn 6 luồng song song: chỉ 25/32 request thành công.
- Hạ xuống 2 luồng kèm thử lại: 10/10, mỗi lượt ~0,9 giây.
- Thỉnh thoảng từ chối vài phút rồi tự hồi (NoAudioReceived), nên retry là bắt buộc.

Đây là endpoint đọc-thành-tiếng của trình duyệt Edge, Microsoft không cam kết gì về nó.
Nếu một ngày nó ngừng chạy, đổi TTS_PROVIDER về `gemini` trong .env là quay lại được ngay.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from .audio import decode_to_pcm
from .errors import SpeechServiceError
from .models import TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

DEFAULT_ATTEMPTS = 5
_BACKOFF_SECONDS = 3.0

# Trần TOÀN CỤC cho số request bắn đồng thời vào edge-tts, tính trên cả tiến trình.
# Đo thực nghiệm: 6 luồng song song → 25/32 thành công; 2 luồng → 10/10. Khi nhiều
# job chạy cùng lúc, mỗi job đều có TTS_WORKERS luồng riêng — không có trần chung
# này thì hai job là đủ để bị bóp tần suất trở lại.
_request_slots = threading.BoundedSemaphore(2)


class EdgeSynthesizer:
    """Bám giao thức `synthesize` của pipeline: trả PCM 16-bit mono TTS_SAMPLE_RATE Hz."""

    def __init__(self, attempts: int = DEFAULT_ATTEMPTS):
        self._attempts = max(1, attempts)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Chạy phần async trong luồng gọi — mỗi worker có event loop riêng của nó."""
        mp3 = asyncio.run(self._synthesize_async(text, voice_id))
        return decode_to_pcm(mp3, TTS_SAMPLE_RATE)

    async def _synthesize_async(self, text: str, voice_id: str) -> bytes:
        import edge_tts  # nạp muộn để test không cần gói này

        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                # Slot chỉ giữ trong lúc request bay — giữ cả lúc ngủ chờ retry thì
                # một job đang bị bóp tần suất sẽ chặn luôn job khỏe.
                with _request_slots:
                    communicate = edge_tts.Communicate(text, voice_id)
                    chunks = [
                        chunk["data"]
                        async for chunk in communicate.stream()
                        if chunk["type"] == "audio"
                    ]
                audio = b"".join(chunks)
                if audio:
                    return audio
                last = SpeechServiceError("edge-tts trả về 0 byte âm thanh")
            except Exception as exc:  # NoAudioReceived, lỗi mạng, bị bóp tần suất…
                last = exc

            if attempt + 1 < self._attempts:
                await asyncio.sleep(_BACKOFF_SECONDS * (attempt + 1))

        raise SpeechServiceError(
            f"edge-tts không tạo được giọng đọc sau {self._attempts} lần thử: {last}"
        )
