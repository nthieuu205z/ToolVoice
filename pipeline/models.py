"""Kiểu dữ liệu dùng chung giữa các bước của pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol

# Thứ tự các bước, khớp với `data-stage` trong web/static/index.html.
# Phụ đề dựng SAU giọng đọc: mốc thời gian của cue bám theo thời lượng đọc thật,
# nếu dựng trước thì phụ đề trải đều trên khung gốc và trôi khỏi tiếng nói.
STAGES = ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"]

# Tỉ trọng thời gian của từng bước, dùng để quy đổi ra phần trăm tổng.
STAGE_WEIGHTS = {
    "extract": 5,
    "transcribe": 25,
    "translate": 15,
    "synthesize": 40,
    "subtitle": 2,
    "assemble": 8,
    "mux": 5,
}

# Định dạng âm thanh Gemini TTS trả về, cũng là định dạng track lồng tiếng.
TTS_SAMPLE_RATE = 24_000
# Định dạng âm thanh đưa vào nhận diện giọng nói.
STT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class MediaInfo:
    """Kết quả ffprobe của video đầu vào."""

    duration: float
    video_codec: str
    has_audio: bool
    width: int = 0
    height: int = 0


@dataclass
class Segment:
    """Một lượt phát ngôn. Mốc thời gian do ffmpeg xác định, nội dung do Gemini chép."""

    start: float
    end: float
    text: str
    text_vi: str = ""
    # Thời lượng giọng đọc thật sau khi tạo, thường ngắn hơn khung gốc.
    # Phụ đề bám theo con số này để không trôi ra khỏi tiếng nói.
    spoken_duration: float = 0.0
    # Mốc phát THẬT sau bước xếp chỗ (plan_placement): thường bằng `start`, nhưng bị
    # đẩy lùi khi lượt trước tràn khung — phụ đề phải bám theo đây, không theo `start`.
    placed_start: float | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def cue_start(self) -> float:
        """Mốc bắt đầu cho phụ đề: mốc phát thật nếu đã xếp chỗ, không thì mốc gốc."""
        return self.placed_start if self.placed_start is not None else self.start

    @property
    def cue_span(self) -> float:
        """Khoảng thời gian phụ đề được trải ra: theo giọng đọc nếu đã có, không thì theo khung."""
        return self.spoken_duration if self.spoken_duration > 0 else self.duration

    def shifted(self, offset: float) -> Segment:
        return replace(self, start=self.start + offset, end=self.end + offset)


@dataclass
class PipelineResult:
    video_path: str
    srt_path: str
    language: str = ""
    segment_count: int = 0
    attempted_count: int = 0
    spoken_count: int = 0
    warnings: list[str] = field(default_factory=list)


# progress(stage, fraction_within_stage 0..1, message)
ProgressFn = Callable[[str, float, str], None]

# Trả True nghĩa là người dùng đã bấm hủy; pipeline dừng ở mốc an toàn gần nhất.
CancelFn = Callable[[], bool]


def noop_progress(stage: str, fraction: float, message: str) -> None:
    """Mặc định: không báo tiến trình (dùng cho test và CLI im lặng)."""


def never_cancel() -> bool:
    return False


def overall_percent(stage: str, fraction: float) -> float:
    """Quy đổi tiến độ trong một bước thành phần trăm của toàn bộ pipeline."""
    total = sum(STAGE_WEIGHTS.values())
    done = sum(STAGE_WEIGHTS[s] for s in STAGES[: STAGES.index(stage)])
    fraction = min(1.0, max(0.0, fraction))
    return round((done + STAGE_WEIGHTS[stage] * fraction) / total * 100, 1)


class GeminiBackend(Protocol):
    """Bề mặt Gemini mà pipeline cần. Test tiêm bản giả để không gọi API thật.

    Lưu ý: KHÔNG hỏi Gemini mốc thời gian. Trên Developer API nó bịa ra
    (xem pipeline/segmentation.py). Thời gian lấy từ ffmpeg, Gemini chỉ chép chữ.
    """

    def transcribe_clip(self, wav_path) -> tuple[str, str]:
        """Trả về (mã ngôn ngữ nguồn, nguyên văn lời thoại) của một đoạn audio ngắn."""
        ...

    def translate(self, texts: list[str], durations: list[float], context: str = "") -> list[str]:
        """Dịch sang tiếng Việt, trả về đúng số phần tử và đúng thứ tự như đầu vào."""
        ...

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Trả về PCM 16-bit little-endian, mono, TTS_SAMPLE_RATE Hz."""
        ...
