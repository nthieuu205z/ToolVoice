"""Lỗi của pipeline. Mỗi lỗi mang sẵn thông điệp tiếng Việt hiển thị được cho người dùng."""


class PipelineError(Exception):
    """Lỗi gốc — `user_message` là chuỗi an toàn để hiện lên giao diện."""

    user_message = "Không thể hoàn tất quá trình xử lý."

    def __init__(self, message: str = "", user_message: str | None = None):
        super().__init__(message or self.user_message)
        if user_message:
            self.user_message = user_message


class FFmpegNotFoundError(PipelineError):
    user_message = (
        "Không tìm thấy ffmpeg trên máy. Cài bằng `brew install ffmpeg`, "
        "hoặc đặt FFMPEG_BIN và FFPROBE_BIN trong file .env."
    )


class FFmpegError(PipelineError):
    user_message = "Xử lý video/âm thanh bằng ffmpeg thất bại."


class UnsupportedMediaError(PipelineError):
    user_message = "File này không phải video hợp lệ hoặc không đọc được."


class NoSpeechDetectedError(PipelineError):
    user_message = "Không tìm thấy lời thoại nào trong video."


class JobCancelledError(PipelineError):
    """Người dùng bấm hủy. Không phải lỗi — pipeline dừng ở mốc an toàn gần nhất."""

    user_message = "Đã hủy xử lý video."


class GeminiAPIError(PipelineError):
    user_message = "Không gọi được Gemini API. Kiểm tra khóa API và kết nối mạng."


class SpeechServiceError(PipelineError):
    """Dịch vụ giọng đọc ngoài Gemini (ví dụ edge-tts) không trả về âm thanh."""

    user_message = (
        "Dịch vụ giọng đọc miễn phí đang từ chối. Thử lại sau ít phút, "
        "hoặc đổi TTS_PROVIDER=gemini trong .env."
    )


class QuotaExhaustedError(PipelineError):
    """Hạn mức theo NGÀY đã cạn. Thử lại vô nghĩa — reset phải đợi tới hôm sau."""

    user_message = (
        "Đã hết hạn mức Gemini TTS trong ngày (bản miễn phí khoảng 100 lượt/ngày). "
        "Những lời thoại sau đó bị bỏ trống. Đợi sang ngày mới hoặc bật thanh toán để nâng hạn mức."
    )


class TranslationAlignmentError(PipelineError):
    user_message = "Bản dịch trả về không khớp số lời thoại gốc nên phải dừng để tránh lệch tiếng."
