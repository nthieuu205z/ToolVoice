"""Cấu hình đọc từ .env. Khóa API không bao giờ nằm trong mã nguồn."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""

    # Khóa API đi tới endpoint nào:
    #   developer — generativelanguage.googleapis.com (khóa từ aistudio.google.com)
    #   vertex    — aiplatform.googleapis.com, Vertex AI Express Mode (khóa từ Google Cloud)
    # Cùng một khóa nhưng hai endpoint; project nào bật API nào thì dùng cái đó.
    gemini_backend: str = "developer"

    # Nhà cung cấp cho từng bước. Mặc định né hẳn trần 100 lượt/ngày của Gemini TTS.
    stt_provider: str = "whisper"   # whisper (miễn phí, chạy trên máy) | gemini
    tts_provider: str = "edge"      # edge | vieneu (offline) | gemini
    # Bước dịch luôn dùng Gemini — chỉ tốn 1–4 lượt gọi cho cả video.

    # Model Gemini — đổi được khi Google cập nhật, không cần sửa code.
    gemini_stt_model: str = "gemini-3.5-flash"
    gemini_translate_model: str = "gemini-3.5-flash"
    gemini_tts_model: str = "gemini-2.5-flash-preview-tts"

    # Whisper chạy trên máy: tiny | base | small | medium | large-v3 (càng lớn càng chậm, càng chuẩn)
    whisper_model: str = "small"
    whisper_compute_type: str = "int8"

    # edge-tts bóp tần suất — thử lại là bắt buộc.
    edge_tts_attempts: int = 5

    # VieNeu mặc định nhúng dấu chìm vào audio; ta tắt, bật lại nếu bạn muốn.
    vieneu_watermark: bool = False

    # Số lượt thoại VieNeu gộp vào MỘT lượt gọi GPU. 0 = tự chọn theo dung lượng card.
    # Đây là đòn bẩy hiệu năng lớn nhất của cả pipeline (từng câu một: 2,3 lần thời gian
    # thực; gộp 32 câu: ~44 lần) — lý do và số đo nằm ở pipeline/vieneu_batch.py.
    # Không ảnh hưởng gì tới đường CPU/ONNX, vốn luôn đọc từng câu một.
    vieneu_batch_size: int = 0

    # ffmpeg: để trống thì tìm trong PATH.
    ffmpeg_bin: str = ""
    ffprobe_bin: str = ""

    # Tinh chỉnh pipeline. Trần lượt phát ngôn là hạt đồng bộ hình–tiếng:
    # càng ngắn càng bám sát mốc câu gốc (xem pipeline/segmentation.py).
    max_utterance_seconds: float = 12.0
    max_utterance_gap: float = 0.5
    # Dùng mốc thời gian cấp CÂU của Whisper: mỗi câu là một lượt đọc đặt đúng thời điểm câu
    # tiếng Anh, thay vì nhồi cả vùng ffmpeg (2–4 câu) làm một khối. Bám hình sát hơn, VieNeu
    # ổn định hơn (đọc từng câu), hết cảnh "Hôm ...(nghỉ)... nay". Đặt False để về cấp vùng.
    sentence_level_timing: bool = True
    # Gemini STT chịu được nhiều luồng song song (đặt 8 trong .env); Whisper thì
    # tự tuần tự hóa bên trong nên chạy whisper hãy hạ về 1 cho khỏi tranh CPU.
    stt_workers: int = 1
    # Đo thực nghiệm: edge-tts 6 luồng → 25/32 hỏng; 2 luồng → 10/10.
    tts_workers: int = 2
    # Số lô dịch chạy song song. Nút cổ chai là model xuất token, không phải CPU — đo thật:
    # 4 lô tuần tự 63,7s → 6 lô song song ~11s. Vertex chịu được; hạ về 2–3 nếu dùng
    # Developer API free tier (RPM thấp) để tránh 429.
    translate_workers: int = 6
    # Trần tăng tốc để ép câu tiếng Việt (dài hơn khe gốc) vừa khung. 1,5× nghe rõ là
    # "nói nhanh"; 1,3× êm hơn hẳn mà timing vẫn khá sát — phần dư tràn sang câu sau rồi
    # tự tan ở khoảng lặng kế tiếp (xem pipeline/audio.py::fit_to_window). Nới lên nếu
    # muốn dub bám hình chặt hơn, hạ xuống nếu muốn giọng êm hơn nữa.
    tts_max_speedup: float = 1.3
    # Sàn kéo-CHẬM để lấp khung khi tiếng Việt đọc xong sớm hơn hình (VieNeu đọc ~1,6×
    # nhanh hơn giọng Anh). 0,9× kéo dài thêm tối đa ~11% — dưới ngưỡng tai — để bám hình
    # thay vì để im lặng cụt lủn. Đặt 1,0 để tắt (giữ hành vi cũ "không bao giờ kéo chậm").
    # Chỉ áp cho câu NGẮN hơn khung; câu dài vẫn tăng tốc như thường.
    tts_fill_slowdown: float = 0.9
    tts_daily_budget: int = 90
    max_upload_mb: int = 8192

    # Số video xử lý đồng thời. Đặt thấp có chủ ý: Whisper/VieNeu bị khóa suy luận
    # toàn cục và edge-tts bị trần 2 request đồng thời — job thứ ba chủ yếu chen hàng
    # chứ không nhanh thêm. Video vượt trần sẽ xếp hàng chờ, không bị từ chối.
    max_concurrent_jobs: int = 2

    @property
    def provider_config(self):
        from pipeline.backends import ProviderConfig

        return ProviderConfig(
            stt_provider=self.stt_provider,
            tts_provider=self.tts_provider,
            gemini_api_key=self.gemini_api_key,
            gemini_backend=self.gemini_backend,
            gemini_stt_model=self.gemini_stt_model,
            gemini_translate_model=self.gemini_translate_model,
            gemini_tts_model=self.gemini_tts_model,
            whisper_model=self.whisper_model,
            whisper_compute_type=self.whisper_compute_type,
            edge_tts_attempts=self.edge_tts_attempts,
            vieneu_watermark=self.vieneu_watermark,
        )

    @property
    def model_specs(self) -> list:
        """Các model cần tải về máy, theo nhà cung cấp đang cấu hình."""
        from pipeline.model_store import vieneu_spec, whisper_spec

        specs = []
        if self.stt_provider == "whisper":
            specs.append(whisper_spec(self.whisper_model))
        if self.tts_provider == "vieneu":
            specs.append(vieneu_spec())
        return specs

    @property
    def jobs_dir(self) -> Path:
        path = ROOT / "jobs"
        path.mkdir(exist_ok=True)
        return path

    @property
    def static_dir(self) -> Path:
        return ROOT / "web" / "static"

    @property
    def previews_dir(self) -> Path:
        return self.static_dir / "previews"

    @property
    def custom_voices_dir(self) -> Path:
        return ROOT / "custom_voices"


settings = Settings()
