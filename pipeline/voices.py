"""Danh sách giọng đọc, theo từng nhà cung cấp.

edge-tts (miễn phí, không hạn mức) chỉ có 2 giọng tiếng Việt.
Gemini có 30 giọng đa ngôn ngữ nhưng bị trần 100 lượt gọi mỗi ngày ở bản miễn phí.

Khi nhà cung cấp đổi danh sách, chỉ cần sửa dữ liệu ở đây.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    id: str            # định danh an toàn cho URL và form
    display_name: str
    # Tên mà nhà cung cấp thật sự nhận, khi khác `id` (VieNeu dùng tên tiếng Việt có dấu).
    native_id: str = ""

    @property
    def native(self) -> str:
        return self.native_id or self.id


# Giọng neural của Microsoft Edge. Không cần API key, không trần theo ngày.
#
# Hai giọng đầu là giọng tiếng Việt bản địa. Mười hai giọng còn lại là giọng
# "Multilingual" — tuy mang nhãn en-US/fr-FR/… nhưng đọc được tiếng Việt.
# Đã kiểm chứng: cho từng giọng đọc một câu tiếng Việt rồi bắt Whisper nghe lại,
# cả 12 giọng đều được nhận là tiếng Việt với độ khớp 0.94–1.00.
EDGE_VOICES: list[Voice] = [
    Voice("vi-VN-HoaiMyNeural", "Hoài My — nữ, giọng Việt bản địa"),
    Voice("vi-VN-NamMinhNeural", "Nam Minh — nam, giọng Việt bản địa"),

    Voice("en-US-AvaMultilingualNeural", "Ava — nữ, đa ngôn ngữ"),
    Voice("en-US-EmmaMultilingualNeural", "Emma — nữ, đa ngôn ngữ"),
    Voice("fr-FR-VivienneMultilingualNeural", "Vivienne — nữ, đa ngôn ngữ"),
    Voice("de-DE-SeraphinaMultilingualNeural", "Seraphina — nữ, đa ngôn ngữ"),
    Voice("pt-BR-ThalitaMultilingualNeural", "Thalita — nữ, đa ngôn ngữ"),

    Voice("en-US-AndrewMultilingualNeural", "Andrew — nam, đa ngôn ngữ"),
    Voice("en-US-BrianMultilingualNeural", "Brian — nam, đa ngôn ngữ"),
    Voice("en-AU-WilliamMultilingualNeural", "William — nam, đa ngôn ngữ"),
    Voice("fr-FR-RemyMultilingualNeural", "Rémy — nam, đa ngôn ngữ"),
    Voice("de-DE-FlorianMultilingualNeural", "Florian — nam, đa ngôn ngữ"),
    Voice("it-IT-GiuseppeMultilingualNeural", "Giuseppe — nam, đa ngôn ngữ"),
    Voice("ko-KR-HyunsuMultilingualNeural", "Hyunsu — nam, đa ngôn ngữ"),
]

# Giọng dựng sẵn của Gemini. Google không công bố giới tính từng giọng — hãy bấm nghe thử.
GEMINI_VOICES: list[Voice] = [
    Voice("Charon", "Charon — dẫn chuyện, nhiều thông tin"),
    Voice("Kore", "Kore — chắc chắn, dứt khoát"),
    Voice("Sulafat", "Sulafat — ấm áp"),
    Voice("Puck", "Puck — tươi tắn, sôi nổi"),
    Voice("Aoede", "Aoede — nhẹ nhàng, thoáng đãng"),
    Voice("Iapetus", "Iapetus — trong trẻo, rõ chữ"),
    Voice("Achird", "Achird — thân thiện"),
    Voice("Vindemiatrix", "Vindemiatrix — dịu dàng"),
    Voice("Gacrux", "Gacrux — chững chạc, trưởng thành"),
    Voice("Rasalgethi", "Rasalgethi — thuyết minh, mạch lạc"),
]

# Giọng dựng sẵn của VieNeu-TTS: chạy offline, Apache-2.0, có cả ba miền.
# `native_id` là tên VieNeu nhận; `id` là slug an toàn cho URL nghe thử.
VIENEU_VOICES: list[Voice] = [
    Voice("truc-ly", "Trúc Ly — nữ, giọng Bắc, tự nhiên", "Trúc Ly"),
    Voice("doan-trang", "Đoan Trang — nữ, giọng Bắc, tự nhiên", "Đoan Trang"),
    Voice("ngoc-linh", "Ngọc Linh — nữ, giọng Bắc, kể chuyện", "Ngọc Linh"),
    Voice("mai-anh", "Mai Anh — nữ, giọng Bắc, tin tức", "Mai Anh"),
    Voice("pham-tuyen", "Phạm Tuyên — nam, giọng Bắc, tự nhiên", "Phạm Tuyên"),
    Voice("thanh-binh", "Thanh Bình — nam, giọng Bắc, kể chuyện", "Thanh Bình"),
    Voice("minh-duc", "Minh Đức — nam, giọng Bắc, tin tức", "Minh Đức"),

    Voice("ngoc-tran", "Ngọc Trân — nữ, giọng Trung, tự nhiên", "Ngọc Trân"),
    Voice("quang-son", "Quang Sơn — nam, giọng Trung, tự nhiên", "Quang Sơn"),

    Voice("thuc-doan", "Thục Đoan — nữ, giọng Nam, kể chuyện", "Thục Đoan"),
    Voice("thuy-dung", "Thùy Dung — nữ, giọng Nam, tin tức", "Thùy Dung"),
    Voice("xuan-vinh", "Xuân Vĩnh — nam, giọng Nam, tự nhiên", "Xuân Vĩnh"),
    Voice("thai-son", "Thái Sơn — nam, giọng Nam, kể chuyện", "Thái Sơn"),
    Voice("minh-triet", "Minh Triết — nam, giọng Nam, tin tức", "Minh Triết"),
]

# OmniVoice không có giọng dựng sẵn — nó là engine nhân bản thuần, nên danh sách gốc rỗng;
# giọng của nó hoàn toàn là các giọng nhân bản người dùng đã lưu (ghép vào bên dưới).
_BY_PROVIDER = {"edge": EDGE_VOICES, "gemini": GEMINI_VOICES, "vieneu": VIENEU_VOICES,
                "omnivoice": []}

# Nhà cung cấp dùng giọng nhân bản làm giọng đọc: VieNeu (kèm giọng dựng sẵn) và OmniVoice
# (chỉ có giọng nhân bản).
_CLONE_PROVIDERS = {"vieneu", "omnivoice"}


def _clone_voices() -> list[Voice]:
    """Giọng nhân bản người dùng đã lưu; id chính là tên mà engine offline nhận."""
    from . import custom_voices

    return [
        Voice(c.id, f"{c.display_name} — giọng nhân bản", c.id)
        for c in custom_voices.list_custom()
    ]


def voices_for(provider: str) -> list[Voice]:
    base = _BY_PROVIDER.get(provider, EDGE_VOICES)
    if provider not in _CLONE_PROVIDERS:
        return base
    return base + _clone_voices()


def available_voices(tts_provider: str, clone_provider: str | None) -> list[Voice]:
    """Giọng cho giao diện/kiểm tra khi ĐỊNH TUYẾN theo loại giọng.

    = giọng dựng sẵn của `tts_provider` + giọng nhân bản (nếu có engine clone). Nhờ vậy
    dù đặt TTS_PROVIDER=edge, người dùng vẫn thấy và chọn được giọng nhân bản — chúng sẽ
    được đọc bằng engine clone (OmniVoice) qua `route_provider`.
    """
    voices = list(_BY_PROVIDER.get(tts_provider, EDGE_VOICES))
    if clone_provider in _CLONE_PROVIDERS:
        voices += _clone_voices()
    return voices


def route_provider(voice_id: str, tts_provider: str, clone_provider: str | None) -> str:
    """Engine đọc cho MỘT giọng: giọng nhân bản → engine clone; còn lại → tts_provider.

    Mỗi job chỉ dùng một giọng, nên đây là một quyết định duy nhất cho cả job (không phải
    mỗi câu). Không có engine clone thì đành trả tts_provider — không còn đường nào khác.
    """
    from . import custom_voices

    if clone_provider and custom_voices.is_custom(voice_id):
        return clone_provider
    return tts_provider


def is_valid(voice_id: str, provider: str) -> bool:
    return any(v.id == voice_id for v in voices_for(provider))


def is_available(voice_id: str, tts_provider: str, clone_provider: str | None) -> bool:
    return any(v.id == voice_id for v in available_voices(tts_provider, clone_provider))


def default_voice(provider: str) -> str:
    """Giọng mặc định của provider. OmniVoice không có giọng dựng sẵn nên khi CHƯA có giọng
    nhân bản nào, danh sách rỗng — báo lỗi rõ ràng thay vì IndexError khó hiểu."""
    voices = voices_for(provider)
    if not voices:
        raise ValueError(
            f"Nhà cung cấp '{provider}' chưa có giọng nào để chọn mặc định. OmniVoice chỉ đọc "
            f"bằng giọng nhân bản — hãy tạo/chỉ định một giọng nhân bản (id 'clone-…'), hoặc "
            f"đổi TTS_PROVIDER sang edge/vieneu."
        )
    return voices[0].id


def native_id(voice_id: str, provider: str) -> str:
    """Slug trong URL → tên mà nhà cung cấp thật sự nhận."""
    for voice in voices_for(provider):
        if voice.id == voice_id:
            return voice.native
    return voice_id
