"""Giọng đọc tiếng Việt bằng VieNeu-TTS — chạy hoàn toàn trên máy, Apache-2.0.

Engine do thư viện tự chọn theo phần cứng (vieneu/v3turbo.py): thấy CUDA thì chạy PyTorch
trên GPU, không thì chạy ONNX trên CPU. Hai engine đọc hai bộ trọng số khác nhau — xem
pipeline/model_store.py.

Trên GPU ta KHÔNG gọi engine từng câu một mà gộp lô (pipeline/vieneu_batch.py). Đo trên
RTX 3090, cùng 48 câu:

    từng câu một   :  2,3 lần thời gian thực   (GPU 34%,  81W)
    gộp lô 32 câu  : 44   lần thời gian thực
    gộp lô 48 câu  : 70   lần thời gian thực   (GPU 36%, 169W)

Lý do khoảng cách lớn đến vậy: vòng sinh token bị trói bởi chi phí PHÓNG kernel, không
phải sức tính — một bước backbone tốn 10,5 ms ở batch 1 và vẫn chỉ 11,7 ms ở batch 64.
Chạy nhiều tiến trình không cứu được (đã thử: chỉ 1,9 lần, vì GeForce chia lượt thời gian
giữa các CUDA context chứ không chạy song song thật).

Đổi lại: không cần mạng, không phụ thuộc endpoint không chính thức của Microsoft,
và có giọng bản địa cả ba miền Bắc/Trung/Nam.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .errors import SpeechServiceError
from .models import TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

# Nạp model tốn ~15–90 giây và vài trăm MB RAM — dùng chung cho cả tiến trình.
_engine = None
_engine_lock = threading.Lock()
# Engine không hứa an toàn khi gọi đồng thời (session ONNX có trạng thái), và mỗi job tự
# tạo một VieNeuSynthesizer riêng — nên khóa phải toàn cục chứ không per-instance.
_infer_lock = threading.Lock()

# Giọng nhân bản đã đăng ký vào engine. Đăng ký (add_voice) tốn vài giây vì phải chạy
# denoiser + speaker encoder — chỉ làm một lần mỗi giọng.
_registered_clones: set[str] = set()

# Số câu gộp vào một lô. 0 = tự chọn. Ứng dụng ghi đè bằng configure().
_batch_override = 0


def configure(*, batch_size: int = 0) -> None:
    """Ứng dụng gọi lúc khởi động, giống custom_voices.configure().

    Có hàm này để `pipeline` khỏi phải import `backend.config` — pipeline là tầng dưới,
    nó không được biết gì về tầng web.
    """
    global _batch_override
    _batch_override = max(0, batch_size)


def _uses_gpu() -> bool:
    from .model_store import uses_gpu

    return uses_gpu()


# Đo thật (RTX 3090): lô 48 câu tốn ~2,4 GB VRAM ngoài model → ~50 MB hoạt hình mỗi câu.
# Đây là thuộc tính của MODEL, gần như không đổi giữa các card — nên chia VRAM trống cho nó
# là suy ra được cỡ lô card chịu nổi. Nhờ vậy công thức tự ép tối đa mọi loại GPU thay vì
# ghim một con số: card 24 GB sẽ chạy lô lớn hơn hẳn card 8 GB.
_VRAM_PER_ITEM_GB = 0.05
# Trần: thuật toán kẹt ở chi phí PHÓNG kernel (một bước ở batch 1 và batch 64 tốn gần bằng
# nhau), nên lô >64 gần như không nhanh thêm mà câu ngắn phải chờ câu dài nhất → phí. Đây là
# "knee" đo được của THUẬT TOÁN, không phải giới hạn của card.
_BATCH_CAP = 64
_BATCH_MIN = 8


def batch_size() -> int:
    """Số câu gộp một lô. 0 = không gộp (đường ONNX/CPU chỉ sinh từng câu một).

    Suy ra từ VRAM CÒN TRỐNG lúc chạy (tự thích nghi mọi card), không ghim cứng. Ép cứng
    được qua VIENEU_BATCH_SIZE trong .env nếu muốn.
    """
    if not _uses_gpu():
        return 0
    if _batch_override > 0:
        return _batch_override
    try:
        import torch

        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024**3
    except Exception:
        return _BATCH_MIN
    # Chừa 25% VRAM trống cho phân mảnh + codec + đỉnh nhất thời; phần còn lại chia đều.
    n = int(free_gb * 0.75 / _VRAM_PER_ITEM_GB)
    return max(_BATCH_MIN, min(n, _BATCH_CAP))


def forget_clone(voice_id: str) -> None:
    """Gọi khi người dùng xóa giọng — để id không trỏ vào embedding cũ trong RAM."""
    _registered_clones.discard(voice_id)
    if _engine is None:
        return
    with _infer_lock:
        try:
            _engine.remove_voice(voice_id, save=False)
        except Exception:
            pass  # engine chưa từng đăng ký giọng này — không có gì để quên


def prewarm() -> None:
    """Nạp engine trong luồng nền lúc server khởi động — job đầu khỏi chờ.

    Chỉ nạp khi model đã nằm sẵn trên đĩa: việc tải 609 MB là do người dùng bấm
    nút trong giao diện, server không tự ý tải lúc boot.
    """
    from .model_store import is_ready, vieneu_spec

    try:
        if not is_ready(vieneu_spec()):
            log.info("VieNeu-TTS chưa tải về — bỏ qua nạp sẵn, chờ người dùng bấm tải")
            return
    except Exception as exc:
        log.warning("Không kiểm tra được model VieNeu-TTS: %s", exc)
        return

    def load() -> None:
        try:
            _get_engine()
            n = batch_size()
            if n:
                log.info("VieNeu-TTS sẵn sàng (GPU, gộp lô %d câu)", n)
            else:
                log.info("VieNeu-TTS sẵn sàng (CPU, từng câu một)")
        except Exception as exc:
            log.warning("Không nạp sẵn được VieNeu-TTS (job đầu sẽ tự nạp): %s", exc)

    threading.Thread(target=load, name="vieneu-prewarm", daemon=True).start()


def _va_torchaudio_load() -> None:
    """Cho torchaudio.load đọc bằng soundfile thay vì torchcodec.

    Engine PyTorch đọc mẫu giọng nhân bản bằng `torchaudio.load` (vieneu/_v3_turbo_engine/
    inference_v3_turbo.py). Từ torchaudio 2.9, hàm đó giao việc giải mã cho gói `torchcodec`,
    mà torchcodec lại đòi bộ DLL FFmpeg bản "shared" — bản ffmpeg static phổ biến trên Windows
    không có. Thiếu nó thì MỌI lượt thoại của giọng nhân bản đều hỏng, còn giọng dựng sẵn vẫn
    chạy: một lỗi chỉ lộ ra trên đường GPU (đường ONNX/CPU đọc audio bằng soundfile).

    Vá được vì mẫu giọng luôn là WAV PCM 16-bit do chính app ghi ra (backend/routes/voices.py
    gọi write_wav), nên soundfile đọc thẳng — rẻ hơn nhiều so với kéo thêm torchcodec và một
    bản ffmpeg thứ hai về mọi máy.
    """
    import torch
    import torchaudio

    if getattr(torchaudio.load, "_toolvietsub_va", False):
        return

    def load(uri, *args, **kwargs):
        import soundfile as sf

        data, rate = sf.read(str(uri), dtype="float32", always_2d=True)
        # soundfile trả (mẫu, kênh); torchaudio hứa (kênh, mẫu).
        return torch.from_numpy(data.T.copy()), rate

    load._toolvietsub_va = True
    torchaudio.load = load


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            try:
                _va_torchaudio_load()
            except Exception as exc:  # không có torch (đường ONNX/CPU) thì chẳng cần vá
                log.debug("Bỏ qua vá torchaudio.load: %s", exc)

            try:
                from vieneu import Vieneu  # nạp muộn: import nặng
            except ImportError as exc:
                raise SpeechServiceError(
                    "Chưa cài VieNeu-TTS",
                    user_message="Chưa cài VieNeu-TTS. Chạy: uv pip install -e '.[vieneu]', "
                                 "hoặc đổi TTS_PROVIDER=edge trong .env.",
                ) from exc

            # Ép thiết bị tường minh thay vì để Vieneu() tự dò. Mặc định `device="auto"`
            # của thư viện hỏi lại `torch.cuda.is_available()` (vieneu/v3turbo.py:46-54) —
            # đúng cái câu hỏi dối trá mà uses_gpu() đã thay bằng phép thử thật. Không ép
            # thì hai bên có thể trả lời khác nhau: model_store tải bộ trọng số ONNX/CPU
            # còn engine lại nạp nhánh PyTorch/GPU rồi nổ. Một nguồn sự thật, một lựa chọn.
            device = "cuda" if _uses_gpu() else "cpu"
            log.info("Đang nạp VieNeu-TTS trên %s (lần đầu sẽ tải model ~609 MB)", device.upper())
            _engine = Vieneu(device=device)
        return _engine


def _to_pcm16(audio: np.ndarray, source_rate: int, target_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """float32 trong [-1, 1] ở 48 kHz → PCM 16-bit mono ở target_rate."""
    if audio.size == 0:
        return b""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if source_rate != target_rate:
        # Nội suy tuyến tính là đủ: 48k→24k chỉ là chia đôi, không sinh méo đáng kể.
        count = int(round(len(audio) * target_rate / source_rate))
        positions = np.linspace(0, len(audio) - 1, count, dtype=np.float64)
        audio = np.interp(positions, np.arange(len(audio)), audio)

    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _style_of(engine, voice: str) -> str:
    try:
        return engine.get_preset_voice(voice).get("style") or "tu_nhien"
    except Exception:
        return "tu_nhien"


def _register_clone(engine, voice: str) -> None:
    """Học giọng từ audio mẫu — một lần cho cả tiến trình, tốn vài giây."""
    from . import custom_voices

    if not custom_voices.is_custom(voice) or voice in _registered_clones:
        return
    engine.add_voice(voice, custom_voices.sample_path(voice), save=False)
    _registered_clones.add(voice)


def _loi(voice: str, exc: Exception) -> SpeechServiceError:
    return SpeechServiceError(
        f"VieNeu-TTS không đọc được lượt thoại (giọng {voice}): {exc}",
        user_message="VieNeu-TTS gặp lỗi khi tạo giọng đọc. "
                     "Thử đổi TTS_PROVIDER=edge trong .env.",
    )


class VieNeuSynthesizer:
    """Bám giao thức `synthesize` của pipeline: trả PCM 16-bit mono TTS_SAMPLE_RATE Hz.

    Trên GPU còn phơi thêm `synthesize_batch` — pipeline/tts.py sẽ ưu tiên dùng nó.
    """

    def __init__(self, *, watermark: bool = False):
        # Mặc định VieNeu nhúng dấu chìm vào audio (thư viện `perth`). Tắt có chủ ý:
        # đây là video của người dùng, không phải sản phẩm của model.
        self._watermark = watermark

    @property
    def batch_size(self) -> int:
        """0 = không gộp lô được (engine ONNX/CPU) — người gọi sẽ quay về từng câu một."""
        return batch_size()

    def synthesize(self, text: str, voice_id: str) -> bytes:
        from .voices import native_id

        voice = native_id(voice_id, "vieneu")
        engine = _get_engine()

        try:
            with _infer_lock:
                _register_clone(engine, voice)
                audio = engine.infer(
                    text,
                    voice=voice,
                    # Mỗi giọng dựng sẵn có phong cách riêng (tự nhiên / tin tức / kể chuyện);
                    # ép một phong cách chung sẽ xóa mất đặc trưng của giọng.
                    style=_style_of(engine, voice),
                    apply_watermark=self._watermark,
                )
                rate = getattr(engine, "sample_rate", 48_000)
        except SpeechServiceError:
            raise
        except Exception as exc:
            raise _loi(voice, exc) from exc

        pcm = _to_pcm16(np.asarray(audio, dtype=np.float32), rate)
        if not pcm:
            raise SpeechServiceError(f"VieNeu-TTS trả về 0 byte âm thanh cho giọng {voice}")
        return pcm

    def synthesize_batch(self, texts: list[str], voice_id: str) -> list[bytes]:
        """Đọc NHIỀU lượt thoại trong một lượt gọi GPU. Chỉ có trên engine PyTorch."""
        from .vieneu_batch import synthesize_batch as _lo
        from .voices import native_id

        voice = native_id(voice_id, "vieneu")
        engine = _get_engine()

        try:
            with _infer_lock:
                _register_clone(engine, voice)
                wavs = _lo(
                    engine, texts, voice, _style_of(engine, voice),
                    batch_size=len(texts),          # người gọi đã cắt lô sẵn
                    apply_watermark=self._watermark,
                )
                rate = getattr(engine, "sample_rate", 48_000)
        except SpeechServiceError:
            raise
        except Exception as exc:
            raise _loi(voice, exc) from exc

        return [_to_pcm16(np.asarray(w, dtype=np.float32), rate) for w in wavs]
