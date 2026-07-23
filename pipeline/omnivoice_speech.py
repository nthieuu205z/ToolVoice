"""Giọng đọc bằng OmniVoice — TTS nhân bản zero-shot đa ngôn ngữ (k2-fsa/OmniVoice).

Khác VieNeu ở điểm cốt lõi: OmniVoice KHÔNG có giọng dựng sẵn. Mỗi lượt đọc phải kèm
một clip tham chiếu (ref_audio) và LỜI của clip đó (ref_text). Ta tái dùng chính các
giọng nhân bản người dùng đã lưu trong custom_voices/ làm ref_audio; ref_text thì để
Whisper chép MỘT lần rồi ghi ra file cạnh clip (sidecar `<id>.reftext.txt`) để lần sau
khỏi chép lại — kể cả sau khi restart.

Model ~3,3 GB tải tự động từ Hugging Face ở lần chạy đầu (from_pretrained). Trọng số
CC-BY-NC (phi thương mại) — khác VieNeu Apache-2.0; dùng cá nhân thì thoải mái.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from . import custom_voices
from .errors import SpeechServiceError
from .models import TTS_SAMPLE_RATE

log = logging.getLogger(__name__)

# OmniVoice sinh audio ở đúng 24 kHz = TTS_SAMPLE_RATE, nên chỉ lượng tử hóa, không nội suy.
assert TTS_SAMPLE_RATE == 24_000  # đổi hằng số này thì phải thêm resample bên dưới

# Nạp model tốn vài giây + vài trăm MB — dùng chung cho cả tiến trình.
_model = None
_model_lock = threading.Lock()
# Model không hứa an toàn khi gọi đồng thời; mỗi job tự tạo synthesizer riêng, nên khóa
# suy luận phải TOÀN CỤC (cùng lý do như vieneu_speech).
_infer_lock = threading.Lock()


def _uses_gpu() -> bool:
    from .model_store import uses_gpu

    return uses_gpu()


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            try:
                from omnivoice import OmniVoice
            except ImportError as exc:
                raise SpeechServiceError(
                    "Chưa cài OmniVoice",
                    user_message="Chưa cài OmniVoice. Chạy: pip install omnivoice, "
                                 "hoặc đổi TTS_PROVIDER=vieneu trong .env.",
                ) from exc
            import torch

            gpu = _uses_gpu()
            device = "cuda:0" if gpu else "cpu"
            dtype = torch.float16 if gpu else torch.float32
            log.info("Đang nạp OmniVoice trên %s (lần đầu sẽ tải model ~3,3 GB)", device)
            _model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        return _model


def prewarm() -> None:
    """Nạp model OmniVoice ở luồng nền lúc boot — job clone đầu khỏi chờ ~5 phút nạp.

    Chỉ nạp khi model đã nằm sẵn trên đĩa: việc tải ~3,3 GB là do người dùng bấm nút trong
    giao diện, server không tự ý tải lúc boot (giống vieneu_speech.prewarm).
    """
    from .model_store import is_ready, omnivoice_spec

    try:
        if not is_ready(omnivoice_spec()):
            log.info("OmniVoice chưa tải về — bỏ qua nạp sẵn, chờ người dùng bấm tải")
            return
    except Exception as exc:
        log.warning("Không kiểm tra được model OmniVoice: %s", exc)
        return

    def load() -> None:
        try:
            _get_model()
            log.info("OmniVoice sẵn sàng (nạp sẵn lúc boot)")
        except Exception as exc:
            log.warning("Không nạp sẵn được OmniVoice (job đầu sẽ tự nạp): %s", exc)

    threading.Thread(target=load, name="omnivoice-prewarm", daemon=True).start()


def forget_clone(voice_id: str) -> None:
    """Xóa ref_text đã nhớ (sidecar) khi người dùng xóa giọng — để id không trỏ lời cũ.

    OmniVoice không giữ giọng trong RAM như VieNeu; "quên" chỉ là bỏ file ref_text đã chép.
    """
    from . import custom_voices

    try:
        sidecar = custom_voices.sample_path(voice_id).with_name(f"{voice_id}.reftext.txt")
        sidecar.unlink(missing_ok=True)
    except OSError:
        pass


def _to_pcm16(audio) -> bytes:
    """float32 trong [-1, 1] @24kHz → PCM 16-bit mono @TTS_SAMPLE_RATE (cùng tần số)."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return b""
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


# Gộp lô GPU. Đo thật trên 12 GB (num_step=32): per-item 5,81s (N=1) → 1,56s (N=4) → 1,60s
# (N=8) → 1,73s (N=16). Khác VieNeu (thích lô TO vì nghẽn phóng kernel), OmniVoice nghẽn
# SỨC TÍNH: per-item chạm đáy ở N≈4–8 rồi hơi xấu đi. Nên "knee" ở 8, KHÔNG cố ép lô to.
# VRAM rẻ (4,8 GB ở N=16) nên hầu như mọi card đều chạy được knee; chỉ card tí hon mới bị VRAM cạp.
_OMNI_BATCH_KNEE = 8
_OMNI_BATCH_MIN = 1
_OMNI_VRAM_PER_ITEM_GB = 0.18   # đo: ~2,6 GB thêm cho 15 item ⇒ ~0,17/item, làm tròn lên


def _auto_batch_size(override: int) -> int:
    """Số câu gộp một lô. 0 = không gộp (CPU). Tự suy từ VRAM nhưng cạp ở knee đo được."""
    if not _uses_gpu():
        return 0
    if override > 0:
        return override
    try:
        import torch

        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    except Exception:
        return _OMNI_BATCH_KNEE
    n = int(free_gb * 0.75 / _OMNI_VRAM_PER_ITEM_GB)
    return max(_OMNI_BATCH_MIN, min(n, _OMNI_BATCH_KNEE))


class OmniVoiceSynthesizer:
    """Bám giao thức `synthesize(text, voice_id) -> bytes` (PCM 16-bit mono TTS_SAMPLE_RATE).

    Trên GPU còn phơi `synthesize_batch` + `batch_size` — pipeline/tts.py sẽ ưu tiên gộp lô.
    Chỉ đọc được giọng NHÂN BẢN (id `clone-…`) — OmniVoice cần clip tham chiếu, không có
    giọng dựng sẵn như VieNeu.
    """

    def __init__(self, *, model=None, transcriber=None,
                 whisper_model: str = "small", whisper_compute_type: str = "int8",
                 num_step: int = 32, batch_size: int = 0):
        self._model = model                 # tiêm vào cho test; None = nạp model thật khi cần
        self._transcriber = transcriber     # callable(Path)->str; None = dùng Whisper
        self._whisper_model = whisper_model
        self._whisper_compute_type = whisper_compute_type
        self._num_step = num_step
        self._batch_override = batch_size
        self._default_tr = None
        self._ref_cache: dict[str, str] = {}

    @property
    def batch_size(self) -> int:
        return _auto_batch_size(self._batch_override)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        return self._generate([text], voice_id)[0]

    def synthesize_batch(self, texts: list[str], voice_id: str) -> list[bytes]:
        """Đọc NHIỀU lượt thoại trong MỘT lượt gọi GPU (generate nhận list)."""
        return self._generate(list(texts), voice_id)

    def _generate(self, texts: list[str], voice_id: str) -> list[bytes]:
        if not custom_voices.is_custom(voice_id):
            raise SpeechServiceError(
                f"OmniVoice chỉ đọc bằng giọng nhân bản, không có giọng dựng sẵn: {voice_id}",
                user_message="OmniVoice chỉ dùng được giọng nhân bản. Chọn một giọng nhân "
                             "bản, hoặc đổi TTS_PROVIDER=vieneu trong .env.",
            )
        sample = custom_voices.sample_path(voice_id)
        if not sample.is_file():
            raise SpeechServiceError(
                f"OmniVoice thiếu clip mẫu cho giọng {voice_id}: {sample}",
                user_message="Giọng nhân bản này thiếu file mẫu. Tạo lại giọng rồi thử lại.",
            )

        ref_text = self._ref_text(voice_id, sample)
        model = self._model or _get_model()
        n = len(texts)

        try:
            with _infer_lock:
                out = model.generate(
                    text=list(texts),
                    ref_audio=[str(sample)] * n,
                    ref_text=[ref_text] * n,
                    # Luôn dub sang Việt → khai báo ngôn ngữ cho nhích chất lượng/tốc độ.
                    language=["Vietnamese"] * n,
                    num_step=self._num_step,
                    # Tự cắt khoảng lặng dài ở đầu ra — thay cho lớp "đọc lại lỗ hổng" của VieNeu.
                    postprocess_output=True,
                )
        except SpeechServiceError:
            raise
        except Exception as exc:
            raise SpeechServiceError(
                f"OmniVoice không đọc được lượt thoại (giọng {voice_id}): {exc}",
                user_message="OmniVoice gặp lỗi khi tạo giọng đọc. "
                             "Thử đổi TTS_PROVIDER=vieneu trong .env.",
            ) from exc

        arrays = list(out) if isinstance(out, (list, tuple)) else [out]
        pcms = [_to_pcm16(a) for a in arrays]
        if len(pcms) != n or any(not p for p in pcms):
            raise SpeechServiceError(
                f"OmniVoice trả về audio rỗng/thiếu ({len(pcms)}/{n}) cho giọng {voice_id}")
        return pcms

    # ── ref_text: chép lời clip mẫu một lần, nhớ ra đĩa ──────────────────────
    def _ref_text(self, voice_id: str, sample) -> str:
        if voice_id in self._ref_cache:
            return self._ref_cache[voice_id]

        sidecar = sample.with_name(f"{voice_id}.reftext.txt")
        text = ""
        if sidecar.is_file():
            text = sidecar.read_text(encoding="utf-8").strip()
        if not text:
            text = self._transcribe(sample).strip()
            if text:
                try:
                    sidecar.write_text(text, encoding="utf-8")
                except OSError as exc:
                    log.warning("Không ghi được ref_text cạnh %s: %s", sample, exc)

        self._ref_cache[voice_id] = text
        return text

    def _transcribe(self, sample) -> str:
        if self._transcriber is not None:
            return self._transcriber(sample)
        if self._default_tr is None:
            from .whisper_stt import WhisperTranscriber

            self._default_tr = WhisperTranscriber(self._whisper_model, self._whisper_compute_type)
        _lang, text = self._default_tr.transcribe_clip(sample)
        return text
