"""Nhận diện giọng nói bằng Whisper chạy ngay trên máy — miễn phí, không hạn mức.

Vì sao không dùng mốc thời gian của Whisper: ffmpeg đã cho ranh giới chính xác tuyệt đối
(xem pipeline/segmentation.py). Whisper chỉ được giao đúng một việc là chép chữ cho từng
vùng đã cắt sẵn, nên không cần tin vào mốc thời gian của nó.

Chạy trên GPU khi máy có card NVIDIA. Đo trên video 18,9 phút, model `small`:

    CPU (int8)   : 268 giây
    GPU (float16):  81 giây     ← nhanh gấp 3,3 lần

Trước đây file này ghim cứng device="cpu", nên Whisper chậm tới mức không dùng nổi và
người dùng buộc phải trả tiền cho Gemini. Trên máy không có CUDA thì vẫn tự lùi về CPU.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

# Nạp model tốn vài giây và ngốn RAM — dùng chung cho mọi job trong tiến trình.
_models: dict[tuple[str, str], object] = {}
_load_lock = threading.Lock()
# Nhiều job chạy song song vẫn trỏ vào CÙNG một model object; faster-whisper không
# hứa an toàn khi gọi đồng thời, nên khóa suy luận phải là toàn cục chứ không phải
# per-instance (mỗi job tạo một WhisperTranscriber riêng, khóa riêng thì vô nghĩa).
_infer_lock = threading.Lock()


# Bật lên khi GPU đã CHỨNG MINH là không dùng được. Từ đó mọi lượt đi thẳng lên CPU thay vì
# thử lại rồi ngã lại cho từng đoạn — hỏng 150 lần trên cùng một lý do thì chậm hơn hẳn CPU.
_gpu_bi_loai = False


def _thiet_bi() -> tuple[str, str]:
    """(device, compute_type) hợp với máy này.

    CTranslate2 (lõi của faster-whisper) cần cuDNN 9 + cuBLAS 12. Trên Windows, các DLL đó
    nằm sẵn trong gói torch bản CUDA — nhưng chỉ tìm thấy được SAU KHI import torch, vì
    chính torch gọi os.add_dll_directory cho thư mục lib của nó. Nên phải import torch
    trước, dù file này không dùng tensor nào.

    Ràng buộc ngầm, và nó ĐẮT: `cuBLAS 12` nghĩa là torch phải là bản CUDA **12.x**, vì bản
    CUDA 13 mang `cublas64_13.dll` — tên khác, CTranslate2 không thấy và ngã ngay lượt chép
    lời đầu tiên. Card Blackwell (RTX 50xx, sm_120) lại đòi CUDA ≥ 12.8. Giao của hai điều
    kiện chỉ còn CUDA 12.8/12.9 — đó là lý do dự án ghim `cu129` chứ không lấy bản mới nhất.
    """
    if _gpu_bi_loai:
        return "cpu", "int8"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception as exc:
        log.debug("Không dò được CUDA cho Whisper: %s", exc)
    return "cpu", "int8"


def _loai_gpu(exc: Exception) -> None:
    """Rớt hẳn về CPU sau khi GPU tỏ ra không dùng được — không hỏi lại nữa."""
    global _gpu_bi_loai
    if _gpu_bi_loai:
        return
    _gpu_bi_loai = True
    _get_pipeline.cache_clear()
    log.warning(
        "Whisper không chạy được trên GPU (%s) — chuyển hẳn sang CPU tới khi khởi động lại. "
        "Thường là torch sai bản CUDA: CTranslate2 cần cuBLAS 12 (torch cu126/cu129), "
        "bản cu130 trở lên mang cuBLAS 13 nên không dùng được.", exc)


def _get_model(name: str, compute_type: str):
    device, dtype_gpu = _thiet_bi()
    # compute_type từ .env chỉ có nghĩa trên CPU; GPU luôn dùng float16 (int8 trên CUDA
    # vừa chậm hơn vừa kém chính xác hơn với CTranslate2).
    ct = dtype_gpu if device == "cuda" else compute_type

    key = (name, ct)
    with _load_lock:
        if key not in _models:
            from faster_whisper import WhisperModel  # nạp muộn: import nặng

            log.info("Đang nạp Whisper '%s' trên %s (lần đầu sẽ tải model về)", name, device.upper())
            _models[key] = WhisperModel(name, device=device, compute_type=ct)
        return _models[key]


def batch_size() -> int:
    """Số vùng chép lời trong MỘT lượt gọi GPU. 0 = chép từng vùng một (CPU).

    Cùng câu chuyện với VieNeu: Whisper cũng bị nghẽn ở chi phí PHÓNG kernel chứ không phải
    sức tính. Đo trên video 18,9 phút (152 vùng), RTX 3090:

        từng vùng một      : 67 giây   (GPU 34%, 151W)
        gộp lô 32 vùng     :  7 giây   (GPU 65%, 254W)   ← nhanh gấp 9,4 lần

    Chạy đa luồng KHÔNG cứu được (đã đo: num_workers 1→8 chỉ nhanh gấp 1,2 lần rồi chững).

    Cỡ lô suy ra từ VRAM còn trống (tự thích nghi mọi card) thay vì ghim 32; card lớn chạy
    lô lớn hơn. Whisper 'small' gộp lô tốn ~0,2 GB/vùng; chừa 30%, trần 48 (knee thuật toán).
    """
    device, _ = _thiet_bi()
    if device != "cuda":
        return 0
    try:
        import torch

        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    except Exception:
        return 16
    n = int(free_gb * 0.7 / 0.2)
    return max(8, min(n, 48))


@lru_cache(maxsize=1)
def _get_pipeline(model):
    from faster_whisper import BatchedInferencePipeline

    return BatchedInferencePipeline(model=model)


def prewarm(model: str = "small", compute_type: str = "int8") -> None:
    """Nạp model Whisper trong luồng nền lúc server khởi động — job đầu khỏi chờ tải model.

    Đo thật: lần nhận diện đầu tốn ~24s, trong đó ~11s chỉ để nạp model + khởi tạo cuDNN;
    phần chép lời thật chỉ ~13s. Nạp sẵn từ lúc boot (song song với VieNeu) để job đầu chỉ
    còn phần tính. Chỉ nạp khi model đã nằm trên đĩa — không tự ý tải 484 MB lúc boot.
    """
    from .model_store import is_ready, whisper_spec

    try:
        if not is_ready(whisper_spec(model)):
            log.info("Whisper chưa tải về — bỏ qua nạp sẵn, chờ người dùng bấm tải")
            return
    except Exception as exc:
        log.warning("Không kiểm tra được model Whisper: %s", exc)
        return

    def load() -> None:
        try:
            _get_pipeline(_get_model(model, compute_type))   # nạp cả BatchedInferencePipeline
            log.info("Whisper sẵn sàng (%s)", _thiet_bi()[0].upper())
        except Exception as exc:
            log.warning("Không nạp sẵn được Whisper (job đầu sẽ tự nạp): %s", exc)

    threading.Thread(target=load, name="whisper-prewarm", daemon=True).start()


_SENTENCE_END = (".", "!", "?", "…")


def sentences_from_words(
    words: list[tuple[float, float, str]],
) -> list[tuple[float, float, str]]:
    """Gom các TỪ (start, end, text) của Whisper thành CÂU, cắt sau từ kết bằng .!?…

    Trả về [(start_câu, end_câu, văn_bản)]. Mốc câu = mốc từ đầu → mốc từ cuối của câu, là
    mốc căn chỉnh thật của Whisper. Câu cuối chưa có dấu kết vẫn được trả (mảnh dang dở) —
    merge_sentence_fragments sẽ ghép nó với mảnh mở của vùng kế. Nhờ tách theo câu, mỗi câu
    thành một lượt đọc VieNeu riêng (ổn định hơn), đặt đúng thời điểm câu tiếng Anh.
    """
    sentences: list[tuple[float, float, str]] = []
    parts: list[str] = []
    start: float | None = None
    end = 0.0
    for w_start, w_end, w_text in words:
        token = w_text.strip()
        if not token:
            continue
        if start is None:
            start = w_start
        parts.append(token)
        end = w_end
        if token.endswith(_SENTENCE_END):
            sentences.append((start, end, " ".join(parts)))
            parts, start = [], None
    if parts and start is not None:
        sentences.append((start, end, " ".join(parts)))
    return sentences


def _vung_chua(regions: list, boundaries: list[tuple[float, float]], start: float,
               end: float) -> int | None:
    """Vùng nào chồng lấn nhiều nhất với [start, end]. None nếu không chồng vào đâu.

    Dùng chồng lấn chứ không dùng điểm giữa: một mảnh trả về có thể lệch nhẹ khỏi mốc ta
    đưa vào, và gán nhầm vùng nghĩa là lời thoại nhảy sang khung thời gian khác — hỏng
    đồng bộ hình–tiếng theo cách rất khó thấy.
    """
    tot_nhat, tot_diem = None, 0.0
    for i, (s, e) in enumerate(boundaries):
        chong = min(end, e) - max(start, s)
        if chong > tot_diem:
            tot_nhat, tot_diem = i, chong
    return tot_nhat


class WhisperTranscriber:
    """Chép lời cho từng đoạn audio ngắn. Bám giao thức `transcribe_clip` của pipeline.

    Trên GPU còn phơi thêm `transcribe_batch` — pipeline/stt.py sẽ ưu tiên dùng nó.
    """

    def __init__(self, model: str = "small", compute_type: str = "int8"):
        self._model_name = model
        self._compute_type = compute_type

    @property
    def stt_batch_size(self) -> int:
        return batch_size()

    @property
    def stt_timed(self) -> bool:
        """Mốc thời gian cấp CÂU chỉ có ở đường gộp lô (GPU) qua BatchedInferencePipeline.
        Trên CPU (từng vùng một) ta không lấy được mốc con → để pipeline lùi về cấp vùng.
        """
        return batch_size() > 0

    def _chep(self, wav_path: Path) -> tuple[str, str]:
        model = _get_model(self._model_name, self._compute_type)
        with _infer_lock:
            # Giữ VAD dù ffmpeg đã cắt đúng vùng có tiếng: đo thật trên 50 clip, tắt VAD chỉ
            # nhanh hơn 2/55 giây và chép ra CHÍNH XÁC cùng số chữ — không đáng để bỏ đi lớp
            # chắn cuối cùng chống Whisper bịa lời trên đoạn im lặng.
            segments, info = model.transcribe(str(wav_path), vad_filter=True)
            text = " ".join(seg.text.strip() for seg in segments)
        return info.language or "", text.strip()

    def transcribe_clip(self, wav_path: Path) -> tuple[str, str]:
        # GPU có thể nạp model trót lọt rồi mới ngã ở phép tính đầu tiên (thiếu cuBLAS chẳng
        # hạn) — nên lưới an toàn phải ở đây, lúc CHẠY, chứ không phải lúc nạp.
        dung_gpu = _thiet_bi()[0] == "cuda"
        try:
            return self._chep(wav_path)
        except Exception as exc:
            if not dung_gpu or _gpu_bi_loai:
                raise
            _loai_gpu(exc)
            return self._chep(wav_path)   # lần này chắc chắn chạy trên CPU

    def transcribe_batch(self, samples, rate: int, regions: list) -> list[tuple[str, str]]:
        """Chép lời cho NHIỀU vùng trong một lượt gọi GPU. Trả về [(ngôn ngữ, lời)] đúng thứ tự.

        Mốc thời gian vẫn do ffmpeg quyết định — ta truyền thẳng chúng vào `clip_timestamps`
        nên Whisper không được phép tự cắt lại. Nguyên tắc của dự án giữ nguyên: ffmpeg lo
        THỜI GIAN, Whisper lo NỘI DUNG (xem pipeline/segmentation.py).
        """
        import numpy as np

        model = _get_model(self._model_name, self._compute_type)
        pipe = _get_pipeline(model)

        # faster-whisper nhận float32 mono 16 kHz; source.wav đã đúng thế (STT_SAMPLE_RATE).
        audio = np.asarray(samples, dtype="float32") / 32768.0

        moc = [{"start": r.start, "end": r.end} for r in regions]
        bien = [(r.start, r.end) for r in regions]

        try:
            with _infer_lock:
                segments, info = pipe.transcribe(audio, clip_timestamps=moc,
                                                 batch_size=batch_size())
                chu: list[list[str]] = [[] for _ in regions]
                for seg in segments:
                    i = _vung_chua(regions, bien, seg.start, seg.end)
                    if i is not None:
                        chu[i].append(seg.text.strip())
        except Exception as exc:
            # Không thử lại cả lô ở đây: sau khi rớt về CPU thì batch_size() = 0, đường gộp lô
            # không còn nghĩa gì. Chỉ hạ cờ rồi ném lên — pipeline/stt.py vốn đã có sẵn đường
            # lùi "lô hỏng → chép lại từng đoạn một", và các đoạn đó giờ sẽ chạy trên CPU.
            if _thiet_bi()[0] == "cuda":
                _loai_gpu(exc)
            raise

        lang = info.language or ""
        return [(lang, " ".join(c).strip()) for c in chu]

    def transcribe_batch_timed(
        self, samples, rate: int, regions: list,
    ) -> tuple[str, list[tuple[float, float, str]]]:
        """Như transcribe_batch nhưng GIỮ mốc thời gian cấp CÂU của Whisper.

        Với clip_timestamps, Whisper trả về MỘT segment cho mỗi vùng (không tự tách câu).
        Nên ta bật `word_timestamps=True` lấy mốc từng TỪ rồi tự tách câu theo dấu chấm câu
        (xem sentences_from_words) — cho mốc câu THẬT bên trong vùng. Mỗi câu → một Segment
        đặt đúng thời điểm, để VieNeu đọc từng câu (ổn định) và bám hình sát. Vùng ffmpeg
        vẫn lo THỜI GIAN THÔ (clip_timestamps ràng buộc Whisper chỉ chép chỗ có tiếng).
        """
        import numpy as np

        model = _get_model(self._model_name, self._compute_type)
        pipe = _get_pipeline(model)
        audio = np.asarray(samples, dtype="float32") / 32768.0

        moc = [{"start": r.start, "end": r.end} for r in regions]
        bien = [(r.start, r.end) for r in regions]

        try:
            with _infer_lock:
                segments, info = pipe.transcribe(audio, clip_timestamps=moc,
                                                 batch_size=batch_size(),
                                                 word_timestamps=True)
                pieces: list[tuple[float, float, str]] = []
                for seg in segments:
                    words = [(float(w.start), float(w.end), w.word)
                             for w in (getattr(seg, "words", None) or [])]
                    # Không có mốc từ (hiếm) → giữ cả segment làm một câu, đừng bỏ lời thoại.
                    cau = sentences_from_words(words) if words else (
                        [(float(seg.start), float(seg.end), seg.text.strip())]
                        if seg.text.strip() else [])
                    for s, e, t in cau:
                        if t.strip() and _vung_chua(regions, bien, s, e) is not None:
                            pieces.append((s, e, t.strip()))
        except Exception as exc:
            if _thiet_bi()[0] == "cuda":
                _loai_gpu(exc)
            raise

        return info.language or "", pieces
