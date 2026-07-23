"""Quản lý các model tải từ Hugging Face về máy, có đo tiến trình.

Đo bằng dung lượng thư mục `blobs/` — đây là nơi duy nhất chứa byte thật (thư mục
`snapshots/` chỉ toàn symlink nên đếm vào sẽ nhân đôi). Cách này không phụ thuộc vào
thư viện bên dưới có chịu phơi ra thanh tiến trình hay không: `faster_whisper` chặn
cứng thanh tiến trình bằng `tqdm_class=disabled_tqdm`.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def uses_gpu() -> bool:
    """VieNeu tự chọn engine PyTorch khi thấy CUDA, ONNX khi không (xem vieneu/v3turbo.py).

    Hai engine đọc HAI bộ trọng số khác nhau, nên câu hỏi "model đã tải chưa" chỉ trả lời
    được sau khi biết engine nào sẽ chạy. Trước đây hàm này không tồn tại và spec luôn khai
    bộ ONNX: máy có GPU bị báo "chưa tải" vĩnh viễn, còn nút tải thì kéo về bộ CPU vô dụng.

    KHÔNG hỏi `torch.cuda.is_available()` rồi tin ngay: hàm đó chỉ nói "có card + có driver",
    KHÔNG nói "bản torch này có kernel chạy được trên card đó". Một bản torch build cho
    CUDA 12.6 (kernel sm_50…sm_90) gặp RTX 5090 (sm_120) vẫn trả True, rồi phép tính CUDA
    ĐẦU TIÊN mới nổ `no kernel image is available` — tức là nổ giữa bước giọng đọc, sau khi
    người dùng đã chờ nhận diện và dịch xong. Cùng một venv chạy ngon trên RTX 3090 (sm_86)
    và chết trên RTX 5090 là vì vậy.

    Nên ta hỏi bằng cách LÀM: chạy thử một phép nhân ma trận bé xíu. Chạy được thì GPU dùng
    được thật; ném lỗi thì lặng lẽ lùi về CPU/ONNX — chậm hơn, nhưng chạy xong.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        probe = torch.zeros(8, 8, device="cuda")
        float((probe @ probe).sum().cpu())
        return True
    except Exception as exc:
        log.warning("Có card NVIDIA nhưng bản torch này không chạy được trên nó (%s). "
                    "Lùi về CPU. Cài lại torch khớp card để nhanh hơn nhiều.", exc)
        return False


@dataclass(frozen=True)
class RepoSpec:
    """Một repo Hugging Face, kèm bộ file cần lấy. `patterns=None` nghĩa là lấy hết."""

    repo_id: str
    patterns: list[str] | None = None


@dataclass(frozen=True)
class ModelSpec:
    key: str            # định danh dùng trong API: "whisper" | "vieneu"
    label: str          # hiển thị cho người dùng
    repos: list[RepoSpec] = field(default_factory=list)


# Cả repo VieNeu là 1512 MB nhưng mỗi engine chỉ dùng một nhánh trọng số, nên ta lọc.
# `denoiser.onnx` + `speaker_encoder.onnx` (28 MB) phục vụ nhân bản giọng và CẢ HAI engine
# đều nạp — thiếu chúng thì lần clone đầu tiên sẽ tự tải giữa chừng thay vì nằm trong thanh
# tiến trình "Tải model".
_VIENEU_CHUNG = ["config.json", "denoiser.onnx", "speaker_encoder.onnx"]
# Engine ONNX (CPU) đọc onnx_update/; engine PyTorch (GPU) đọc update/ — xem v3turbo.py.
_VIENEU_ONNX = _VIENEU_CHUNG + ["onnx_update/*"]
_VIENEU_TORCH = _VIENEU_CHUNG + ["update/*"]

# Đúng bộ file mà faster-whisper tải; giữ khớp để đo và tải cùng một tập.
_WHISPER_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]

_total_bytes_cache: dict[str, int] = {}


@dataclass(frozen=True)
class ModelInfo:
    key: str
    label: str
    ready: bool
    downloaded_bytes: int
    total_bytes: int

    @property
    def percent(self) -> float:
        if self.ready:
            return 100.0
        if self.total_bytes <= 0:
            return 0.0
        return round(min(100.0, self.downloaded_bytes / self.total_bytes * 100), 1)


# ─── định nghĩa model ───────────────────────────────────────────────

def whisper_spec(name: str) -> ModelSpec:
    from faster_whisper.utils import _MODELS

    repo = name if "/" in name else _MODELS.get(name)
    if repo is None:
        raise ValueError(f"Model Whisper không hợp lệ: {name}. Chọn một trong: {', '.join(_MODELS)}")
    return ModelSpec(
        key="whisper",
        label=f"Nhận diện giọng nói — Whisper '{name}'",
        repos=[RepoSpec(repo, _WHISPER_PATTERNS)],
    )


def vieneu_spec() -> ModelSpec:
    """Bộ trọng số phải khớp engine mà VieNeu sẽ tự chọn, nếu không thì đo và tải đều sai chỗ."""
    if uses_gpu():
        return ModelSpec(
            key="vieneu",
            label="Giọng đọc — VieNeu-TTS v3 Turbo (GPU)",
            repos=[
                RepoSpec("pnnbao-ump/VieNeu-TTS-v3-Turbo", _VIENEU_TORCH),
                # Engine PyTorch nạp tokenizer bản safetensors, không phải bản -ONNX.
                RepoSpec("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano", None),
            ],
        )
    return ModelSpec(
        key="vieneu",
        label="Giọng đọc — VieNeu-TTS v3 Turbo (CPU)",
        repos=[
            RepoSpec("pnnbao-ump/VieNeu-TTS-v3-Turbo", _VIENEU_ONNX),
            RepoSpec("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX", None),
        ],
    )


def omnivoice_spec() -> ModelSpec:
    """OmniVoice tự chứa trong một repo (~3,3 GB). Chỉ chạy trên GPU thực tế."""
    return ModelSpec(
        key="omnivoice",
        label="Giọng nhân bản — OmniVoice (GPU)",
        repos=[RepoSpec("k2-fsa/OmniVoice", None)],
    )


# ─── đo và tải ──────────────────────────────────────────────────────

def _cache_dir(repo_id: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _downloaded_bytes(repo_id: str) -> int:
    """Byte thật trên đĩa, kể cả phần đang tải dở (`*.incomplete`).

    Đếm mọi file THẬT (bỏ symlink) trong cả thư mục repo, không riêng `blobs/`. Lý do:
    huggingface_hub chỉ đặt byte vào `blobs/` rồi trỏ symlink từ `snapshots/` khi máy CÓ
    symlink. Windows không cho tạo symlink nếu thiếu Developer Mode/quyền admin, lúc đó nó
    CHUYỂN hẳn file sang `snapshots/` và bỏ `blobs/` gần như rỗng — chỉ đếm `blobs/` thì
    thanh tiến trình đứng im ở 0% suốt 400 MB rồi nhảy phịch sang "xong".

    Bỏ symlink là đủ để không đếm hai lần: chỗ nào có symlink thì byte nằm ở `blobs/`,
    chỗ nào không thì byte nằm ở `snapshots/`, không bao giờ cả hai.
    """
    repo_dir = _cache_dir(repo_id)
    if not repo_dir.is_dir():
        return 0
    return sum(f.stat().st_size for f in repo_dir.rglob("*")
               if f.is_file() and not f.is_symlink())


def _total_bytes(repo: RepoSpec) -> int:
    """Hỏi Hugging Face tổng dung lượng cần tải. Cần mạng; trả 0 nếu hỏi không được."""
    if repo.repo_id in _total_bytes_cache:
        return _total_bytes_cache[repo.repo_id]
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo.repo_id, files_metadata=True)
        files = info.siblings
        if repo.patterns is not None:
            files = [s for s in files
                     if any(fnmatch.fnmatch(s.rfilename, p) for p in repo.patterns)]
        total = sum(s.size or 0 for s in files)
    except Exception as exc:
        log.warning("Không hỏi được dung lượng %s: %s", repo.repo_id, exc)
        return 0

    _total_bytes_cache[repo.repo_id] = total
    return total


def _repo_ready(repo: RepoSpec) -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo.repo_id, allow_patterns=repo.patterns, local_files_only=True)
        return True
    except Exception:
        return False


def is_ready(spec: ModelSpec) -> bool:
    """Model đã đủ file trên đĩa chưa — kiểm tra offline, không đụng mạng."""
    return all(_repo_ready(r) for r in spec.repos)


def describe(spec: ModelSpec) -> ModelInfo:
    ready = is_ready(spec)
    downloaded = sum(_downloaded_bytes(r.repo_id) for r in spec.repos)
    total = downloaded if ready else sum(_total_bytes(r) for r in spec.repos)
    return ModelInfo(key=spec.key, label=spec.label, ready=ready,
                     downloaded_bytes=downloaded, total_bytes=total)


def download(spec: ModelSpec) -> None:
    """Tải mọi repo của model. Chạy đồng bộ — người gọi tự đặt nó vào luồng nền."""
    from huggingface_hub import snapshot_download

    for repo in spec.repos:
        log.info("Đang tải %s", repo.repo_id)
        snapshot_download(repo.repo_id, allow_patterns=repo.patterns)
