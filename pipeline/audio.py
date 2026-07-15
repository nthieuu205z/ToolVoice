"""Tiện ích âm thanh: đọc/ghi WAV, đổi tốc độ giữ nguyên cao độ, trộn nhiều đoạn.

Toàn bộ pipeline làm việc với PCM 16-bit mono nên numpy + module `wave` chuẩn là đủ.
"""

from __future__ import annotations

import re
import tempfile
import wave
from pathlib import Path

import numpy as np

from .errors import FFmpegError
from .ffmpeg_utils import resolve, run, run_piped
from .models import TTS_SAMPLE_RATE

_RATE_IN_MIME = re.compile(r"rate=(\d+)")


def decode_to_pcm(data: bytes, rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Bất kỳ định dạng nén nào (mp3, ogg…) → PCM 16-bit mono ở tần số mong muốn."""
    if not data:
        return b""
    return run_piped([
        resolve("ffmpeg"), "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ac", "1", "-ar", str(rate),
        "pipe:1",
    ], data)


def parse_pcm_rate(mime_type: str | None) -> int | None:
    """Đọc tần số lấy mẫu từ mime kiểu `audio/L16;codec=pcm;rate=24000`.

    Đoán sai tần số là hỏng ngầm: giọng đọc sẽ sai cao độ và lệch hết mốc thời gian.
    """
    if not mime_type:
        return None
    match = _RATE_IN_MIME.search(mime_type)
    return int(match.group(1)) if match else None


def resample_pcm(pcm: bytes, from_rate: int, to_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """Đổi tần số lấy mẫu của PCM 16-bit mono."""
    if from_rate == to_rate or not pcm:
        return pcm
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        write_wav(src, pcm_to_array(pcm), from_rate)
        run([
            resolve("ffmpeg"), "-y", "-loglevel", "error",
            "-i", str(src), "-ar", str(to_rate), "-ac", "1",
            "-c:a", "pcm_s16le", str(dst),
        ], timeout=300)
        samples, _ = read_wav(dst)
        return samples.tobytes()


def pcm_to_array(pcm: bytes) -> np.ndarray:
    """PCM 16-bit little-endian → mảng int16. Bỏ byte lẻ cuối nếu có."""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    return np.frombuffer(pcm, dtype="<i2")


def write_wav(path: Path, samples: np.ndarray, rate: int = TTS_SAMPLE_RATE) -> Path:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.astype("<i2").tobytes())
    return path


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise FFmpegError(f"WAV không phải mono 16-bit: {path}")
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2"), rate


def duration_of(samples: np.ndarray, rate: int = TTS_SAMPLE_RATE) -> float:
    return len(samples) / float(rate)


def _atempo_chain(speed: float) -> str:
    """Mỗi bộ lọc atempo chỉ nhận 0.5–2.0, nên tốc độ ngoài khoảng đó phải nối chuỗi."""
    factors: list[float] = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={f:.6f}" for f in factors)


def time_stretch(samples: np.ndarray, speed: float, rate: int = TTS_SAMPLE_RATE) -> np.ndarray:
    """Đổi tốc độ phát mà giữ nguyên cao độ. speed > 1 = nhanh hơn (ngắn lại)."""
    if abs(speed - 1.0) < 0.01 or len(samples) == 0:
        return samples

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / "out.wav"
        write_wav(src, samples, rate)
        run([
            resolve("ffmpeg"), "-y", "-loglevel", "error",
            "-i", str(src),
            "-filter:a", _atempo_chain(speed),
            "-c:a", "pcm_s16le",
            str(dst),
        ], timeout=300)
        stretched, _ = read_wav(dst)
    # `np.frombuffer` trả về mảng chỉ đọc trỏ vào buffer tạm — copy trước khi tmpdir biến mất.
    return stretched.copy()


def fit_to_slot(samples: np.ndarray, target_duration: float, max_speedup: float,
                rate: int = TTS_SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Ép một đoạn lời thoại vào khung thời gian gốc.

    Chỉ tăng tốc khi âm thanh dài hơn khung — không bao giờ kéo chậm lại để lấp
    khoảng trống, vì giọng bị kéo chậm nghe giả hơn hẳn so với việc để im lặng.
    Trả về (mẫu đã chỉnh, số giây còn tràn ra ngoài khung).
    """
    if target_duration <= 0 or len(samples) == 0:
        return samples, 0.0

    actual = duration_of(samples, rate)
    if actual <= target_duration:
        return samples, 0.0

    needed = actual / target_duration
    speed = min(needed, max_speedup)
    fitted = time_stretch(samples, speed, rate)
    overflow = max(0.0, duration_of(fitted, rate) - target_duration)
    return fitted, overflow


# Tăng tốc tới mức này tai người gần như không nhận ra — dùng để đưa câu về sát
# khung gốc trước khi phải đi mượn khoảng lặng (đo: 132/156 lượt bị dồn toa khi
# không có nấc này, vì tiếng Việt thường dài hơn tiếng Anh một chút).
COMFORT_SPEEDUP = 1.15


def fit_to_window(samples: np.ndarray, slot_duration: float, window_duration: float,
                  max_speedup: float, comfort_speedup: float = COMFORT_SPEEDUP,
                  rate: int = TTS_SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Ép lời thoại theo ba nấc, ưu tiên bám mốc câu gốc mà không làm méo giọng:

    1. Vừa khung gốc (`slot_duration`) → giữ nguyên.
    2. Dài hơn khung → tăng tốc NHẸ (≤ `comfort_speedup`, dưới ngưỡng tai để ý)
       để về sát khung; phần còn dư tràn tự nhiên vào khoảng lặng (`window_duration`
       tính tới sát mốc câu kế tiếp).
    3. Hết cả chỗ mượn → tăng tốc mạnh tới `max_speedup`. Vẫn dư thì trả về số giây
       tràn — bước xếp chỗ sẽ lùi câu sau lại, không cắt chữ.
    """
    if slot_duration <= 0 or len(samples) == 0:
        return samples, 0.0

    window = max(slot_duration, window_duration)
    actual = duration_of(samples, rate)
    if actual <= slot_duration:
        return samples, 0.0

    speed = min(actual / slot_duration, comfort_speedup)
    if actual / speed > window:
        speed = min(actual / window, max_speedup)
    fitted = time_stretch(samples, speed, rate) if speed > 1.01 else samples
    return fitted, max(0.0, duration_of(fitted, rate) - window)


def truncate_with_fade(samples: np.ndarray, max_duration: float,
                       fade: float = 0.08, rate: int = TTS_SAMPLE_RATE) -> np.ndarray:
    """Cắt audio về `max_duration`, làm mờ dần đoạn cuối để không nghe 'cạch'."""
    limit = int(max_duration * rate)
    if len(samples) <= limit:
        return samples
    out = samples[:limit].astype(np.float64)
    steps = min(len(out), int(fade * rate))
    if steps:
        out[-steps:] *= np.linspace(1.0, 0.0, steps)
    return out.astype(samples.dtype)


def silent_track(duration: float, rate: int = TTS_SAMPLE_RATE) -> np.ndarray:
    return np.zeros(max(0, int(round(duration * rate))), dtype=np.int32)


def mix_into(track: np.ndarray, samples: np.ndarray, start: float,
             rate: int = TTS_SAMPLE_RATE) -> None:
    """Trộn `samples` vào `track` (int32) tại mốc `start` giây, cộng dồn tại chỗ.

    Cộng dồn thay vì ghi đè để lời thoại bị tràn khung không cắt cụt lời kế tiếp.
    """
    if len(samples) == 0:
        return
    begin = max(0, int(round(start * rate)))
    if begin >= len(track):
        return
    end = min(len(track), begin + len(samples))
    track[begin:end] += samples[: end - begin].astype(np.int32)


def finalize_track(track: np.ndarray) -> np.ndarray:
    """int32 đã trộn → int16, chặn tràn biên bằng cách kẹp giá trị."""
    return np.clip(track, -32768, 32767).astype("<i2")
