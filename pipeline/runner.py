"""Điều phối toàn bộ pipeline. Cả CLI lẫn backend web đều gọi đúng hàm này."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .assembly import build_audio_track, plan_placement
from .audio import read_wav
from .errors import JobCancelledError
from .extract_audio import extract_audio
from .models import (
    CancelFn,
    GeminiBackend,
    MediaInfo,
    PipelineResult,
    ProgressFn,
    never_cancel,
    noop_progress,
)
from .mux import run_mux
from .probe import probe_video
from .segmentation import plan_utterances
from .stt import merge_sentence_fragments, transcribe_regions
from .subtitles import build_srt
from .translate import translate_segments
from .tts import synthesize_segments

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    voice_id: str
    max_utterance_seconds: float = 12.0
    max_utterance_gap: float = 0.5
    # Đặt câu theo mốc thời gian cấp câu của Whisper (bám hình sát hơn). Mặc định True;
    # CLI/test không có backend timed thì tự lùi về cấp vùng nên bật sẵn cũng an toàn.
    sentence_level_timing: bool = True
    stt_workers: int = 4
    tts_workers: int = 4
    translate_workers: int = 6
    tts_max_speedup: float = 1.5
    # Sàn kéo-chậm để lấp khung khi tiếng Việt xong sớm hơn hình. 1,0 = tắt (mặc định,
    # giữ hành vi cũ cho CLI/test); backend web truyền settings.tts_fill_slowdown (0,9).
    tts_fill_slowdown: float = 1.0
    # Bản miễn phí cho khoảng 100 lượt TTS mỗi ngày; cảnh báo sớm khi sắp chạm trần.
    # Chỉ có nghĩa với Gemini TTS — VieNeu/edge chạy không giới hạn nên tts_is_metered=False.
    tts_daily_budget: int = 90
    tts_is_metered: bool = False
    # Engine có KHUYẾT TẬT ngẫu nhiên chèn lỗ hổng im lặng giữa câu (VieNeu, edge) → bật đọc
    # lại lượt xấu. OmniVoice KHÔNG cần: nó tự vá bằng postprocess, và mỗi lần đọc lại là một
    # single-synth ~5,8s (không gộp lô) — rất đắt ở video dài. Đặt False cho OmniVoice.
    resynthesize_holes: bool = True


def run_pipeline(
    backend: GeminiBackend,
    video_path: Path,
    workdir: Path,
    options: PipelineOptions,
    progress: ProgressFn = noop_progress,
    media: MediaInfo | None = None,
    should_cancel: CancelFn = never_cancel,
) -> PipelineResult:
    """Chạy 7 bước từ video gốc tới video lồng tiếng + file .srt.

    `media` truyền vào khi backend web đã ffprobe lúc nhận upload (fail-fast trước khi
    tiêu tốn token); CLI để trống thì probe tại đây.

    Hủy là hợp tác: không giết thread giữa chừng (ffmpeg đang ghi file, ONNX đang chạy),
    mà kiểm tra cờ ở các mốc an toàn giữa hai bước và trong hai vòng lặp dài nhất.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    media = media or probe_video(video_path)
    warnings: list[str] = []

    def abort_if_cancelled() -> None:
        if should_cancel():
            raise JobCancelledError()

    # Đo thời gian TỪNG bước để biết nút thắt thật ở đâu (repo theo "đo trước, sửa sau").
    # Chỉ log, không đổi hành vi. Bảng phân rã in ở cuối.
    timings: list[tuple[str, float]] = []
    _clock = time.perf_counter()

    def mark(label: str) -> None:
        nonlocal _clock
        now = time.perf_counter()
        dt = now - _clock
        timings.append((label, dt))
        log.info("⏱ %s: %.1fs", label, dt)
        _clock = now

    # 1. Tách âm thanh
    abort_if_cancelled()
    progress("extract", 0.0, "Đang tách âm thanh khỏi video")
    source_wav = extract_audio(video_path, workdir / "source.wav")
    samples, rate = read_wav(source_wav)
    progress("extract", 1.0, "Đã tách âm thanh")
    mark("Tách âm thanh")

    # 2. Nhận diện giọng nói — ffmpeg định khung thời gian, Whisper chép nội dung
    abort_if_cancelled()
    regions = plan_utterances(
        source_wav, samples, rate, media.duration,
        max_gap=options.max_utterance_gap,
        max_duration=options.max_utterance_seconds,
    )
    log.info("ffmpeg tìm thấy %d lượt phát ngôn", len(regions))
    language, segments, stt_warnings = transcribe_regions(
        backend, samples, rate, regions, workdir,
        workers=options.stt_workers, sentence_level=options.sentence_level_timing,
        progress=progress, should_cancel=should_cancel,
    )
    warnings.extend(stt_warnings)

    # 2b. Gộp mảnh vụn thành câu trọn theo dấu câu: ffmpeg cắt theo im lặng nên hay chẻ
    # một câu thành nhiều vùng ở chỗ ngừng lấy hơi. Gộp lại cho tiếng Việt liền mạch,
    # dịch đúng cả câu, và hết cảnh "Hôm ...(nghỉ)... nay".
    segments = merge_sentence_fragments(segments, options.max_utterance_seconds)
    mark("Nhận diện + tách câu")

    # 3. Dịch (các lô chạy song song — xem pipeline/translate.py)
    segments = translate_segments(backend, segments, progress, should_cancel,
                                  workers=options.translate_workers)
    mark("Dịch")

    # 4. Tạo giọng đọc
    attempted = sum(1 for seg in segments if seg.text_vi.strip())
    if options.tts_is_metered and attempted > options.tts_daily_budget:
        warnings.append(
            f"Video này cần khoảng {attempted} lượt gọi Gemini TTS, vượt hạn mức miễn phí "
            f"(~100 lượt/ngày). Một số lời thoại có thể bị bỏ trống."
        )

    fitted, tts_warnings = synthesize_segments(
        backend, segments, options.voice_id,
        workers=options.tts_workers,
        max_speedup=options.tts_max_speedup,
        total_duration=media.duration,
        # Đọc lại lượt có lỗ hổng im lặng bất thường — chỉ với backend local (miễn phí) VÀ có
        # khuyết tật đó (VieNeu/edge). Gemini tính tiền theo lượt; OmniVoice tự vá (postprocess)
        # nên resynthesize_holes=False để khỏi tốn single-synth đắt ở video dài.
        resynthesize_bad=(not options.tts_is_metered) and options.resynthesize_holes,
        fill_slowdown=options.tts_fill_slowdown,
        progress=progress,
        should_cancel=should_cancel,
    )
    warnings.extend(tts_warnings)
    # Chốt mốc phát thật (chống hai lượt đè nhau) TRƯỚC khi dựng phụ đề,
    # để cue bám theo tiếng nói thật chứ không theo mốc lý thuyết.
    placed = plan_placement(fitted)
    mark("Đọc giọng (TTS + ép khung)")

    # 5. Phụ đề — dựng SAU giọng đọc để cue bám theo thời lượng đọc thật
    abort_if_cancelled()
    progress("subtitle", 0.0, "Đang dựng file phụ đề")
    srt_path = workdir / "output.srt"
    srt_path.write_text(build_srt(segments), encoding="utf-8")
    progress("subtitle", 1.0, "Đã tạo phụ đề")
    mark("Phụ đề")

    # 6. Dựng track lồng tiếng dài đúng bằng video
    abort_if_cancelled()
    progress("assemble", 0.0, "Đang ghép các lượt thoại thành một track")
    dubbed_wav = build_audio_track(placed, media.duration, workdir / "dubbed.wav")
    progress("assemble", 1.0, "Đã ghép âm thanh")
    mark("Ghép track")

    # 7. Ghép vào video, loại bỏ hoàn toàn audio gốc
    abort_if_cancelled()
    progress("mux", 0.0, "Đang ghép âm thanh vào video")
    out_video = run_mux(video_path, dubbed_wav, workdir / "output.mp4")
    progress("mux", 1.0, "Hoàn tất")
    mark("Ghép vào video (mux)")

    # Bảng phân rã: nút thắt thật ở đâu, mỗi bước bao nhiêu % — dùng để tối ưu đúng chỗ.
    total = sum(dt for _, dt in timings)
    breakdown = "  ·  ".join(f"{label} {dt:.1f}s ({dt / max(total, 1e-9) * 100:.0f}%)"
                             for label, dt in timings)
    log.info("⏱ TỔNG %.1fs cho %d câu — %s", total, len(segments), breakdown)

    return PipelineResult(
        video_path=str(out_video),
        srt_path=str(srt_path),
        language=language,
        segment_count=len(segments),
        attempted_count=attempted,
        spoken_count=len(fitted),
        warnings=warnings,
    )
