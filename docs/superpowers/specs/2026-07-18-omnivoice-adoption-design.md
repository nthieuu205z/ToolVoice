# Thiết kế: Áp dụng OmniVoice làm engine giọng nhân bản

Ngày: 2026-07-18 · Trạng thái: đã brainstorm, chờ duyệt để lên plan

## Mục tiêu

Giọng **nhân bản** được đọc bằng **OmniVoice** (chất lượng người dùng đã nghe và chọn);
giọng **dựng sẵn** giữ nguyên engine đang cấu hình (VieNeu/edge). Người dùng chỉ chọn
giọng — công cụ **tự định tuyến** đúng engine. Nền tảng v1 (backend `OmniVoiceSynthesizer`
sequential, tests xanh, đã cài phòng thủ không đụng torch/VieNeu) đã có; đợt này biến nó
thành first-class.

Bối cảnh phần cứng: [[hardware-setup]] (GPU 12 GB). Cách làm: [[workflow-and-style]]
(đo trước sửa sau, công thức tự thích nghi). Nền kỹ thuật: [[perf-and-quality-work]].

## Quyết định đã chốt (qua brainstorm)

1. **Định tuyến theo loại giọng** (không phải mặc định toàn cục, không phải tối giản).
2. **Quản lý model tích hợp giao diện** (nút tải + %, nạp sẵn lúc boot — như VieNeu).
3. **Dùng `duration` gốc của OmniVoice để căn khớp** — nhưng **A/B trước**, hơn thật mới thay atempo.

## Kiến trúc

### 1. Định tuyến (một quyết định mỗi job)

Mỗi job dùng **một** giọng, nên chọn engine ngay lúc bắt đầu job, không phải mỗi câu:

- `custom_voices.is_custom(voice_id)` (id `clone-…`) → **OmniVoice**
- ngược lại → engine của `TTS_PROVIDER` (VieNeu preset / edge…)

Hiện thực bằng **provider hiệu lực theo job**: `create_job` đã có `voice_id`, nên tính
`effective_tts_provider` rồi truyền vào `build_backend` qua closure `backend_factory`.
KHÔNG cần lớp routing-wrapper → `batch_size`/`synthesize_batch` giữ nguyên đơn giản.

Config mới: `CLONE_TTS_PROVIDER` (mặc định `omnivoice`; **tự lùi về `vieneu`** nếu OmniVoice
chưa cài/chưa tải model — không cài gì thì không hỏng).

### 2. Tốc độ — gộp lô GPU cho OmniVoice

`generate()` nhận **list** text → thêm `synthesize_batch` + `batch_size` **tự suy từ VRAM**
(như VieNeu, không ghim số). ⚠️ **Ẩn số**: chưa biết OmniVoice gộp lô có lợi không (có thể
nghẽn phóng kernel như VieNeu, hoặc tràn VRAM). → **spike đo trước** trên card 12 GB; nếu
gộp lô không lợi thì lùi về sequential-song-song. Báo số thật trước khi chốt cách làm.

### 3. Tham số sinh (đặt cứng giá trị tốt, KHÔNG phơi ra UI per-video)

Bảng knob của Space demo là để tinh chỉnh tay từng clip — pipeline tự động không cần.
Nhưng dùng ngầm mấy cái:

- `num_step` (Inference Steps, mặc định 32): dial tốc độ⇄chất lượng — **tinh trong spike**
  (thử 16/24/32, chọn mặc định). Đây là đòn bẩy tốc độ chính.
- `language="Vietnamese"`: luôn dub sang Việt → nhích chất lượng/tốc độ miễn phí.
- `postprocess_output=on` ("remove long silences"): nhiều khả năng OmniVoice **tự vá lỗ hổng
  im lặng** (đúng khuyết tật Fix A của VieNeu) → đường clone khỏi cần logic đọc-lại-lỗ-hổng.
- Tiền xử lý ref (thêm dấu câu vào ref_text): bật — vì ref_text từ Whisper hay thiếu dấu chấm cuối.
- `denoise=on`, `guidance_scale=2.0`: để mặc định.

### 4. Căn khớp bằng `duration` gốc (có A/B gate) — ĐÃ ĐO: KHÔNG LÀM

**Kết luận A/B (2026-07-19):** cho nghe cùng một câu ép về 80% độ dài — `spike_natural.wav`
(đọc tự nhiên, KHÔNG ép) nghe hay nhất; cả `duration` gốc lẫn `atempo` (đều ép ngắn) đều
kém hơn. Tức là **ép ngắn kiểu nào cũng hại chất lượng**, không riêng atempo. Nên KHÔNG thay
atempo bằng native duration — không có lợi. Giữ nguyên `fit_to_window`, vốn ĐÃ ưu tiên đọc
tự nhiên (tràn vào khoảng lặng trước, chỉ tăng tốc khi cùng đường, ≤1,3×). OmniVoice cũng đi
qua đúng đường đó nên tự thừa hưởng hành vi ít-ép-nhất. Phần căn chỉnh 1:30–1:40 giữ nguyên.

### 5. ref_text

Giữ bước Whisper chép ref_text một lần (ghi sidecar `<id>.reftext.txt`) — tái dùng stack,
kiểm soát tốt. (Ghi chú: OmniVoice tự chép được nếu ref_text rỗng; ta không dựa vào để khỏi
thêm phụ thuộc ASR của nó.)

### 6. Quản lý model (giao diện, như VieNeu)

Thêm OmniVoice vào `model_store`: nút tải + thanh %, và **prewarm** lúc boot khi model đã có
→ job clone đầu không khựng ~5 phút. Đường cài ghi rõ trong tài liệu (mẹo `--no-deps` +
`accelerate` để né xung đột numpy/librosa).

### 7. Nhân bản & UX giọng

- Bật nhân bản khi CÓ engine clone (VieNeu **hoặc** OmniVoice) — nay đang khóa cứng `vieneu`.
  → có thể chạy edge preset + OmniVoice clone.
- Danh sách giọng = preset (theo `TTS_PROVIDER`) + clone (luôn có). Kiểm tra hợp lệ nhận cả hai.
- Nghe thử giọng clone tạo bằng đúng engine clone (để preview khớp job).
- Xóa giọng clone → dọn luôn sidecar `reftext` (+ `forget_clone` của engine tương ứng).

## Kiểm thử (TDD)

Viết test trước cho: bộ giải-provider theo job, gộp lô, bật-nhân-bản theo engine, dọn sidecar
khi xóa. Giữ toàn bộ suite xanh.

## Ngoài phạm vi (để sau)

- Voice design (`instruct`) — không hợp luồng clone-từ-mẫu.
- Phơi các knob sinh ra UI per-video.
- Batched codec / tinh chỉnh sâu ngoài `num_step`.

## Phân đợt

- **① Spike đo (gate)**: gộp lô có lợi không + `num_step` tốt nhất + A/B `duration` vs atempo,
  trên 12 GB. Số ở đây quyết cách làm đợt ② và ⑥.
- **② Làm cứng engine OmniVoice**: `synthesize_batch`/`batch_size` tự thích nghi, `language`,
  `postprocess`, cấu hình sinh.
- **③ Định tuyến + config**: `CLONE_TTS_PROVIDER`, effective provider, kiểm tra hợp lệ, liệt kê giọng.
- **④ Nhân bản/UX giọng**: bật nhân bản theo engine, preview đúng engine, dọn sidecar khi xóa.
- **⑤ model_store + prewarm + UI tải model** (frontend app.js).
- **⑥ Căn khớp `duration`** (nếu spike nói hơn) — ghép vào nhánh clone.
- **⑦ Tài liệu**: README + HUONG_DAN_WINDOWS + .env.example, ghi rõ license **CC-BY-NC**.

③④⑤ không phụ thuộc spike; ②⑥ thì có. Spike chạy trước.
