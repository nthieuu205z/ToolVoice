# Cài ToolVietSub trên Windows — máy có GPU rời NVIDIA

Hướng dẫn này dành cho laptop/PC Windows 10/11 64-bit có card NVIDIA. GPU tăng tốc **cả hai
bước nặng nhất**: **nhận diện giọng nói (Whisper)** và **tạo giọng đọc (VieNeu)** — cả hai
đều gộp lô trên GPU, cỡ lô tự suy từ VRAM trống. Chỉ bước dịch cần mạng (Gemini). Không có
GPU thì vẫn chạy được nhưng chậm hơn nhiều (tự lùi về CPU).

## Cần chuẩn bị

- Windows 10/11 64-bit, card NVIDIA (VRAM từ 4 GB là thoải mái), driver NVIDIA bản mới.
  **Không cần cài CUDA Toolkit riêng** — PyTorch mang sẵn runtime CUDA trong gói cài.
- Khoảng 8 GB đĩa trống (thư viện + model).
- Khóa Gemini API (dùng lại khóa trong `.env` từ máy cũ).

## 1. Chép dự án từ máy cũ

Copy cả thư mục `ToolVietSub` sang máy mới, **trừ** các thư mục sau (đừng chép):

| Chép | KHÔNG chép |
|---|---|
| toàn bộ code | `.venv/` — binary của macOS, vô dụng trên Windows |
| `.env` — chứa khóa API, giữ kín, đừng gửi qua kênh công khai | `jobs/` — kết quả cũ, chép hay không tùy bạn |
| `custom_voices/` — giọng nhân bản của bạn nằm ở đây, sang máy mới dùng tiếp được | `__pycache__/` |
| `previews/` — file nghe thử của giọng, chỉ phục vụ qua API bảo vệ (tùy chọn) | cache Hugging Face (`~/.cache/huggingface`) — dùng symlink kiểu Mac, chép sang Windows sẽ hỏng; tải lại sạch hơn |

## 2. Cài công cụ nền

Mở **PowerShell** và chạy:

```powershell
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

Đóng PowerShell, **mở cửa sổ mới** (để PATH cập nhật), kiểm tra đủ ba thứ:

```powershell
python --version    # Python 3.12.x
ffmpeg -version     # ffmpeg version ...
nvidia-smi          # thấy tên card + driver là ổn
```

Nếu `ffmpeg` không được nhận: mở `.env`, điền đường dẫn đầy đủ vào `FFMPEG_BIN` và
`FFPROBE_BIN` (ví dụ `C:\ffmpeg\bin\ffmpeg.exe`).

## 3. Môi trường Python

```powershell
cd C:\duong\dan\toi\ToolVietSub
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[vieneu]"
```

Hai lỗi kinh điển của PowerShell:

- **"running scripts is disabled"** khi activate → chạy
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` rồi mở lại PowerShell.
- Ngoặc kép quanh `".[vieneu]"` là **bắt buộc** — PowerShell coi `[...]` là ký tự đặc biệt.

## 4. Bật GPU

`pip` mặc định cài PyTorch bản CPU. Thay bằng bản CUDA và cài thêm `transformers`
(engine GPU của VieNeu cần nó, engine CPU thì không):

```powershell
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers
```

(`cu126` = CUDA 12.6. Nếu lệnh lỗi vì phiên bản đã đổi, lấy lệnh mới nhất tại
<https://pytorch.org/get-started/locally/> — chọn Windows / Pip / CUDA.)

Kiểm tra — dòng dưới phải in `True` kèm tên card:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4b. (Tùy chọn) OmniVoice — giọng nhân bản chất lượng cao

Mặc định `CLONE_TTS_PROVIDER=omnivoice`: khi bạn chọn một **giọng nhân bản**, tool đọc bằng
OmniVoice (nhân bản zero-shot, nghe tự nhiên hơn hẳn VieNeu); giọng **dựng sẵn** vẫn dùng
`TTS_PROVIDER`. Nếu chưa cài OmniVoice, tool **tự lùi về VieNeu** — không hỏng gì.

Cài (chỉ phần *inference*, né xung đột `numpy 2.x` ↔ `librosa/numba` mà ta không cần):

```powershell
pip install omnivoice --no-deps
pip install accelerate
```

Model **~3,3 GB** tự tải lần đầu, hoặc bấm nút tải trong mục **"Model trên máy"** (có thanh %).
Lời của clip mẫu (`ref_text`) do Whisper chép một lần rồi nhớ cạnh clip — bạn không phải nhập.

> **License:** trọng số OmniVoice là **CC-BY-NC (phi thương mại)** — dùng cá nhân thoải mái,
> nhưng nếu thương mại hoá thì đây là ràng buộc (VieNeu là Apache-2.0, không vướng).

Không muốn dùng OmniVoice? Đặt `CLONE_TTS_PROVIDER=vieneu` (nhân bản bằng VieNeu như cũ) hoặc
`none` (tắt nhân bản) trong `.env`.

## 5. Kiểm tra `.env`

Nếu đã chép `.env` từ máy cũ thì giữ nguyên. Nếu làm mới: `copy .env.example .env`
rồi điền `GEMINI_API_KEY`. Cấu hình khuyến nghị (đã là mặc định trong `.env` hiện tại):

```
GEMINI_BACKEND=vertex        # khóa tạo trong Google Cloud; khóa AI Studio thì để developer
STT_PROVIDER=whisper         # nhận diện trên GPU (miễn phí) + cho mốc từng từ để cắt câu
STT_WORKERS=1                # Whisper tự tuần tự hóa bên trong; để 1 cho khỏi tranh CPU
TTS_PROVIDER=vieneu          # 14 giọng ba miền + nhân bản giọng, chạy trên GPU
SENTENCE_LEVEL_TIMING=true   # đọc theo từng câu theo mốc từng từ → bám hình sát
MAX_UTTERANCE_SECONDS=12     # trần vùng ffmpeg TRƯỚC khi tách câu, đừng tăng nếu không có lý do
```

## 6. Chạy

```powershell
.venv\Scripts\activate
uvicorn backend.main:app --port 8000
```

Mở <http://localhost:8000>. Server chỉ nghe trên máy của bạn (localhost), không mở ra mạng.
Muốn đỡ gõ lệnh, tạo file `run.bat` trong thư mục dự án với nội dung:

```bat
@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
uvicorn backend.main:app --port 8000
```

## 7. Lần chạy đầu tiên

1. Vào mục **"Model trên máy"** trên giao diện, bấm tải model VieNeu (~610 MB), đợi 100%.
2. Chạy thử một video **ngắn** trước. Lần đầu tạo giọng, engine GPU sẽ **tự tải thêm
   trọng số PyTorch** (cỡ 1–2 GB, chỉ một lần) — tiến trình hiện ở **cửa sổ terminal**,
   không có thanh trên web, đừng tưởng treo.
3. Xác nhận đang chạy GPU thật: terminal phải có dòng
   `VieNeu-TTS v3 Turbo ready (backend=pytorch)`.
   Nếu là `backend=onnx` tức đang chạy CPU → xem mục sự cố bên dưới.
   Chắc ăn hơn: mở `nvidia-smi` khi đang ở bước "Đang tạo giọng đọc" — python phải chiếm VRAM.

## 8. Cách dùng hằng ngày

Giao diện y hệt trên máy cũ:

- **Thêm video**: kéo thả (.mp4, .mkv, .mov… tối đa 8 GB), có phần trăm tải lên.
- **Chọn giọng**: 14 giọng Việt ba miền, bấm nghe thử từng giọng.
- **Nhân bản giọng**: bấm *＋ Nhân bản giọng*, đặt tên + tải mẫu audio **3–8 giây, một
  người nói, ít tạp âm**. Chỉ dùng giọng bạn có quyền sử dụng.
- **Bắt đầu chuyển đổi**: theo dõi 7 bước trên thẻ job. Chạy song song tối đa 2 video,
  video nộp thêm tự xếp hàng; bấm **Hủy** dừng được giữa chừng; đóng trình duyệt hay
  khởi động lại máy vẫn còn nguyên danh sách.
- **Xong**: tải video lồng tiếng + phụ đề `.srt`. Nếu có đoạn bị thiếu (mạng chập chờn…)
  màn kết quả sẽ báo đỏ kèm mốc thời gian — chạy lại video thường khắc phục được.
- Đổi cấu hình: sửa `.env` → `Ctrl+C` tắt server → chạy lại.

## 9. Sự cố thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, mở lại PowerShell |
| `torch.cuda.is_available()` in `False` | Driver cũ (chạy `nvidia-smi` xem có lỗi không) hoặc lỡ cài torch bản CPU — làm lại bước 4 |
| Terminal in `backend=onnx` dù có GPU | Thiếu `transformers` hoặc torch đang là bản CPU — làm lại bước 4 rồi khởi động lại server |
| Lần đầu chạy GPU báo `No module named ...` | Codec của VieNeu nạp code từ Hugging Face nên có thể đòi thêm gói lẻ — `pip install <tên gói>` rồi chạy lại, chỉ phải làm một lần |
| Lần đầu nạp model rất chậm | Windows Defender quét file — thêm thư mục dự án vào *Exclusions*; laptop nhớ cắm sạc (chạy pin bị bóp xung) |
| Cổng 8000 bận | `uvicorn backend.main:app --port 8080` |
| GPU trục trặc, muốn ép về CPU | `pip uninstall -y torch torchaudio` rồi `pip install torch torchaudio` (bản PyPI mặc định là CPU) — engine tự quay về ONNX/CPU, mọi tính năng kể cả nhân bản giọng vẫn chạy |
