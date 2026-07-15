# ToolVietSub

Lồng tiếng Việt và tạo phụ đề tiếng Việt cho video, chạy hoàn toàn trên máy bạn.

Đưa vào một video bất kỳ (.mp4, .mkv, .mov…), nhận về:
- video với âm thanh gốc **được thay hẳn** bằng giọng đọc tiếng Việt
- file phụ đề `.srt` tiếng Việt riêng

Hai cấu hình chính, đổi qua `.env`:

| Bước | Nhanh nhất (Gemini trả phí) | Miễn phí |
|---|---|---|
| Khung thời gian | ffmpeg `silencedetect` — trên máy | như nhau |
| Nhận diện giọng nói | **Gemini Flash, 8 luồng song song** | Whisper trên máy (chậm, tuần tự) |
| Dịch | Gemini Flash (1–4 lượt gọi cho cả video) | như nhau |
| Giọng đọc | VieNeu / edge-tts — trên máy, không giới hạn | như nhau |

Cấu hình nhanh: `STT_PROVIDER=gemini` + `STT_WORKERS=8`. Bước nhận diện gửi từng vùng
tiếng nói cho Gemini (đã tắt thinking — chép lời không cần suy luận, đo được nhanh gấp
~2–7 lần mỗi request), video 19 phút nhận diện xong trong ~100 giây thay vì hàng chục phút
Whisper CPU. Muốn về miễn phí: `STT_PROVIDER=whisper` + `STT_WORKERS=1`.

### Chọn giọng đọc

Đổi `TTS_PROVIDER` trong `.env`:

| | Số giọng | Cần tải | Tốc độ (video 8,8 phút) | Ghi chú |
|---|---|---|---|---|
| **`edge`** (mặc định) | **14** | không | ~0,5 phút | 2 giọng Việt bản địa + 12 giọng đa ngôn ngữ |
| **`vieneu`** | **14** | ~610 MB | ~4 phút | Giọng Việt bản địa **cả ba miền**, chạy offline hoàn toàn |
| `gemini` | 30 | không | — | Trần ~100 lượt/ngày → video dài sẽ câm giữa chừng |

Mười hai giọng "Multilingual" của edge-tts mang nhãn `en-US`, `fr-FR`, `ko-KR`… nhưng đọc được tiếng Việt. Đây không phải suy đoán: mỗi giọng được cho đọc một câu tiếng Việt rồi bắt Whisper nghe lại — cả 12 đều được nhận là tiếng Việt với độ khớp 0,94–1,00.

Dùng VieNeu (`uv pip install -e '.[vieneu]'`) khi bạn cần **giọng miền Trung / miền Nam**, hoặc muốn chạy hoàn toàn offline không phụ thuộc Microsoft. Model tải ngay trên giao diện, có thanh tiến trình.

### Nhân bản giọng (chỉ VieNeu)

Bấm **＋ Nhân bản giọng** cạnh tiêu đề "Chọn giọng đọc", đặt tên và tải một đoạn audio **3–8 giây** (một người nói, ít tạp âm). Giọng mới xuất hiện ngay trong danh sách; file nghe thử được tạo ở nền (lần đầu chờ nạp engine ~90 giây). Chỉ dùng giọng bạn có quyền sử dụng.

Vài điều đáng biết:
- Không có gì phải huấn luyện — engine trích embedding từ mẫu ngay lúc dùng (vài giây, một lần mỗi phiên). Xóa giọng là sạch.
- Giọng dựng sẵn chạy **không cần torch**, nhưng nhân bản cần `torch/torchaudio` (~490 MB, đã nằm trong extra `[vieneu]`) — bước trích đặc trưng của speaker encoder dùng torchaudio, đây là hạn chế của thư viện.
- Id giọng đã xóa **không bao giờ được cấp lại** (file `.tombstone` giữ chỗ): engine giữ embedding theo id trong RAM, tái dùng id với mẫu khác sẽ đọc bằng giọng cũ tới khi restart.

---

## Cài đặt

Hướng dẫn dưới đây dành cho macOS. **Windows (kèm bật GPU NVIDIA cho bước giọng đọc):
xem [HUONG_DAN_WINDOWS.md](HUONG_DAN_WINDOWS.md).**

**1. ffmpeg** (bắt buộc — dùng để tách, ghép, chỉnh tốc độ âm thanh)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
```

Nếu ffmpeg nằm ngoài `PATH`, khai báo `FFMPEG_BIN` và `FFPROBE_BIN` trong `.env`.

**2. Python 3.10+ và thư viện**

Dự án dùng [uv](https://docs.astral.sh/uv/). Nếu chưa có:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Rồi:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

**3. Khóa Gemini API** (chỉ bước dịch cần)

Lấy khóa miễn phí tại <https://aistudio.google.com/apikey>, rồi:

```bash
cp .env.example .env
# mở .env và điền GEMINI_API_KEY
```

Nếu khóa của bạn tạo trong **Google Cloud** thay vì AI Studio, Developer API sẽ trả `403 API_KEY_SERVICE_BLOCKED`. Khi đó đặt trong `.env`:

```
GEMINI_BACKEND=vertex
```

Cùng một khóa nhưng đi tới `aiplatform.googleapis.com` (Vertex AI Express Mode) thay vì `generativelanguage.googleapis.com`. Vertex không có `models.list()` và không phải model nào cũng tồn tại ở đó — `gemini-3.5-flash`, `gemini-2.5-flash`, `gemini-2.5-pro` thì có.

**4. Kiểm tra mọi thứ đã sẵn sàng**

```bash
./.venv/bin/python scripts/check_setup.py
```

Script này xác nhận ffmpeg, khóa API, và **kiểm tra tên model đã cấu hình có thật không** — Google đổi tên model khá thường xuyên, nếu sai nó sẽ in ra danh sách model đang khả dụng để bạn sửa `.env`.

---

## Chạy

```bash
./.venv/bin/uvicorn backend.main:app --port 8000
```

Mở <http://localhost:8000>, kéo thả video vào, chọn giọng đọc, bấm **Bắt đầu chuyển đổi**.

Tùy chọn — tạo file nghe thử giọng đọc để bấm nghe ngay trên giao diện (mỗi giọng tốn một lần gọi TTS, chỉ cần chạy một lần):

```bash
./.venv/bin/python scripts/generate_voice_previews.py
```

Hoặc chạy thẳng bằng dòng lệnh, không qua giao diện:

```bash
./.venv/bin/python scripts/run_pipeline_cli.py video.mp4 --voice Charon -v
```

---

## Cách hoạt động

Bảy bước, chạy tuần tự trong `pipeline/runner.py`:

| Bước | Việc làm |
|---|---|
| `extract` | ffmpeg tách audio ra WAV mono 16 kHz |
| `transcribe` | **ffmpeg** định khung từng lượt phát ngôn, **Gemini/Whisper** chép lời cho từng lượt |
| `translate` | Gemini dịch sang tiếng Việt, giữ nguyên số dòng và thứ tự |
| `synthesize` | VieNeu/edge-tts đọc từng lượt thoại bằng giọng đã chọn |
| `subtitle` | Dựng `.srt`, cue bám theo thời lượng giọng đọc thật |
| `assemble` | Đặt từng đoạn vào đúng mốc thời gian trên nền im lặng dài bằng video |
| `mux` | ffmpeg ghép video gốc + track mới, **không** map audio gốc |

### ffmpeg lo thời gian, Gemini lo nội dung

Đây là quyết định thiết kế quan trọng nhất, và nó đến từ đo đạc chứ không phải trực giác.

Gemini có tham số `audio_timestamp`, nhưng nó **chỉ chạy trên Vertex AI**. Với khóa AI Studio (Developer API), khi bị hỏi mốc thời gian Gemini sẽ *bịa ra*. Đo trên 120 giây đầu của một video thật:

```
ffmpeg:  0.79–13.37   13.79–25.00   26.27–45.24   45.70–54.42
Gemini:  0.00–13.90   13.90–25.90   25.90–37.90   ... và một mốc kết thúc ở giây 200
```

Hai mốc đầu còn khớp, sau đó trôi dần, và cuối cùng nó trả về một segment kết thúc ở giây 200 của một đoạn audio chỉ dài 120 giây. Khoảng lặng luôn đúng bằng 0.00s — dấu hiệu rõ ràng là các mốc được suy ra chứ không được đo.

Nên pipeline dùng `silencedetect` của ffmpeg để tìm các vùng có tiếng nói thật, rồi gửi **từng vùng** cho Gemini chép lời. Mốc thời gian là thật, và văn bản chắc chắn thuộc đúng vùng đó.

### Vài điểm đáng biết

- **Audio gốc bị xóa hoàn toàn.** Lệnh mux chỉ map `0:v:0` và `1:a:0`. Có hẳn một test (`tests/test_mux_args.py`) canh để không ai vô tình thêm `-map 0:a`.
- **Video không bị encode lại** (`-c:v copy`) nên nhanh và không giảm chất lượng hình. Nếu container `.mp4` không chứa nổi codec gốc, hệ thống tự lùi về `.mkv`.
- **Phụ đề dựng SAU giọng đọc.** Giọng Việt thường đọc xong sớm hơn khung thời gian gốc; nếu trải cue theo khung thì các cue cuối rơi vào chỗ im lặng. Đo thực tế: 17% cue lệch khỏi tiếng nói → 0% sau khi bám theo thời lượng đọc thật.
- **Mỗi lượt phát ngôn là một mốc neo đồng bộ.** Trần một lượt là `MAX_UTTERANCE_SECONDS` (mặc định 12) — càng ngắn thì tiếng Việt càng bám sát mốc câu gốc, vì cả khối tiếng Việt của một lượt được đặt tại đúng một mốc.
- **Tiếng Việt dài hơn khung gốc → ba nấc, không bao giờ cắt chữ.** (1) Tăng tốc *nhẹ* (≤1,15× — dưới ngưỡng tai người nhận ra) đưa câu về sát khung gốc; (2) phần còn dư tràn tự nhiên vào khoảng lặng phía sau, tới sát mốc câu kế tiếp; (3) hết cả chỗ mượn mới tăng tốc mạnh (tối đa `TTS_MAX_SPEEDUP`), và nếu vẫn dư thì câu sau **lùi lại chờ** thay vì hai giọng đè nhau — phần lệch tự tan ở khoảng lặng kế tiếp. Không bao giờ kéo chậm để lấp khoảng trống — giọng bị kéo lê nghe giả hơn im lặng. Phụ đề luôn bám mốc phát thật sau khi xếp chỗ. Nấc (1) đến từ đo đạc: không có nó, 132/156 lượt của một video thật bị dồn toa vì tiếng Việt thường dài hơn tiếng Anh một chút.
- **Không đoạn nào bị bỏ rơi trong im lặng.** Đoạn nhận diện hỏng (dính 429 lúc 8 luồng dồn dập) được thử lại tuần tự sau khi cơn dồn request dịu; vẫn hỏng thì báo rõ trên màn kết quả kèm mốc thời gian, không lẳng lặng thiếu lời thoại.
- **Chống TTS "chạy hoang".** Model giọng nói tự hồi quy thỉnh thoảng bịa thêm lời khi đầu vào quá ngắn — đo thật: chữ "Và" (2 ký tự) sinh ra 7,1 giây giọng nói, kéo 15 câu sau lệch 3–6 giây. Không văn bản nào được phép sinh nhiều audio hơn `số ký tự ÷ 8 + 1s` (đọc chậm nhất còn hợp lý; VieNeu thực tế đọc 14–17 ký tự/giây) — vượt trần là audio bịa, bị cắt kèm fade, từ thật luôn nằm ở phần đầu.
- **Lời thoại không được đưa trần vào TTS.** Gặp câu hỏi, model tưởng đó là câu lệnh và định trả lời (`"Model tried to generate text, but it should only be used for TTS"`). Pipeline bọc mỗi lượt trong một câu lệnh đọc nguyên văn — đã kiểm chứng là câu lệnh đó không bị đọc thành tiếng.
- **Số dòng dịch phải khớp tuyệt đối.** Lệch một dòng là lệch giờ toàn bộ phần sau, nên hệ thống thử lại một lần rồi báo lỗi thay vì xuất ra video sai tiếng. Prompt dịch cũng cấm lược ý: khung thời gian chỉ quyết định *cách diễn đạt*, không quyết định *lượng thông tin* — câu dài ra đã có cơ chế mượn khoảng lặng ở trên lo.
- **Nhiều video cùng lúc.** Trang chủ là form thêm video + danh sách job bên dưới; mỗi video một thẻ với tiến trình riêng. Tối đa `MAX_CONCURRENT_JOBS` (mặc định 2) video chạy đồng thời, video nộp thêm xếp hàng chờ. Trần đặt thấp có chủ ý: Whisper/VieNeu bị khóa suy luận toàn cục và edge-tts bị trần 2 request đồng thời, nên job thứ ba chủ yếu chen hàng chứ không nhanh thêm.
- **Danh sách job sống sót qua mọi thứ.** Mỗi job ghi `job.json` vào thư mục của nó; đóng tab, quay lại trang chủ, hay khởi động lại server đều thấy nguyên danh sách và tải lại được kết quả cũ. Job đang chạy dở lúc server chết được đánh dấu lỗi kèm lời nhắn "hãy chạy lại" — không bao giờ hiện "đang chạy" ma.
- **Hủy được giữa chừng.** Bấm **Hủy** trên thẻ job (hoặc `POST /api/jobs/{id}/cancel`). Việc hủy là *hợp tác*: máy chủ không giết thread giữa chừng — lúc đó ffmpeg có thể đang ghi file và ONNX đang chạy — mà đặt cờ rồi để pipeline tự dừng ở mốc an toàn gần nhất. Đo thực tế trên video 19 phút: bấm hủy ở đoạn 2/64, dừng hẳn sau **4 giây**. Job còn xếp hàng thì hủy tức thì.

---

## Hạn mức và đánh đổi

### Vì sao mặc định không dùng Gemini TTS

**Bản miễn phí của Gemini cho khoảng 100 lượt gọi TTS mỗi ngày, cho mỗi model** — đo được, không phải suy đoán (`GenerateRequestsPerDayPerProjectPerModel = 100`). Pipeline gọi TTS một lần cho mỗi lượt phát ngôn: video 8,8 phút cần 32 lượt, video 18,9 phút cần 64 lượt. Tức là **khoảng 30 phút video mỗi ngày**, cộng dồn.

`edge-tts` không có trần đó. Đánh đổi:

- **Chỉ 2 giọng tiếng Việt** thay vì 30 giọng của Gemini.
- Là endpoint đọc-thành-tiếng của trình duyệt Edge, dùng theo cách **không chính thức**. Microsoft không cam kết gì; nó có thể ngừng chạy bất cứ lúc nào. Khi đó đổi `TTS_PROVIDER=gemini` trong `.env` là quay lại được ngay.
- **Nó bóp tần suất.** Đo thực tế: 6 luồng song song → 25/32 request thành công; 2 luồng kèm thử lại → 10/10. Vì vậy `TTS_WORKERS=2` và `EDGE_TTS_ATTEMPTS=5`.

### Gemini hay Whisper ở bước nhận diện

`STT_PROVIDER=gemini` (khuyến nghị khi chấp nhận trả phí): mỗi vùng tiếng nói gửi thẳng cho
Gemini, 8 luồng song song, và **tắt thinking** — chép lời không cần suy luận, đo trên
gemini-3.5-flash qua Vertex thấy 4,2–18,9 giây/request khi để mặc định giảm còn ~2,3 giây
khi tắt (model nào không nhận `thinking_config` thì tự lùi về mặc định). Chuẩn hơn
Whisper `small`, nhất là với thuật ngữ và tên riêng.

`STT_PROVIDER=whisper`: miễn phí, không hạn mức, chạy offline. Đo trên máy Apple Silicon
(CPU, model `small`): **8,8 phút audio hết 78 giây**. Lần chạy đầu tải model về (~460 MB).
Muốn chép chính xác hơn thì đổi `WHISPER_MODEL=medium`, chậm hơn.

Lưu ý: pipeline **không** dùng mốc thời gian của model nào cả — ffmpeg đã cho ranh giới
chính xác tuyệt đối. Gemini/Whisper chỉ chép chữ.

### Nếu vẫn muốn dùng Gemini TTS

Đặt `TTS_PROVIDER=gemini`. Khi chạm trần, app **không im lặng giả vờ thành công**: nó dừng thử lại ngay (Google bảo đợi hơn 4 tiếng), báo đỏ ở màn hoàn tất, cho biết bao nhiêu lượt bị bỏ trống, và vẫn cho tải video + phụ đề đầy đủ về. Muốn bỏ trần thì bật thanh toán trong Google Cloud — khi đó **giọng đọc là khoản tốn nhất**, tra giá tại <https://ai.google.dev/gemini-api/docs/pricing>.

---

## Test

```bash
./.venv/bin/python -m pytest
```

Toàn bộ test dùng một Gemini giả (`tests/conftest.py`) — **không gọi API thật, không tốn token**, và không cần ffmpeg.

---

## Cấu trúc

```
backend/     FastAPI: config, quản lý job, các route
pipeline/    7 bước xử lý + adapter Gemini (gemini.py là nơi duy nhất gọi API thật)
web/static/  Giao diện: HTML/CSS/JS thuần, không build step
scripts/     check_setup, chạy CLI, tạo file nghe thử giọng
tests/       Test đơn vị với Gemini giả
jobs/        File tạm và kết quả (đã gitignore)
```
