# ToolVietSub — Brief thiết kế lại frontend (v2)

> Bản này **thay thế** brief v1. Backend đã hoàn chỉnh và đóng băng — thiết kế chỉ việc
> gọi đúng các API dưới đây. Sản phẩm cuối: `index.html` + `style.css` + `app.js`
> (HTML/CSS/JS thuần, **không build step**, không framework), được FastAPI phục vụ
> dưới dạng file tĩnh tại `http://localhost:8000`.

## Ứng dụng làm gì

App chạy local trên máy người dùng: nhận video bất kỳ (.mp4, .mkv…), tự nhận diện lời
thoại, dịch sang tiếng Việt, lồng giọng đọc tiếng Việt thay hẳn audio gốc, và xuất kèm
phụ đề `.srt`. Một người dùng duy nhất, nhưng chạy được **nhiều video cùng lúc**
(tối đa 2 song song, video thêm vào xếp hàng chờ).

Toàn bộ chữ trên giao diện là **tiếng Việt**.

---

## Các khối chức năng cần có trên giao diện

### 1. Thêm video
- Vùng kéo-thả / bấm chọn file video (`accept="video/*"`). Sau khi chọn: hiện tên file,
  dung lượng, thời lượng (đọc qua thẻ `<video>` ẩn), nút bỏ chọn.
- Nút hành động chính **"Bắt đầu chuyển đổi"** — chỉ bật khi đủ 3 điều kiện:
  đã chọn file + đã chọn giọng + mọi model trên máy đã sẵn sàng. Kèm dòng gợi ý
  nói rõ đang thiếu gì.
- **Tiến trình upload**: gửi bằng `XMLHttpRequest` (bắt buộc — `fetch` không có
  `upload.onprogress`). Hiện phần trăm + `1.2 GB / 3.4 GB`. Khi đạt 100% chuyển sang
  trạng thái "Đang kiểm tra video…" (server còn ghi file + ffprobe). Video tối đa 8 GB.
- Nộp xong: form reset về trống (giữ giọng đã chọn), video xuất hiện trong danh sách,
  người dùng thêm video khác được ngay.

### 2. Model chạy trên máy (0–2 thẻ, tùy cấu hình server)
- `GET /api/model` trả danh sách model cần có (Whisper nhận diện, VieNeu giọng đọc).
- Mỗi model một thẻ: tên, trạng thái (sẵn sàng / chưa tải / đang tải / lỗi), dung lượng.
- Model chưa có: nút **"Tải model"** → `POST /api/model/download?key=<key>` rồi theo dõi
  tiến trình qua SSE `GET /api/model/events` (mỗi event là `{"models":[…]}`; stream tự
  đóng khi không còn model nào đang tải). Hiện thanh phần trăm + `243.1 / 486.2 MB`.
- Khi `models` rỗng (`required=false`): ẩn cả khối.
- Chừng nào còn model chưa sẵn sàng thì khóa nút "Bắt đầu chuyển đổi".

### 3. Chọn giọng đọc
- `GET /api/voices` → danh sách giọng (số lượng thay đổi theo cấu hình: 10–15+).
  Mỗi giọng: tên hiển thị, nút **nghe thử** phát file `preview_url` (có thể rỗng →
  ẩn nút), chọn một giọng duy nhất (radio-style).
- Giọng có `custom: true` là **giọng nhân bản**: thêm nút xóa (×) → xác nhận →
  `DELETE /api/voices/custom/{id}` → làm mới danh sách.

### 4. Nhân bản giọng (chỉ hiện khi được bật)
- `GET /api/voices/cloning` → `{"enabled": true|false}`. `false` → ẩn hoàn toàn.
- Form: tên giọng (≤40 ký tự) + file audio (`accept="audio/*"`, mẫu 3–8 giây, một
  người nói, ít tạp âm) + nút tạo. Bắt buộc kèm một dòng nhắc:
  *"Chỉ dùng giọng bạn có quyền sử dụng."*
- `POST /api/voices/custom` (FormData: `name`, `audio`) → giọng mới xuất hiện ngay,
  tự chọn nó. File nghe thử được tạo **ở nền** (lần đầu chờ nạp engine ~90 giây) —
  hiện thông báo *"File nghe thử sẽ sẵn sàng sau ít phút."*
- Lỗi trả về trong `detail` (tiếng Việt sẵn): mẫu quá ngắn, file không phải audio,
  provider không hỗ trợ…

### 5. Danh sách video (trái tim của giao diện)
- `GET /api/jobs` → `{"jobs":[…]}` **mới nhất trước**. Poll mỗi ~1,5 giây khi còn job
  active; ngừng poll khi tất cả đã kết thúc.
- Mỗi job một thẻ, dữ liệu từ snapshot (xem schema dưới):
  - **queued** — "Chờ đến lượt" + nút Hủy (hủy tức thì).
  - **running** — nhãn bước hiện tại + thanh phần trăm + nút Hủy.
  - **cancelling** — "Đang dừng…", nút Hủy vô hiệu (dừng hợp tác, mất vài giây).
  - **cancelled** — "Đã hủy".
  - **error** — hiện `message` (đã là câu tiếng Việt hoàn chỉnh).
  - **done** — hai nút tải: **"Tải video"** (`/api/jobs/{id}/download/video`) và
    **"Tải phụ đề (.srt)"** (`…/download/srt`). Hiện `Đã đọc {spoken}/{attempted} lượt thoại`.
  - **done + degraded=true** — hoàn tất nhưng thiếu nhiều giọng đọc: phải nhìn ra
    ngay là có vấn đề (đỏ/cảnh báo), kèm danh sách `warnings` (các câu tiếng Việt sẵn).
    Nút tải vẫn hoạt động.
- Hủy: `POST /api/jobs/{id}/cancel` → cập nhật thẻ theo poll.
- Danh sách **sống sót qua reload trang và restart server** — chỉ cần render lại từ
  `GET /api/jobs`, không cần lưu gì phía client.

### 6. Chế độ xem thử (bắt buộc giữ)
Khi `fetch("/api/voices")` thất bại (mở file không có server — chính là môi trường
Claude Design): bật cờ mock, hiện badge "Chế độ xem thử", dùng dữ liệu giả cho giọng
và mô phỏng một job chạy để xem được mọi trạng thái thẻ. Nhờ cơ chế này bản thiết kế
tự chạy được ngay trong trình xem.

---

## Hợp đồng API (backend đã đóng băng)

```
GET  /api/voices                     → [ {id, display_name, preview_url, custom} ]
GET  /api/voices/cloning             → {enabled: bool}
POST /api/voices/custom              FormData(name, audio) → {id, display_name, preview_url:"", custom:true}
DELETE /api/voices/custom/{id}       → {deleted} | 404
GET  /previews/{voice_id}.wav        → file audio nghe thử (tĩnh)

GET  /api/model                      → {required, ready, models:[{key,label,ready,status,message,percent,downloaded_mb,total_mb}]}
                                       status: "ready"|"idle"|"downloading"|"error"
POST /api/model/download?key={key}   → {started,key} | 409 (đang tải cái khác / đang xử lý video)
GET  /api/model/events               → SSE, mỗi event: data: {"models":[…]}

POST /api/jobs                       multipart(video, voice_id) → {job_id}
                                       400 giọng sai / video hỏng · 413 quá 8 GB · 500 thiếu API key
                                       (message lỗi tiếng Việt nằm trong body.detail)
GET  /api/jobs                       → {jobs:[snapshot…]} mới nhất trước
GET  /api/jobs/{id}                  → snapshot
POST /api/jobs/{id}/cancel           → snapshot | 404 | 409 (đã kết thúc)
GET  /api/jobs/{id}/events           → SSE snapshot từng job (tùy chọn — poll danh sách là đủ)
GET  /api/jobs/{id}/download/video   → file (409 khi chưa done)
GET  /api/jobs/{id}/download/srt     → file (409 khi chưa done)
```

**Schema snapshot của job:**
```json
{
  "job_id": "bf68709df1da",
  "status": "queued|running|cancelling|cancelled|done|error",
  "stage": "extract|transcribe|translate|synthesize|subtitle|assemble|mux",
  "percent": 61.5,
  "message": "Đang đọc lượt thoại 12/32",
  "filename": "video-cua-toi.mp4",
  "voice_id": "truc-ly",
  "created_at": 1783701234.5,
  "warnings": ["…câu tiếng Việt hoàn chỉnh…"],
  "attempted_count": 32,
  "spoken_count": 30,
  "silent_ratio": 0.062,
  "degraded": false
}
```

**Nhãn tiếng Việt của 7 bước** (đúng thứ tự chạy):

| stage | nhãn |
|---|---|
| extract | Đang trích xuất âm thanh |
| transcribe | Đang nhận diện giọng nói |
| translate | Đang dịch sang tiếng Việt |
| synthesize | Đang tạo giọng đọc |
| subtitle | Đang tạo phụ đề |
| assemble | Đang ghép âm thanh |
| mux | Đang ghép video |

---

## Ràng buộc kỹ thuật bắt buộc

1. **HTML + CSS + JS thuần, không build step** — file tĩnh FastAPI phục vụ trực tiếp.
2. **Upload video phải dùng `XMLHttpRequest`** để có `upload.onprogress`.
3. **Giữ nguyên các thuộc tính `data-*`** — test Playwright bám vào chúng:
   - thẻ giọng: `data-voice-id="…"`; nút xóa giọng nhân bản: `data-action="voice-delete"`
   - nút nhân bản: `data-action="clone-toggle"`, `data-action="clone-create"`
   - thẻ model: `data-model-key="…"`, `data-model-state="ready|missing|downloading|error"`,
     nút tải: `data-model-action="download"`
   - thẻ job: `data-job-id="…"`, `data-status="…"`; nút hủy: `data-action="cancel"`;
     nút tải: `data-download="video"` và `data-download="srt"`
4. Mọi text động đổ vào DOM bằng `textContent` (không `innerHTML` với dữ liệu server).
5. Poll danh sách job ~1,5 s khi còn job active; dừng khi hết.
6. Giữ chế độ xem thử (mục 6) để bản thiết kế tự chạy không cần server.

## Được toàn quyền thay đổi

Toàn bộ phần nhìn: bố cục, palette, typography, icon, animation, cách tổ chức
các khối (một cột / hai cột / sidebar…), dark/light. Không cần giữ bất kỳ class CSS
hay cấu trúc DOM cũ nào — miễn là các `data-*` ở trên còn nguyên và luồng chức năng
đầy đủ. Font hiện tại (Be Vietnam Pro + Bricolage Grotesque, Google Fonts) chỉ là
gợi ý, thay được.

## Sản phẩm cần trả về

`index.html`, `style.css`, `app.js` — đầy đủ chức năng theo brief, chạy được ngay
ở chế độ xem thử, và hoạt động thật khi đặt vào `web/static/` của project.
