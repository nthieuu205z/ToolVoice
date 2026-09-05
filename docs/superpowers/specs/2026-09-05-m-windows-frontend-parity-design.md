# Windows Frontend Parity Design

## Mục tiêu

Thay hoàn toàn frontend cũ của `ToolVietSubWin` bằng giao diện operations dashboard đang dùng trong `ToolVietSubMac`, đồng thời giữ nguyên pipeline, nhà cung cấp giọng và hành vi chạy riêng của Windows.

Kết quả phải có cùng cấu trúc, phong cách, luồng thao tác và trạng thái trực tiếp như bản Mac; những thông tin phụ thuộc nền tảng phải được lấy từ backend Windows thay vì ghi cứng macOS hoặc MPS.

## Phạm vi

### Thay thế frontend

- Thay toàn bộ nội dung `web/static/index.html`, `web/static/style.css` và `web/static/app.js` của Windows bằng frontend operations dashboard của Mac.
- Loại bỏ các section, selector, animation và logic mock-preview thuộc frontend Windows cũ.
- Giữ các sửa lỗi đã có trên bản Mac: queue cập nhật qua SSE, không có progress bar trùng trong `SELECTED JOB`, node `Export` hoàn tất đúng trạng thái và nút chọn giọng không tràn chữ.
- Giữ cache-busting cho CSS và JavaScript để trình duyệt không dùng lại frontend cũ.

### Giữ đặc tính Windows

- Không thay pipeline xử lý video, cấu hình provider hoặc cơ chế VieNeu/OmniVoice/Edge hiện có của Windows chỉ để phục vụ việc đổi giao diện.
- Không ghi cứng `MPS`, Apple Silicon hoặc OmniVoice vào dữ liệu runtime. Tên engine, device và batch lấy từ snapshot backend khi có; nếu thiếu thì hiển thị trạng thái trung tính.
- Nội dung mô tả engine trong Voice Lab và Execution Graph phải phù hợp với provider thực tế hoặc dùng nhãn chung `TTS local`.

### Lớp tương thích backend

Frontend mới được phép dùng các endpoint sau:

- `GET /api/voices`
- `POST /api/voices/custom`
- `DELETE /api/voices/custom/{voice_id}`
- `GET /api/voices/{voice_id}/preview`
- `GET /api/model`
- `GET /api/jobs`
- `POST /api/jobs`
- `POST /api/jobs/{job_id}/cancel`
- `DELETE /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/events`
- `GET /api/jobs/{job_id}/download/video`
- `GET /api/jobs/{job_id}/download/srt`
- `GET /api/settings/gemini`
- `POST /api/settings/gemini`
- `POST /api/shutdown`

Các endpoint chưa có trong Windows sẽ được bổ sung bằng phần triển khai nhỏ nhất cần thiết. Chúng không được làm thay đổi provider routing hoặc thuật toán pipeline.

## Kiến trúc

### Static application shell

FastAPI tiếp tục mount `web/static` tại `/`. HTML tạo cấu trúc dashboard; CSS đảm nhiệm layout desktop/mobile; JavaScript giữ một state store nhỏ ở phía client và render giao diện từ API snapshot.

Frontend không dùng framework hoặc bước build mới. Điều này giữ nguyên cách khởi động hiện tại trên Windows và tránh thêm dependency Node.js.

### Luồng job trực tiếp

1. Khi trang tải, client gọi `GET /api/jobs` và render queue, selected job, metrics và execution graph.
2. Với mỗi job đang hoạt động, client mở một `EventSource` riêng tới `/api/jobs/{job_id}/events`.
3. Mỗi snapshot SSE được merge vào state, sau đó cập nhật cả queue và selected job trong cùng lượt render.
4. Khi job kết thúc, stream đóng; queue hiển thị nút tải/xóa và node `Export` được đánh dấu hoàn tất.
5. Nếu SSE lỗi trong khi job vẫn hoạt động, client kết nối lại có giới hạn; refresh định kỳ vẫn là lớp dự phòng.

### Telemetry tương thích

Snapshot Windows được mở rộng bằng các trường không phá vỡ client cũ:

- `elapsed_seconds`
- `stage_elapsed_seconds`
- `eta_seconds`
- `stage_fraction`
- `events`
- `engine`
- `device`
- `batch_size`

Các trường mới có giá trị mặc định an toàn khi khôi phục job cũ từ đĩa. Frontend cũng giữ fallback khi trường không tồn tại.

### Gemini settings và shutdown

- Settings API chỉ trả trạng thái `configured` và backend hiện tại; không bao giờ trả lại API key.
- POST settings ghi khóa vào file `.env` local theo cơ chế an toàn và cập nhật cấu hình runtime cần thiết.
- Shutdown API chỉ lập lịch dừng process tree của tool sau khi đã trả response cho trình duyệt.

## Trạng thái lỗi và an toàn

- Lỗi API được hiển thị bằng toast hoặc helper text gần control liên quan.
- Chỉ job đã kết thúc mới được xóa; job đang chạy trả `409`.
- Download và voice preview chỉ phục vụ file đã xác thực nằm trong thư mục thuộc quyền quản lý của tool.
- Dữ liệu do người dùng cung cấp như tên file, tên giọng và message phải được gán bằng `textContent` hoặc escape trước khi dùng trong HTML.
- Nút upload, clone voice và settings bị khóa trong lúc request đang chạy để tránh gửi lặp.

## Responsive và accessibility

- Không có cuộn ngang toàn trang tại viewport 360 px.
- Execution Graph có vùng cuộn ngang riêng và chỉ dẫn rõ ràng trên màn hình hẹp.
- Control tương tác có accessible name, focus state nhìn thấy được và dùng được bằng bàn phím.
- Nội dung tên file, voice ID và trạng thái dài phải wrap hoặc ellipsis trong giới hạn card.
- Dynamic status dùng live region phù hợp nhưng không tự chuyển focus.

## Kiểm thử

### Automated contracts

- Port các frontend contract test từ Mac sang Windows.
- Thêm regression test chứng minh HTML không còn marker của frontend cũ.
- Thêm API test cho xóa job, voice preview, Gemini settings và shutdown.
- Thêm telemetry tests cho snapshot mới, khôi phục job cũ và SSE terminal state.

### MCP Playwright

Chạy ứng dụng Windows local và xác minh:

- Dashboard mới được serve, asset mới được tải và console không có lỗi runtime.
- Queue thay đổi phần trăm trực tiếp từ SSE mà không cần F5.
- Selected Job không có progress bar trùng.
- Job `done` làm sáng node `Export` và hiện download actions.
- Voice Lab chọn/nghe thử/xóa giọng hoạt động; chữ không tràn ở 360 px.
- Upload, Gemini settings, cancel/delete job và shutdown control gọi đúng endpoint.
- Viewport 360 px và 1440 px không bị horizontal overflow ngoài vùng graph chủ đích.

## Ngoài phạm vi

- Không di chuyển migration OmniVoice-only từ Mac sang Windows.
- Không thay đổi thuật toán STT, dịch, TTS, subtitle, assembly hoặc mux.
- Không thêm framework frontend, bundler hoặc dependency JavaScript mới.
- Không thay đổi packaging Windows ngoài những gì cần để các endpoint mới được import và serve.

## Tiêu chí hoàn tất

- Frontend Windows không còn giao diện cũ và có cùng operations dashboard với bản Mac.
- Mọi control hiển thị đều có backend tương ứng hoặc fallback rõ ràng.
- Các regression test và test suite Windows đều pass.
- Kịch bản MCP Playwright nêu trên pass trên ứng dụng Windows local.
