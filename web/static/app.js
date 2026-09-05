/* Sub. Ops Console — realtime monitor client. */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const STAGES = ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"];
const STAGE_META = {
  extract: { label: "Tách âm thanh", detail: "Chuẩn bị audio từ video" },
  transcribe: { label: "Nhận diện giọng nói", detail: "Whisper tạo transcript" },
  translate: { label: "Dịch thuật", detail: "Gemini chuyển ngữ tiếng Việt" },
  synthesize: { label: "Tạo giọng đọc", detail: "Engine giọng đọc xử lý audio" },
  subtitle: { label: "Tạo phụ đề", detail: "Căn cue theo giọng đọc" },
  assemble: { label: "Ghép âm thanh", detail: "Đặt các segment vào timeline" },
  mux: { label: "Xuất video", detail: "Đóng gói video hoàn tất" },
};
const STAGE_LABELS = Object.fromEntries(Object.entries(STAGE_META).map(([stage, meta]) => [stage, `Đang ${meta.label.toLowerCase()}`]));
const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"]);
const STATUS_META = {
  queued: { label: "Đang chờ", tone: "queued" }, running: { label: "Đang chạy", tone: "running" },
  cancelling: { label: "Đang dừng", tone: "running" }, cancelled: { label: "Đã hủy", tone: "cancelled" },
  done: { label: "Hoàn tất", tone: "done" }, error: { label: "Lỗi", tone: "error" },
};
const MAX_BYTES = 8 * 1024 ** 3;
const SSE_MAX_RECONNECTS = 4;
const SSE_RECONNECT_BASE_MS = 900;
const JOB_POLL_INTERVAL_MS = 3000;
const PREVIEW_POLL_INTERVAL_MS = 2000;
const PREVIEW_POLL_ATTEMPTS = 60;
const state = { voices: [], jobs: [], selectedJobId: null, eventSources: new Map(), streamRetries: new Map(), reconnectTimers: new Map(), fallbackJobs: new Set(), jobsPollTimer: null, previewPollTimers: new Map(), lastEvents: [], connected: false, refreshing: false, uploadBusy: false, cloneBusy: false, voiceSource: "", previewToken: 0 };

function fmtBytes(bytes) { if (!Number.isFinite(Number(bytes))) return "—"; const value = Number(bytes); if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB`; if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`; if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`; return `${value} B`; }
function fmtDuration(seconds) { if (!Number.isFinite(Number(seconds))) return "—"; const total = Math.max(0, Math.round(Number(seconds))); const hours = Math.floor(total / 3600); const minutes = Math.floor((total % 3600) / 60); const secs = total % 60; if (hours) return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`; return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`; }
function fmtEta(seconds) { if (seconds == null || !Number.isFinite(Number(seconds))) return "ETA —"; return Number(seconds) <= 0 ? "Xong" : `ETA ${fmtDuration(seconds)}`; }
function fmtClock(timestamp) { if (!timestamp) return "—"; return new Date(Number(timestamp) * 1000).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
function relativeTime(timestamp) { if (!timestamp) return "—"; const delta = Math.max(0, Date.now() / 1000 - Number(timestamp)); if (delta < 60) return `${Math.round(delta)}s trước`; if (delta < 3600) return `${Math.round(delta / 60)}m trước`; return `${Math.round(delta / 3600)}h trước`; }
function voiceName(id) { return state.voices.find(voice => voice.id === id)?.display_name || id || "Chưa chọn giọng"; }
function voiceInitials(name) { return String(name || "?").trim().split(/\s+/).map(part => part[0]).join("").slice(0, 2).toUpperCase(); }
function stageLabel(stage) { return STAGE_META[stage]?.label || stage || "Đang chờ"; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }
function escapeAttr(value) { return escapeHtml(value); }
function announce(message) { const status = $("#srStatus"); if (status) status.textContent = message; }
function toast(message, tone = "") { const item = document.createElement("div"); item.className = `toast ${tone}`; item.textContent = message; $("#toastStack").append(item); window.setTimeout(() => item.remove(), 4200); }
function setConnection(connected, message = connected ? "Đã kết nối" : "Mất kết nối") { state.connected = connected; const pill = $("#connectionPill"); pill.classList.toggle("connected", connected); $("#connectionPill .status-dot").className = `status-dot ${connected ? "live" : "error"}`; $("#connectionText").textContent = message; }
async function apiJson(url, options = {}) { const response = await fetch(url, options); const body = await response.json().catch(() => ({})); if (!response.ok) throw new Error(body.detail || `Request thất bại (${response.status})`); return body; }

async function loadVoices() {
  try {
    state.voices = await apiJson("/api/voices"); state.voiceSource = "api"; const select = $("#voiceSelect"); select.textContent = "";
    state.voices.forEach(voice => { const option = document.createElement("option"); option.value = voice.id; option.textContent = voice.custom ? `${voice.display_name} · Nhân bản` : voice.display_name; select.append(option); });
    const saved = localStorage.getItem("sub.voice"); if (saved && state.voices.some(voice => voice.id === saved)) select.value = saved; if (!select.value && state.voices[0]) select.value = state.voices[0].id; renderVoiceLab(); updateUploadState();
  } catch (error) { state.voiceSource = "error"; $("#voiceSelect").innerHTML = "<option value=\"\">Không tải được danh sách giọng</option>"; $("#voiceList").innerHTML = '<div class="event-empty">Không tải được danh sách giọng.</div>'; toast(error.message, "error"); }
}

function previewUrl(voice) { return voice.preview_url || `/api/voices/${encodeURIComponent(voice.id)}/preview`; }
function renderVoiceLab() {
  const list = $("#voiceList"); list.textContent = "";
  const custom = state.voices.filter(voice => voice.custom); $("#voiceCount").textContent = `${custom.length} giọng`;
  if (!custom.length) { list.innerHTML = '<div class="event-empty">Chưa có giọng nhân bản. Tạo giọng đầu tiên ở nút bên phải.</div>'; return; }
  custom.forEach(voice => {
    const selected = $("#voiceSelect").value === voice.id; const item = document.createElement("div"); item.className = `voice-item${selected ? " selected" : ""}`;
    const avatar = document.createElement("span"); avatar.className = "voice-avatar"; avatar.textContent = voiceInitials(voice.display_name);
    const body = document.createElement("div"); body.className = "voice-item-main"; const name = document.createElement("strong"); name.textContent = voice.display_name; name.title = voice.display_name; const id = document.createElement("span"); id.textContent = voice.id; id.title = voice.id; const status = document.createElement("small"); const ready = voice.preview_status === "ready" && Boolean(voice.preview_url); status.className = ready ? "voice-ready" : "voice-missing"; status.textContent = ready ? "Demo sẵn sàng" : voice.preview_status === "generating" ? "Đang tạo demo…" : voice.preview_status === "failed" ? "Không tạo được demo" : "Chưa có demo"; body.append(name, id, status); item.append(avatar, body);
    const actions = document.createElement("div"); actions.className = "voice-actions";
    const select = document.createElement("button"); select.type = "button"; select.className = "voice-select-button"; select.textContent = selected ? "Đang chọn" : "Chọn"; select.setAttribute("aria-label", `Chọn giọng ${voice.display_name}`); select.addEventListener("click", () => { $("#voiceSelect").value = voice.id; $("#voiceSelect").dispatchEvent(new Event("change")); }); actions.append(select);
    const preview = document.createElement("button"); preview.type = "button"; preview.className = "voice-preview"; preview.innerHTML = '<span class="icon icon-play" aria-hidden="true"></span><span>Nghe thử</span>'; preview.setAttribute("aria-label", `Nghe thử giọng ${voice.display_name}`); preview.disabled = !ready; preview.addEventListener("click", () => togglePreview(voice, preview)); actions.append(preview);
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "voice-delete"; remove.textContent = "×"; remove.title = `Xóa ${voice.display_name}`; remove.setAttribute("aria-label", `Xóa giọng ${voice.display_name}`); remove.addEventListener("click", () => deleteVoice(voice)); actions.append(remove); item.append(actions); list.append(item);
  });
}
let activePreview = null;
function resetPreviewButton(button) { if (!button) return; button.classList.remove("playing"); button.innerHTML = '<span class="icon icon-play" aria-hidden="true"></span><span>Nghe thử</span>'; }
function togglePreview(voice, button) {
  if (activePreview?.button === button) { activePreview.audio.pause(); activePreview = null; resetPreviewButton(button); return; }
  if (activePreview) { activePreview.audio.pause(); resetPreviewButton(activePreview.button); }
  const token = ++state.previewToken;
  const audio = new Audio(previewUrl(voice)); audio.preload = "metadata";
  activePreview = { audio, button, token, voiceId: voice.id };
  button.classList.add("playing"); button.innerHTML = '<span class="icon icon-pause" aria-hidden="true"></span><span>Đang phát</span>';
  const fail = () => { if (activePreview?.token !== token) return; toast("Chưa có file demo cho giọng này.", "error"); activePreview = null; resetPreviewButton(button); };
  audio.addEventListener("error", fail); audio.addEventListener("ended", () => { if (activePreview?.token !== token) return; activePreview = null; resetPreviewButton(button); }); audio.play().catch(fail);
}
function stopPreviewPolling(voiceId) { const timer = state.previewPollTimers.get(voiceId); if (timer != null) window.clearTimeout(timer); state.previewPollTimers.delete(voiceId); }
async function pollVoicePreview(voiceId, attempt = 0) {
  stopPreviewPolling(voiceId);
  try {
    const result = await apiJson(`/api/voices/${encodeURIComponent(voiceId)}/preview-status`);
    if (result.status === "ready") { await loadVoices(); return; }
    if (result.status === "failed" || attempt + 1 >= PREVIEW_POLL_ATTEMPTS) { await loadVoices(); return; }
    const timer = window.setTimeout(() => pollVoicePreview(voiceId, attempt + 1), PREVIEW_POLL_INTERVAL_MS);
    state.previewPollTimers.set(voiceId, timer);
  } catch (_) {
    if (attempt + 1 >= PREVIEW_POLL_ATTEMPTS) return;
    const timer = window.setTimeout(() => pollVoicePreview(voiceId, attempt + 1), PREVIEW_POLL_INTERVAL_MS);
    state.previewPollTimers.set(voiceId, timer);
  }
}
function releaseActivePreview(voiceId) { if (activePreview?.voiceId !== voiceId) return; activePreview.audio.pause(); activePreview.audio.removeAttribute("src"); activePreview.audio.load(); resetPreviewButton(activePreview.button); activePreview = null; }
async function deleteVoice(voice) { if (!window.confirm(`Xóa giọng “${voice.display_name}”?`)) return; stopPreviewPolling(voice.id); releaseActivePreview(voice.id); try { await apiJson(`/api/voices/custom/${encodeURIComponent(voice.id)}`, { method: "DELETE" }); if ($("#voiceSelect").value === voice.id) $("#voiceSelect").value = ""; await loadVoices(); toast("Đã xóa giọng nhân bản.", "success"); } catch (error) { toast(error.message, "error"); } }
function setupVoiceLab() {
  const toggle = $("#toggleVoiceForm"); const form = $("#voiceForm"); toggle.addEventListener("click", () => { form.hidden = !form.hidden; toggle.setAttribute("aria-expanded", String(!form.hidden)); });
  form.addEventListener("submit", async event => { event.preventDefault(); if (state.cloneBusy) return; const name = $("#cloneName").value.trim(); const audio = $("#cloneAudio").files[0]; const hint = $("#voiceFormHint"); if (!name || !audio) { hint.textContent = "Nhập tên và chọn audio mẫu trước khi tạo."; return; } state.cloneBusy = true; $("#createVoiceButton").disabled = true; $("#createVoiceButton").textContent = "Đang tạo…"; hint.textContent = "Đang chuẩn hóa audio và đăng ký giọng local…"; try { const data = new FormData(); data.append("name", name); data.append("audio", audio); const voice = await apiJson("/api/voices/custom", { method: "POST", body: data }); await loadVoices(); $("#voiceSelect").value = voice.id; $("#voiceSelect").dispatchEvent(new Event("change")); form.reset(); form.hidden = true; toggle.setAttribute("aria-expanded", "false"); hint.textContent = "Nên dùng audio 3–8 giây, một người nói, ít tạp âm."; toast("Đã tạo giọng nhân bản.", "success"); pollVoicePreview(voice.id); } catch (error) { hint.textContent = error.message; toast(error.message, "error"); } finally { state.cloneBusy = false; $("#createVoiceButton").disabled = false; $("#createVoiceButton").textContent = "Tạo giọng"; } });
}

async function loadSystemHealth() { const stack = $("#healthStack"); stack.textContent = ""; try { const data = await apiJson("/api/model"); const models = Array.isArray(data.models) ? data.models : []; const rows = models.map(model => ({ label: model.key === "omnivoice" ? "OmniVoice" : model.key === "whisper" ? "Whisper STT" : model.label, status: model.ready ? "Sẵn sàng" : model.status === "downloading" ? "Đang tải" : "Chưa sẵn sàng", tone: model.ready ? "live" : model.status === "error" ? "error" : "warn" })); rows.push({ label: "API server", status: "Online", tone: "live" }); rows.forEach(row => { const item = document.createElement("div"); item.className = "health-row"; item.innerHTML = `<span class="status-dot ${row.tone}"></span><span></span><em></em>`; item.children[1].textContent = row.label; item.children[2].textContent = row.status; stack.append(item); }); } catch (_) { stack.innerHTML = '<div class="health-row"><span class="status-dot error"></span><span>API server</span><em>Offline</em></div>'; } }
function updateMetrics() { const running = state.jobs.filter(job => job.status === "running" || job.status === "cancelling"); const queued = state.jobs.filter(job => job.status === "queued"); const done = state.jobs.filter(job => job.status === "done"); const focus = running[0] || state.jobs.find(job => job.job_id === state.selectedJobId) || null; const runtimeEngine = focus?.engine || "Chưa có engine đang chạy"; const runtimeDevice = focus?.device || "—"; $("#metricRunning").textContent = running.length; $("#metricQueued").textContent = queued.length; $("#metricDone").textContent = done.length; $("#navActiveCount").textContent = running.length; $("#navJobCount").textContent = state.jobs.length; $("#metricRunningNote").textContent = running.length ? `${stageLabel(focus.stage)} · ${Math.round(focus.percent || 0)}%` : "Không có job hoạt động"; $("#metricDevice").textContent = runtimeDevice; $("#metricEngine").textContent = focus?.batch_size ? `${runtimeEngine} · batch ${focus.batch_size}` : runtimeEngine; $("#lastUpdated").textContent = `Cập nhật ${new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`; }
function jobCardMarkup(job) {
  const statusMeta = STATUS_META[job.status] || STATUS_META.queued;
  const percent = Math.round(job.percent || 0);
  const action = ACTIVE_STATUSES.has(job.status) ? `<button class="mini-button danger" type="button" data-action="cancel" data-job-id="${escapeAttr(job.job_id)}">${job.status === "cancelling" ? "Đang dừng…" : "Hủy"}</button>` : "";
  const remove = !ACTIVE_STATUSES.has(job.status) ? `<button class="mini-button danger" type="button" data-action="delete-job" data-job-id="${escapeAttr(job.job_id)}">Xóa</button>` : "";
  const downloads = job.status === "done" ? `<a class="mini-button" href="/api/jobs/${encodeURIComponent(job.job_id)}/download/video">Video</a><a class="mini-button" href="/api/jobs/${encodeURIComponent(job.job_id)}/download/srt">SRT</a>${remove}` : remove;
  return { statusMeta, percent, action, downloads, html: `<div class="job-card-top"><div><div class="job-name" title="${escapeAttr(job.filename)}">${escapeHtml(job.filename)}</div><div class="job-voice">${escapeHtml(voiceName(job.voice_id))}</div></div><span class="job-status ${statusMeta.tone}"><span class="status-dot ${statusMeta.tone === "running" ? "active" : statusMeta.tone === "done" ? "live" : statusMeta.tone === "error" ? "error" : "neutral"}"></span>${statusMeta.label}</span></div><div class="job-progress-row"><span>${escapeHtml(job.message || stageLabel(job.stage))}</span><strong>${percent}%</strong></div><div class="job-progress"><div class="progress-track"><span style="width:${Math.max(0, Math.min(100, Number(job.percent) || 0))}%"></span></div></div><div class="job-card-foot"><span>${escapeHtml(stageLabel(job.stage))}</span><span class="mono">${fmtDuration(job.elapsed_seconds)} · ${relativeTime(job.created_at)}</span></div>${(action || downloads) ? `<div class="job-card-actions">${action}${downloads}</div>` : ""}` };
}
function updateJobCard(card, job, {rebuild = false} = {}) {
  const { statusMeta, percent, html } = jobCardMarkup(job);
  if (rebuild || card.dataset.status !== job.status || card.dataset.jobStage !== job.stage) card.innerHTML = html;
  else { const progress = card.querySelector(".progress-track > span"); card.querySelector(".job-progress-row span").textContent = job.message || stageLabel(job.stage); card.querySelector(".job-progress-row strong").textContent = `${percent}%`; progress.style.width = `${Math.max(0, Math.min(100, Number(job.percent) || 0))}%`; card.querySelector(".job-card-foot > span").textContent = stageLabel(job.stage); }
  card.dataset.status = job.status; card.dataset.jobStage = job.stage || ""; card.className = `job-card ${job.job_id === state.selectedJobId ? "selected" : ""} ${ACTIVE_STATUSES.has(job.status) ? "is-active" : ""}`; card.setAttribute("aria-label", `${job.filename || "Video"}, ${statusMeta.label}, ${percent}%`);
}
function bindJobCard(card, job) { card.dataset.jobId = job.job_id; card.setAttribute("role", "button"); card.setAttribute("tabindex", "0"); card.addEventListener("click", event => { if (event.target.closest("button, a")) return; selectJob(job.job_id); }); card.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectJob(job.job_id); } }); }
function syncJobListViewport() {
  const list = $("#jobList");
  if (!list) return;
  const cards = $$(".job-card", list);
  if (!cards.length) { list.style.maxHeight = ""; return; }
  const visibleCards = cards.slice(0, 3);
  const gap = Number.parseFloat(getComputedStyle(list).rowGap) || 0;
  const height = visibleCards.reduce((total, card) => total + card.getBoundingClientRect().height, 0) + gap * Math.max(0, visibleCards.length - 1);
  list.style.maxHeight = `${Math.ceil(height)}px`;
}
function renderJobs() {
  const list = $("#jobList"); const existing = new Map($$(".job-card", list).map(card => [card.dataset.jobId, card])); const seen = new Set(); $("#jobsEmpty").hidden = state.jobs.length > 0;
  state.jobs.forEach(job => { let card = existing.get(job.job_id); if (!card) { card = document.createElement("article"); bindJobCard(card, job); list.append(card); } updateJobCard(card, job, {rebuild: !existing.has(job.job_id)}); seen.add(job.job_id); });
  existing.forEach((card, jobId) => { if (!seen.has(jobId)) card.remove(); }); syncJobListViewport(); updateMetrics();
}
async function loadGeminiSettings() {
  try {
    const data = await apiJson("/api/settings/gemini");
    $("#geminiBackend").value = data.backend || "developer";
    $("#geminiSettingsStatus").classList.toggle("configured", Boolean(data.configured));
    $("#geminiSettingsStatus").textContent = data.configured ? "Đã cấu hình" : "Chưa cấu hình";
  } catch (_) {}
}
function setupGeminiSettings() {
  const form = $("#geminiSettingsForm");
  if (!form) return;
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const apiKey = $("#geminiApiKey").value.trim();
    const hint = $("#geminiSettingsHint");
    if (!apiKey) { hint.textContent = "Hãy điền Gemini API key trước khi lưu."; return; }
    const button = form.querySelector("button[type=\"submit\"]");
    button.disabled = true;
    hint.textContent = "Đang lưu cấu hình local…";
    try {
      const data = await apiJson("/api/settings/gemini", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ api_key: apiKey, backend: $("#geminiBackend").value }) });
      form.reset();
      $("#geminiBackend").value = data.backend || "developer";
      $("#geminiSettingsStatus").classList.toggle("configured", Boolean(data.configured));
      $("#geminiSettingsStatus").textContent = data.configured ? "Đã cấu hình" : "Chưa cấu hình";
      hint.textContent = "Đã lưu vào .env local. Key không được hiển thị lại.";
      toast("Đã lưu cấu hình Gemini.", "success");
    } catch (error) { hint.textContent = error.message; toast(error.message, "error"); }
    finally { button.disabled = false; }
  });
  loadGeminiSettings();
}
async function deleteJob(job) {
  if (ACTIVE_STATUSES.has(job.status)) { toast("Không thể xóa video đang chạy.", "error"); return; }
  if (!window.confirm(`Xóa project “${job.filename || job.job_id}” và toàn bộ file kết quả?`)) return;
  try {
    await apiJson(`/api/jobs/${encodeURIComponent(job.job_id)}`, { method: "DELETE" });
    if (state.selectedJobId === job.job_id) {
      state.selectedJobId = null;
      localStorage.removeItem("sub.job");
    }
    await refreshJobs();
    renderSelectedJob();
    renderEvents(state.jobs.find(item => item.job_id === state.selectedJobId) || null);
    toast("Đã xóa project và file kết quả.", "success");
  } catch (error) { toast(error.message, "error"); }
}
function setupShutdown() {
  const button = $("#shutdownButton");
  if (!button) return;
  button.addEventListener("click", async () => {
    if (!window.confirm("Tắt ToolVietSub và dừng các tiến trình do tool khởi chạy?")) return;
    button.disabled = true;
    try { await apiJson("/api/shutdown", { method: "POST" }); button.textContent = "Đang tắt…"; }
    catch (error) { button.disabled = false; toast(error.message, "error"); }
  });
}
function renderSelectedJob() {
  const job = state.jobs.find(item => item.job_id === state.selectedJobId) || null;
  const empty = $("#selectedEmpty");
  const content = $("#selectedContent");
  if (!job) {
    empty.hidden = false;
    content.hidden = true;
    $("#selectedLiveLabel").innerHTML = '<span class="status-dot neutral"></span> IDLE';
    renderGraph(null);
    return;
  }
  empty.hidden = true;
  content.hidden = false;
  const meta = STATUS_META[job.status] || STATUS_META.queued;
  $("#selectedLiveLabel").innerHTML = `<span class="status-dot ${ACTIVE_STATUSES.has(job.status) ? "active" : meta.tone === "done" ? "live" : meta.tone === "error" ? "error" : "neutral"}"></span> ${meta.label.toUpperCase()}`;
  $("#selectedFilename").textContent = job.filename || "Không tên";
  $("#selectedVoice").textContent = `Giọng: ${voiceName(job.voice_id)}`;
  $("#selectedStage").textContent = stageLabel(job.stage);
  $("#selectedPercent").textContent = `${Math.round(job.percent || 0)}%`;
  $("#selectedMessage").textContent = job.message || STAGE_META[job.stage]?.detail || "—";
  $("#selectedEta").textContent = fmtEta(job.eta_seconds);
  $("#selectedElapsed").textContent = fmtDuration(job.elapsed_seconds);
  $("#selectedStageElapsed").textContent = fmtDuration(job.stage_elapsed_seconds);
  $("#selectedSegments").textContent = job.attempted_count ? `${job.spoken_count || 0}/${job.attempted_count}` : "—";
  $("#selectedEngine").textContent = job.engine || "—";
  $("#selectedDevice").textContent = job.device || "—";
  $("#selectedBatch").textContent = job.batch_size || "—";
  const actions = $("#selectedActions");
  actions.textContent = "";
  if (ACTIVE_STATUSES.has(job.status)) {
    const cancel = document.createElement("button");
    cancel.className = "button danger small";
    cancel.type = "button";
    cancel.textContent = job.status === "cancelling" ? "Đang dừng…" : "Hủy job";
    cancel.disabled = job.status === "cancelling";
    cancel.addEventListener("click", () => cancelJob(job.job_id));
    actions.append(cancel);
  }
  if (job.status === "done") {
    const video = document.createElement("a");
    video.className = "button primary small";
    video.href = `/api/jobs/${encodeURIComponent(job.job_id)}/download/video`;
    video.textContent = "Tải video";
    const srt = document.createElement("a");
    srt.className = "button small";
    srt.href = `/api/jobs/${encodeURIComponent(job.job_id)}/download/srt`;
    srt.textContent = "Tải SRT";
    actions.append(video, srt);
  }
  renderGraph(job);
}
function renderGraph(job) {
  const currentIndex = job ? STAGES.indexOf(job.stage) : -1;
  $$(".graph-node").forEach((node, index) => {
    const completed = Boolean(job) && (job.status === "done" ? index <= currentIndex : index < currentIndex);
    node.classList.toggle("completed", completed);
    node.classList.toggle("active", Boolean(job) && index === currentIndex && ACTIVE_STATUSES.has(job.status));
    node.classList.toggle("failed", Boolean(job) && index === currentIndex && job.status === "error");
    node.setAttribute("aria-current", Boolean(job) && index === currentIndex ? "step" : "false");
  });
  const title = $("#graphDetailTitle");
  const text = $("#graphDetailText");
  const timer = $("#graphDetailTime");
  if (!job) {
    title.textContent = "Chưa có job đang chạy";
    text.textContent = "Sơ đồ sẽ phát sáng theo từng bước khi pipeline hoạt động.";
    timer.textContent = "—";
    $("#graphTtsNote").textContent = "Theo cấu hình job";
    updateGraphFlow(null);
    return;
  }
  title.textContent = `${stageLabel(job.stage)} · ${Math.round(job.percent || 0)}%`;
  text.textContent = job.message || STAGE_META[job.stage]?.detail || "Đang xử lý";
  timer.textContent = fmtDuration(job.stage_elapsed_seconds);
  const runtimeEngine = job.engine || "TTS";
  const runtimeDevice = job.device || "—";
  const runtimeBatch = job.batch_size ? ` · batch ${job.batch_size}` : "";
  $("#graphTtsNote").textContent = `${runtimeEngine} · ${runtimeDevice}${runtimeBatch}`;
  updateGraphFlow(job);
}
function syncGraphScrollAffordance() {
  const shell = $(".graph-scroll-shell");
  if (!shell) return;
  const frame = shell.closest(".graph-scroll-frame");
  if (!frame) return;
  const scrollable = shell.scrollWidth > shell.clientWidth + 1;
  frame.classList.toggle("is-scrollable", scrollable);
  frame.classList.toggle("is-at-end", scrollable && shell.scrollLeft + shell.clientWidth >= shell.scrollWidth - 1);
}
function updateGraphFlow(job) {
  const currentIndex = job ? STAGES.indexOf(job.stage) : -1;
  const flowIndex = currentIndex - 1;
  $$(".graph-link").forEach((link, index) => {
    const complete = Boolean(job) && (job.status === "done" ? index <= flowIndex : index < flowIndex);
    const active = Boolean(job) && index === flowIndex && ACTIVE_STATUSES.has(job.status);
    link.classList.toggle("flow-complete", complete);
    link.classList.toggle("flow-active", active);
    link.setAttribute("aria-label", active ? `Đang truyền từ ${stageLabel(STAGES[index])} sang ${stageLabel(STAGES[index + 1])}` : complete ? "Đã truyền xong" : "Đang chờ");
  });
  syncGraphScrollAffordance();
}
function renderEvents(job) { const list = $("#eventList"); list.textContent = ""; const events = (job?.events || state.lastEvents || []).slice(-36).reverse(); if (!events.length) { list.innerHTML = '<div class="event-empty">Chưa có sự kiện mới.</div>'; return; } events.forEach(event => { const row = document.createElement("div"); const tone = event.kind === "finished" && event.status === "error" ? "error" : event.kind === "finished" || event.kind === "status" && event.status === "done" ? "complete" : ""; row.className = `event-row ${tone}`; const heading = event.kind === "started" ? "Bắt đầu job" : event.kind === "finished" ? `Job ${STATUS_META[event.status]?.label?.toLowerCase() || event.status}` : event.kind === "status" ? (event.message || event.status) : stageLabel(event.stage); row.innerHTML = `<time class="event-time"></time><span class="event-marker"></span><div class="event-copy"><strong></strong><span></span></div>`; row.querySelector(".event-time").textContent = fmtClock(event.at); row.querySelector("strong").textContent = heading; row.querySelector(".event-copy span").textContent = event.message || (event.percent != null ? `${event.percent}%` : ""); list.append(row); }); }
function selectJob(jobId) { state.selectedJobId = jobId; localStorage.setItem("sub.job", jobId); renderJobs(); renderSelectedJob(); const job = state.jobs.find(item => item.job_id === jobId); renderEvents(job); updateGraphFlow(job); ensureJobStream(jobId); }
async function refreshJobs() { if (state.refreshing) return; state.refreshing = true; try { const data = await apiJson("/api/jobs"); state.jobs = Array.isArray(data.jobs) ? data.jobs : []; setConnection(true); const saved = localStorage.getItem("sub.job"); const selectedExists = state.jobs.some(job => job.job_id === state.selectedJobId); if (!selectedExists) state.selectedJobId = state.jobs.some(job => job.job_id === saved) ? saved : state.jobs[0]?.job_id || null; renderJobs(); const selected = state.jobs.find(job => job.job_id === state.selectedJobId); renderSelectedJob(); renderEvents(selected); updateGraphFlow(selected); [...state.fallbackJobs].forEach(jobId => { const job = state.jobs.find(item => item.job_id === jobId); if (!job || !ACTIVE_STATUSES.has(job.status)) clearJobRecovery(jobId); }); state.jobs.filter(job => ACTIVE_STATUSES.has(job.status)).forEach(job => ensureJobStream(job.job_id)); } catch (_) { setConnection(false, "API offline"); } finally { state.refreshing = false; } }
function syncJobsPolling() { if (state.fallbackJobs.size && state.jobsPollTimer == null) state.jobsPollTimer = window.setInterval(refreshJobs, JOB_POLL_INTERVAL_MS); else if (!state.fallbackJobs.size && state.jobsPollTimer != null) { window.clearInterval(state.jobsPollTimer); state.jobsPollTimer = null; } }
function clearJobRecovery(jobId) { const timer = state.reconnectTimers.get(jobId); if (timer != null) window.clearTimeout(timer); state.reconnectTimers.delete(jobId); state.streamRetries.delete(jobId); state.fallbackJobs.delete(jobId); syncJobsPolling(); }
function ensureJobStream(jobId) {
  if (state.eventSources.has(jobId) || state.reconnectTimers.has(jobId) || state.streamRetries.get(jobId) > SSE_MAX_RECONNECTS) return;
  const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
  state.eventSources.set(jobId, source);
  source.onopen = () => { if (state.eventSources.get(jobId) !== source) return; state.streamRetries.delete(jobId); state.fallbackJobs.delete(jobId); syncJobsPolling(); setConnection(true); };
  source.onmessage = event => { try { if (state.eventSources.get(jobId) !== source) return; const snapshot = JSON.parse(event.data); state.streamRetries.delete(jobId); state.fallbackJobs.delete(jobId); syncJobsPolling(); const index = state.jobs.findIndex(job => job.job_id === snapshot.job_id); if (index >= 0) state.jobs[index] = { ...state.jobs[index], ...snapshot }; else state.jobs.unshift(snapshot); if (snapshot.job_id === state.selectedJobId) { renderSelectedJob(); renderEvents(snapshot); updateGraphFlow(snapshot); } renderJobs(); if (!ACTIVE_STATUSES.has(snapshot.status)) { source.close(); state.eventSources.delete(jobId); clearJobRecovery(jobId); if (snapshot.status === "done") toast("Video đã xử lý xong.", "success"); if (snapshot.status === "error") toast(snapshot.message || "Job gặp lỗi.", "error"); } } catch (_) {} };
  source.onerror = () => { if (state.eventSources.get(jobId) !== source) return; source.close(); state.eventSources.delete(jobId); state.fallbackJobs.add(jobId); syncJobsPolling(); const attempt = (state.streamRetries.get(jobId) || 0) + 1; state.streamRetries.set(jobId, attempt); if (attempt > SSE_MAX_RECONNECTS) return; const delay = SSE_RECONNECT_BASE_MS * 2 ** (attempt - 1); const timer = window.setTimeout(() => { state.reconnectTimers.delete(jobId); const current = state.jobs.find(job => job.job_id === jobId); if (current && ACTIVE_STATUSES.has(current.status)) ensureJobStream(jobId); else clearJobRecovery(jobId); }, delay); state.reconnectTimers.set(jobId, timer); };
}
async function cancelJob(jobId) { try { await apiJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }); toast("Đã gửi yêu cầu dừng job."); } catch (error) { toast(error.message, "error"); } }
let uploadRequest = null;
function setupUpload() {
  const input = $("#videoInput"); const dropzone = $("#dropzone");
  input.addEventListener("change", () => setUploadFile(input.files[0] || null));
  ["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove("drag"); }));
  dropzone.addEventListener("drop", event => setUploadFile([...event.dataTransfer.files].find(file => isVideoFile(file)) || null));
  dropzone.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
  $("#voiceSelect").addEventListener("change", event => { if (event.target.value) localStorage.setItem("sub.voice", event.target.value); updateUploadState(); renderVoiceLab(); });
  $("#uploadForm").addEventListener("submit", submitUpload);
  $("#cancelUploadButton").addEventListener("click", () => { if (uploadRequest) uploadRequest.abort(); });
  $("[data-action=\"open-upload\"]").addEventListener("click", () => { $("#uploadPanel").scrollIntoView({ behavior: "smooth", block: "center" }); });
  $("#newJobButton").addEventListener("click", () => { $("#uploadPanel").scrollIntoView({ behavior: "smooth", block: "center" }); });
}
function isVideoFile(file) { return Boolean(file && (file.type.startsWith("video/") || /\.(mp4|mkv|mov|avi|webm)$/i.test(file.name))); }
function showUploadError(message, focus = false) { const box = $("#uploadError"); $("#uploadErrorText").textContent = message; box.hidden = false; if (focus) box.focus(); }
function clearUploadError() { $("#uploadError").hidden = true; $("#uploadErrorText").textContent = ""; }
function setUploadFile(file) {
  clearUploadError();
  state.selectedFile = isVideoFile(file) ? file : null;
  const selected = $("#selectedUpload");
  if (file && !state.selectedFile) showUploadError("File này không phải định dạng video được hỗ trợ.", true);
  if (file && file.size > MAX_BYTES) { state.selectedFile = null; showUploadError("Video vượt quá giới hạn 8 GB.", true); }
  $("#dropTitle").textContent = state.selectedFile ? "Video đã sẵn sàng" : "Kéo thả video vào đây";
  $("#dropHint").textContent = state.selectedFile ? `${fmtBytes(state.selectedFile.size)} · sẵn sàng tải lên` : "hoặc bấm để chọn · MP4, MKV, MOV, AVI, WebM · tối đa 8 GB";
  selected.hidden = !state.selectedFile; selected.textContent = state.selectedFile?.name || ""; updateUploadState();
}
function updateUploadState() {
  const ready = Boolean(state.selectedFile && $("#voiceSelect").value && !state.uploadBusy);
  const button = $("#startButton");
  button.disabled = !ready;
  button.setAttribute("aria-disabled", String(!ready));
  const hint = state.uploadBusy ? "Đang tải video lên server local…" : ready ? "Sẵn sàng — pipeline sẽ xuất hiện ở hàng đợi." : "Chọn video và giọng đọc để bắt đầu.";
  $("#uploadHint").textContent = hint;
  announce(hint);
}
function submitUpload(event) {
  event.preventDefault();
  clearUploadError();
  const voiceId = $("#voiceSelect").value;
  if (!state.selectedFile || !voiceId || state.uploadBusy) {
    showUploadError(!state.selectedFile ? "Hãy chọn một video trước." : "Hãy chọn giọng đọc trước.", true);
    return;
  }
  state.uploadBusy = true;
  updateUploadState();
  $("#uploadProgress").hidden = false;
  $("#uploadBar").style.width = "0%";
  $("#uploadPercent").textContent = "0%";
  $("#uploadProgressLabel").textContent = "Đang tải video lên";

  const xhr = new XMLHttpRequest();
  uploadRequest = xhr;
  xhr.open("POST", "/api/jobs");
  xhr.timeout = 120000;
  const form = new FormData();
  form.append("video", state.selectedFile);
  form.append("filename", state.selectedFile.name);
  form.append("voice_id", voiceId);
  xhr.upload.onprogress = event => {
    if (!event.lengthComputable) return;
    const pct = event.loaded / event.total * 100;
    $("#uploadBar").style.width = `${pct}%`;
    $("#uploadPercent").textContent = `${Math.round(pct)}%`;
  };
  const reset = () => {
    if (uploadRequest === xhr) uploadRequest = null;
    state.uploadBusy = false;
    $("#uploadProgress").hidden = true;
    updateUploadState();
  };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {}
      reset();
      state.selectedFile = null;
      $("#videoInput").value = "";
      setUploadFile(null);
      refreshJobs();
      if (body.job_id) selectJob(body.job_id);
      toast("Đã thêm video vào hàng đợi.", "success");
      return;
    }
    let message = "Không tải được video.";
    try { message = JSON.parse(xhr.responseText).detail || message; } catch (_) {}
    reset();
    showUploadError(message, true);
    toast(message, "error");
  };
  xhr.onerror = () => { reset(); showUploadError("Mất kết nối khi tải video. Kiểm tra server rồi thử lại.", true); toast("Mất kết nối khi tải video.", "error"); };
  xhr.ontimeout = () => { reset(); showUploadError("Tải video quá thời gian chờ. Hãy kiểm tra server và thử lại.", true); toast("Tải video quá thời gian chờ.", "error"); };
  xhr.onabort = () => { reset(); showUploadError("Đã hủy tải video. Bạn có thể chọn lại file để thử lại.", true); toast("Đã hủy tải video.", "error"); };
  xhr.send(form);
}

$("#refreshButton").addEventListener("click", () => { loadSystemHealth(); loadVoices(); refreshJobs(); loadGeminiSettings(); });
$("#clearEvents").addEventListener("click", () => { state.lastEvents = []; renderEvents(null); });
document.addEventListener("click", event => { const button = event.target.closest("[data-action=\"cancel\"]"); if (button) { event.stopPropagation(); cancelJob(button.dataset.jobId); return; } const remove = event.target.closest("[data-action=\"delete-job\"]"); if (remove) { event.stopPropagation(); const job = state.jobs.find(item => item.job_id === remove.dataset.jobId); if (job) deleteJob(job); } });
(async function init() { setupUpload(); setupVoiceLab(); setupGeminiSettings(); setupShutdown(); const graphScrollShell = $(".graph-scroll-shell"); window.addEventListener("resize", syncGraphScrollAffordance); window.addEventListener("resize", syncJobListViewport); graphScrollShell?.addEventListener("scroll", syncGraphScrollAffordance, { passive: true }); await Promise.all([loadVoices(), loadSystemHealth(), refreshJobs()]); syncGraphScrollAffordance(); window.setInterval(() => { const job = state.jobs.find(item => item.job_id === state.selectedJobId); if (job && ACTIVE_STATUSES.has(job.status)) { renderSelectedJob(); updateMetrics(); } }, 1000); })();
