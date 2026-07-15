/* ============================================================
   Sub. — app.js
   HTML/CSS/JS thuần, không build step.
   Có chế độ xem thử (mock) khi không có server.
   ============================================================ */
"use strict";

/* ---------- tiện ích ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const STAGES = ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"];
const STAGE_LABELS = {
  extract: "Đang trích xuất âm thanh",
  transcribe: "Đang nhận diện giọng nói",
  translate: "Đang dịch sang tiếng Việt",
  synthesize: "Đang tạo giọng đọc",
  subtitle: "Đang tạo phụ đề",
  assemble: "Đang ghép âm thanh",
  mux: "Đang ghép video",
};
const ACTIVE_STATUSES = new Set(["queued", "running", "cancelling"]);
const MAX_BYTES = 8 * 1024 * 1024 * 1024; // 8 GB

function fmtBytes(b) {
  if (b == null || isNaN(b)) return "";
  if (b >= 1024 ** 3) return (b / 1024 ** 3).toFixed(1) + " GB";
  if (b >= 1024 ** 2) return (b / 1024 ** 2).toFixed(1) + " MB";
  if (b >= 1024) return (b / 1024).toFixed(1) + " KB";
  return b + " B";
}
function fmtMB(mb) { return (mb >= 1024 ? (mb / 1024).toFixed(2) + " GB" : mb.toFixed(1) + " MB"); }
function fmtDur(sec) {
  if (!isFinite(sec)) return "";
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? `${m} phút ${s} giây` : `${s} giây`;
}

/* ---------- trạng thái toàn cục ---------- */
const state = {
  mock: false,
  file: null,
  fileDuration: null,
  voices: [],
  selectedVoice: localStorage.getItem("sub.voice") || null,
  cloningEnabled: false,
  models: { required: false, ready: true, models: [] },
  jobs: [],
  uploading: false,
  pollTimer: null,
  modelSSE: null,
};

/* ---------- MOCK: chế độ xem thử ---------- */
const mock = {
  voices: [
    { id: "truc-ly", display_name: "Trúc Ly", preview_url: "", custom: false },
    { id: "minh-anh", display_name: "Minh Anh", preview_url: "", custom: false },
    { id: "quoc-bao", display_name: "Quốc Bảo", preview_url: "", custom: false },
    { id: "ha-vy", display_name: "Hà Vy", preview_url: "", custom: false },
    { id: "tuan-kiet", display_name: "Tuấn Kiệt", preview_url: "", custom: false },
    { id: "thao-nhi", display_name: "Thảo Nhi", preview_url: "", custom: false },
    { id: "gia-han", display_name: "Gia Hân", preview_url: "", custom: false },
    { id: "duc-thinh", display_name: "Đức Thịnh", preview_url: "", custom: false },
    { id: "custom-1", display_name: "Giọng của tôi", preview_url: "", custom: true },
  ],
  models: {
    required: true, ready: false,
    models: [
      { key: "whisper", label: "Whisper — nhận diện giọng nói", ready: true, status: "ready", message: "", percent: 100, downloaded_mb: 1550, total_mb: 1550 },
      { key: "vieneu", label: "VieNeu — giọng đọc tiếng Việt", ready: false, status: "idle", message: "", percent: 0, downloaded_mb: 0, total_mb: 486.2 },
    ],
  },
  jobs: [
    { job_id: "mock-run", status: "running", stage: "synthesize", percent: 61.5, message: "Đang đọc lượt thoại 12/32", filename: "phong-van-elon.mp4", voice_id: "truc-ly", created_at: Date.now() / 1000 - 300, warnings: [], attempted_count: 32, spoken_count: 12, silent_ratio: 0, degraded: false },
    { job_id: "mock-queue", status: "queued", stage: "extract", percent: 0, message: "Chờ đến lượt", filename: "tap-3-du-lich-nhat.mkv", voice_id: "minh-anh", created_at: Date.now() / 1000 - 120, warnings: [], attempted_count: 0, spoken_count: 0, silent_ratio: 0, degraded: false },
    { job_id: "mock-done", status: "done", stage: "mux", percent: 100, message: "Hoàn tất", filename: "bai-giang-vat-ly.mp4", voice_id: "quoc-bao", created_at: Date.now() / 1000 - 4000, warnings: [], attempted_count: 45, spoken_count: 45, silent_ratio: 0.01, degraded: false },
    { job_id: "mock-degraded", status: "done", stage: "mux", percent: 100, message: "Hoàn tất (có cảnh báo)", filename: "hoi-thao-marketing.mp4", voice_id: "ha-vy", created_at: Date.now() / 1000 - 8000, warnings: ["12/40 lượt thoại không tạo được giọng đọc và bị thay bằng khoảng lặng.", "Tỷ lệ khoảng lặng chiếm 28% thời lượng lời thoại."], attempted_count: 40, spoken_count: 28, silent_ratio: 0.28, degraded: true },
    { job_id: "mock-error", status: "error", stage: "translate", percent: 34, message: "Không kết nối được dịch vụ dịch thuật. Kiểm tra API key rồi thử lại.", filename: "vlog-cuoi-tuan.mp4", voice_id: "truc-ly", created_at: Date.now() / 1000 - 9000, warnings: [], attempted_count: 0, spoken_count: 0, silent_ratio: 0, degraded: false },
  ],
  timer: null,
  /* mô phỏng job chạy để xem đủ trạng thái */
  tick() {
    const j = this.jobs.find(x => x.job_id === "mock-run");
    if (j && j.status === "running") {
      j.percent = Math.min(100, j.percent + 2.5);
      const si = STAGES.indexOf(j.stage);
      const spoken = Math.min(j.attempted_count, Math.round(j.attempted_count * j.percent / 100));
      j.spoken_count = spoken;
      j.message = j.stage === "synthesize" ? `Đang đọc lượt thoại ${spoken}/${j.attempted_count}` : STAGE_LABELS[j.stage];
      if (j.percent >= 100) {
        if (si < STAGES.length - 1) { j.stage = STAGES[si + 1]; j.percent = 5; }
        else { j.status = "done"; j.message = "Hoàn tất"; j.spoken_count = j.attempted_count; }
      }
    }
    const q = this.jobs.find(x => x.job_id === "mock-queue");
    if (q && q.status === "queued" && !this.jobs.some(x => x.status === "running" && x !== q)) {
      q.status = "running"; q.stage = "extract"; q.percent = 3; q.message = STAGE_LABELS.extract;
      q.attempted_count = 21;
    }
  },
  modelTimer: null,
};

/* ---------- API (thật hoặc mock) ---------- */
const api = {
  async voices() {
    if (state.mock) return structuredClone(mock.voices);
    const r = await fetch("/api/voices");
    if (!r.ok) throw new Error("voices");
    return r.json();
  },
  async cloning() {
    if (state.mock) return { enabled: true };
    const r = await fetch("/api/voices/cloning");
    return r.ok ? r.json() : { enabled: false };
  },
  async createVoice(name, audio) {
    if (state.mock) {
      await new Promise(res => setTimeout(res, 900));
      if (audio.size < 2000) throw new Error("Mẫu âm thanh quá ngắn. Cần đoạn 3–8 giây.");
      const v = { id: "custom-" + Date.now(), display_name: name, preview_url: "", custom: true };
      mock.voices.push(v);
      return v;
    }
    const fd = new FormData();
    fd.append("name", name); fd.append("audio", audio);
    const r = await fetch("/api/voices/custom", { method: "POST", body: fd });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || "Không tạo được giọng.");
    return body;
  },
  async deleteVoice(id) {
    if (state.mock) {
      mock.voices = mock.voices.filter(v => v.id !== id);
      return { deleted: id };
    }
    const r = await fetch(`/api/voices/custom/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (!r.ok) throw new Error("Không xóa được giọng.");
    return r.json();
  },
  async model() {
    if (state.mock) return structuredClone(mock.models);
    const r = await fetch("/api/model");
    if (!r.ok) throw new Error("model");
    return r.json();
  },
  async modelDownload(key) {
    if (state.mock) {
      const m = mock.models.models.find(x => x.key === key);
      if (m) { m.status = "downloading"; m.percent = 0; m.downloaded_mb = 0; }
      clearInterval(mock.modelTimer);
      mock.modelTimer = setInterval(() => {
        const mm = mock.models.models.find(x => x.status === "downloading");
        if (!mm) { clearInterval(mock.modelTimer); return; }
        mm.percent = Math.min(100, mm.percent + 7);
        mm.downloaded_mb = mm.total_mb * mm.percent / 100;
        if (mm.percent >= 100) { mm.status = "ready"; mm.ready = true; }
        mock.models.ready = mock.models.models.every(x => x.ready);
        renderModels(mock.models);
        updateStartState();
      }, 450);
      return { started: true, key };
    }
    const r = await fetch(`/api/model/download?key=${encodeURIComponent(key)}`, { method: "POST" });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || "Không bắt đầu tải được model.");
    }
    return r.json();
  },
  async jobs() {
    if (state.mock) { mock.tick(); return { jobs: structuredClone(mock.jobs).sort((a, b) => b.created_at - a.created_at) }; }
    const r = await fetch("/api/jobs");
    if (!r.ok) throw new Error("jobs");
    return r.json();
  },
  async cancel(id) {
    if (state.mock) {
      const j = mock.jobs.find(x => x.job_id === id);
      if (!j) return;
      if (j.status === "queued") { j.status = "cancelled"; j.message = "Đã hủy"; }
      else if (j.status === "running") {
        j.status = "cancelling"; j.message = "Đang dừng…";
        setTimeout(() => { j.status = "cancelled"; j.message = "Đã hủy"; renderJobs(); }, 2600);
      }
      return j;
    }
    const r = await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
    if (!r.ok && r.status !== 409) throw new Error("cancel");
    return r.json().catch(() => null);
  },
};

/* ---------- Thêm video ---------- */
const dropzone = $("#dropzone"), fileInput = $("#fileInput");
const dzIdle = $("#dzIdle"), dzFile = $("#dzFile"), dzName = $("#dzName"), dzMeta = $("#dzMeta");
const startBtn = $("#startBtn"), startHint = $("#startHint");
const upWrap = $("#uploadProgress"), upBar = $("#upBar"), upPct = $("#upPct"), upBytes = $("#upBytes"), upLabel = $("#upLabel");

function setFile(f) {
  if (f && f.size > MAX_BYTES) {
    startHint.textContent = "Video vượt quá 8 GB — hãy chọn file nhỏ hơn.";
    return;
  }
  state.file = f || null;
  state.fileDuration = null;
  dzIdle.hidden = !!f;
  dzFile.hidden = !f;
  dropzone.classList.toggle("hasfile", !!f);
  if (f) {
    dzName.textContent = f.name;
    dzMeta.textContent = fmtBytes(f.size);
    const url = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.onloadedmetadata = () => {
      state.fileDuration = v.duration;
      dzMeta.textContent = `${fmtBytes(f.size)} · ${fmtDur(v.duration)}`;
      URL.revokeObjectURL(url);
    };
    v.onerror = () => URL.revokeObjectURL(url);
    v.src = url;
  }
  updateStartState();
}

dropzone.addEventListener("click", e => {
  if (e.target.closest("#dzRemove")) return;
  if (!state.file) fileInput.click();
});
dropzone.addEventListener("keydown", e => {
  if ((e.key === "Enter" || e.key === " ") && !state.file) { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("#dzRemove").addEventListener("click", () => { fileInput.value = ""; setFile(null); });
["dragenter", "dragover"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", e => {
  const f = [...e.dataTransfer.files].find(x => x.type.startsWith("video/") || /\.(mp4|mkv|mov|avi|webm)$/i.test(x.name));
  if (f) setFile(f);
});

function updateStartState() {
  const missing = [];
  if (!state.file) missing.push("chọn video");
  if (!state.selectedVoice) missing.push("chọn giọng đọc");
  if (state.models.required && !state.models.ready) missing.push("tải model còn thiếu");
  const ok = missing.length === 0 && !state.uploading;
  startBtn.disabled = !ok;
  startHint.textContent = state.uploading ? "Đang tải video lên…" :
    ok ? "Sẵn sàng — bấm để bắt đầu." : "Còn thiếu: " + missing.join(", ") + ".";
}

/* upload bằng XMLHttpRequest để có onprogress */
startBtn.addEventListener("click", () => {
  if (!state.file || !state.selectedVoice || state.uploading) return;
  state.uploading = true;
  updateStartState();
  upWrap.hidden = false;
  upLabel.textContent = "Đang tải lên…";
  upBar.style.width = "0%"; upPct.textContent = "0%"; upBytes.textContent = "";

  const done = ok => {
    state.uploading = false;
    upWrap.hidden = true;
    if (ok) { fileInput.value = ""; setFile(null); refreshJobs(true); }
    updateStartState();
  };

  if (state.mock) {
    /* mô phỏng upload */
    let pct = 0;
    const total = state.file.size;
    const t = setInterval(() => {
      pct = Math.min(100, pct + 9);
      upBar.style.width = pct + "%";
      upPct.textContent = Math.round(pct) + "%";
      upBytes.textContent = `${fmtBytes(total * pct / 100)} / ${fmtBytes(total)}`;
      if (pct >= 100) {
        clearInterval(t);
        upLabel.textContent = "Đang kiểm tra video…";
        upPct.textContent = "";
        setTimeout(() => {
          mock.jobs.unshift({
            job_id: "mock-" + Date.now(), status: "queued", stage: "extract", percent: 0,
            message: "Chờ đến lượt", filename: state.file.name, voice_id: state.selectedVoice,
            created_at: Date.now() / 1000, warnings: [], attempted_count: 0, spoken_count: 0, silent_ratio: 0, degraded: false,
          });
          done(true);
        }, 900);
      }
    }, 160);
    return;
  }

  const fd = new FormData();
  fd.append("video", state.file);
  fd.append("voice_id", state.selectedVoice);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable) return;
    const pct = e.loaded / e.total * 100;
    upBar.style.width = pct + "%";
    upPct.textContent = Math.round(pct) + "%";
    upBytes.textContent = `${fmtBytes(e.loaded)} / ${fmtBytes(e.total)}`;
    if (pct >= 100) { upLabel.textContent = "Đang kiểm tra video…"; upPct.textContent = ""; }
  };
  xhr.onload = () => {
    if (xhr.status >= 200 && xhr.status < 300) { done(true); return; }
    let msg = "Không gửi được video.";
    try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
    if (xhr.status === 413) msg = "Video vượt quá 8 GB.";
    startHint.textContent = msg;
    state.uploading = false; upWrap.hidden = true; updateStartState();
  };
  xhr.onerror = () => { startHint.textContent = "Lỗi mạng khi tải video lên."; state.uploading = false; upWrap.hidden = true; updateStartState(); };
  xhr.send(fd);
});

/* ---------- Model ---------- */
const secModels = $("#sec-models"), modelGrid = $("#modelGrid");

function renderModels(data) {
  state.models = data;
  if (!data.required || !data.models.length) { secModels.hidden = true; updateStartState(); return; }
  secModels.hidden = false;
  modelGrid.textContent = "";
  data.models.forEach(m => {
    const stateName = m.status === "ready" ? "ready" : m.status === "downloading" ? "downloading" : m.status === "error" ? "error" : "missing";
    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.modelKey = m.key;
    card.dataset.modelState = stateName;

    const top = document.createElement("div"); top.className = "mc-top";
    const name = document.createElement("div"); name.className = "mc-name"; name.textContent = m.label;
    const st = document.createElement("span"); st.className = "mc-state";
    st.textContent = stateName === "ready" ? "Sẵn sàng" : stateName === "downloading" ? "Đang tải" : stateName === "error" ? "Lỗi" : "Chưa tải";
    top.append(name, st);
    card.append(top);

    const size = document.createElement("div"); size.className = "mc-size";
    size.textContent = m.total_mb ? fmtMB(m.total_mb) : "";
    card.append(size);

    if (m.message) {
      const msg = document.createElement("div"); msg.className = "mc-msg"; msg.textContent = m.message;
      card.append(msg);
    }
    if (stateName === "downloading") {
      const prog = document.createElement("div"); prog.className = "mc-progress";
      const bar = document.createElement("div"); bar.className = "bar";
      const fill = document.createElement("div"); fill.className = "bar-fill";
      fill.style.width = (m.percent || 0) + "%";
      bar.append(fill);
      const bytes = document.createElement("div"); bytes.className = "mc-bytes";
      const l = document.createElement("span"); l.textContent = `${(m.downloaded_mb || 0).toFixed(1)} / ${(m.total_mb || 0).toFixed(1)} MB`;
      const r = document.createElement("span"); r.textContent = Math.round(m.percent || 0) + "%";
      bytes.append(l, r);
      prog.append(bar, bytes);
      card.append(prog);
    }
    if (stateName === "missing" || stateName === "error") {
      const btn = document.createElement("button");
      btn.className = "mc-btn"; btn.dataset.modelAction = "download";
      btn.textContent = stateName === "error" ? "Thử tải lại" : "Tải model";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try { await api.modelDownload(m.key); if (!state.mock) openModelSSE(); }
        catch (err) { btn.disabled = false; alert(err.message); }
      });
      card.append(btn);
    }
    modelGrid.append(card);
  });
  updateStartState();
}

function openModelSSE() {
  if (state.mock || state.modelSSE) return;
  const es = new EventSource("/api/model/events");
  state.modelSSE = es;
  es.onmessage = e => {
    try {
      const data = JSON.parse(e.data);
      state.models.models = data.models;
      state.models.ready = data.models.every(m => m.ready);
      renderModels(state.models);
    } catch (_) {}
  };
  es.onerror = () => { es.close(); state.modelSSE = null; refreshModels(); };
}

async function refreshModels() {
  try { renderModels(await api.model()); }
  catch (_) { secModels.hidden = true; }
}

/* ---------- Giọng đọc ---------- */
const voiceGrid = $("#voiceGrid");
let currentAudio = null, currentPlayBtn = null;

function stopPreview() {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if (currentPlayBtn) { currentPlayBtn.classList.remove("playing"); currentPlayBtn.textContent = "▶ Nghe thử"; currentPlayBtn = null; }
}

function voiceHue(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) % 360;
  return h;
}
function voiceInitials(name) {
  const parts = name.trim().split(/\s+/);
  return (parts.length > 1 ? parts[0][0] + parts[parts.length - 1][0] : name.slice(0, 2)).toUpperCase();
}

function renderVoices() {
  voiceGrid.textContent = "";
  state.voices.forEach(v => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "voice-card" + (state.selectedVoice === v.id ? " sel" : "");
    card.dataset.voiceId = v.id;

    const ava = document.createElement("div");
    ava.className = "vc-ava";
    const h = voiceHue(v.id);
    ava.style.background = `linear-gradient(145deg, oklch(0.62 0.16 ${h}), oklch(0.42 0.14 ${(h + 40) % 360}))`;
    ava.textContent = voiceInitials(v.display_name);
    card.append(ava);

    const body = document.createElement("div");
    body.className = "vc-body";
    const name = document.createElement("div"); name.className = "vc-name"; name.textContent = v.display_name;
    body.append(name);
    if (v.custom) {
      const tag = document.createElement("span"); tag.className = "vc-tag"; tag.textContent = "Nhân bản";
      body.append(tag);
    }
    const row = document.createElement("div"); row.className = "vc-row";
    if (v.preview_url) {
      const play = document.createElement("span");
      play.className = "vc-play"; play.textContent = "▶ Nghe thử"; play.setAttribute("role", "button"); play.tabIndex = 0;
      const toggle = e => {
        e.stopPropagation();
        if (currentPlayBtn === play) { stopPreview(); return; }
        stopPreview();
        currentAudio = new Audio(v.preview_url);
        currentPlayBtn = play;
        play.classList.add("playing"); play.textContent = "■ Dừng";
        currentAudio.play().catch(() => stopPreview());
        currentAudio.onended = stopPreview;
      };
      play.addEventListener("click", toggle);
      play.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(e); } });
      row.append(play);
    }
    if (v.custom) {
      const del = document.createElement("span");
      del.className = "vc-del"; del.dataset.action = "voice-delete"; del.textContent = "×";
      del.setAttribute("role", "button"); del.tabIndex = 0; del.title = "Xóa giọng nhân bản";
      del.addEventListener("click", async e => {
        e.stopPropagation();
        if (!confirm(`Xóa giọng "${v.display_name}"?`)) return;
        try {
          await api.deleteVoice(v.id);
          if (state.selectedVoice === v.id) { state.selectedVoice = null; localStorage.removeItem("sub.voice"); }
          await refreshVoices();
        } catch (err) { alert(err.message); }
      });
      row.append(del);
    }
    if (row.children.length) body.append(row);
    card.append(body);

    card.addEventListener("click", () => {
      state.selectedVoice = v.id;
      localStorage.setItem("sub.voice", v.id);
      $$(".voice-card", voiceGrid).forEach(c => c.classList.toggle("sel", c.dataset.voiceId === v.id));
      updateStartState();
    });
    voiceGrid.append(card);
  });
  updateStartState();
}

async function refreshVoices() {
  state.voices = await api.voices();
  if (state.selectedVoice && !state.voices.some(v => v.id === state.selectedVoice)) state.selectedVoice = null;
  renderVoices();
}

/* ---------- Nhân bản giọng ---------- */
const cloneBlock = $("#cloneBlock"), cloneToggle = $("#cloneToggle"), cloneForm = $("#cloneForm");
const cloneName = $("#cloneName"), cloneAudio = $("#cloneAudio"), cloneAudioLabel = $("#cloneAudioLabel");
const cloneCreate = $("#cloneCreate"), cloneMsg = $("#cloneMsg");

cloneToggle.addEventListener("click", () => {
  const open = cloneForm.hidden;
  cloneForm.hidden = !open;
  cloneToggle.classList.toggle("open", open);
});
cloneAudioLabel.addEventListener("click", () => cloneAudio.click());
cloneAudio.addEventListener("change", () => {
  cloneAudioLabel.textContent = cloneAudio.files[0] ? cloneAudio.files[0].name : "Chọn file · 3–8 giây, một người nói, ít tạp âm";
});
cloneForm.addEventListener("submit", async e => {
  e.preventDefault();
  const name = cloneName.value.trim();
  const audio = cloneAudio.files[0];
  cloneMsg.hidden = true; cloneMsg.classList.remove("err");
  if (!name) { cloneMsg.textContent = "Hãy đặt tên cho giọng."; cloneMsg.classList.add("err"); cloneMsg.hidden = false; return; }
  if (!audio) { cloneMsg.textContent = "Hãy chọn file audio mẫu."; cloneMsg.classList.add("err"); cloneMsg.hidden = false; return; }
  cloneCreate.disabled = true; cloneCreate.textContent = "Đang tạo…";
  try {
    const v = await api.createVoice(name, audio);
    await refreshVoices();
    state.selectedVoice = v.id;
    localStorage.setItem("sub.voice", v.id);
    renderVoices();
    cloneName.value = ""; cloneAudio.value = "";
    cloneAudioLabel.textContent = "Chọn file · 3–8 giây, một người nói, ít tạp âm";
    cloneMsg.textContent = "Đã tạo giọng. File nghe thử sẽ sẵn sàng sau ít phút.";
    cloneMsg.hidden = false;
  } catch (err) {
    cloneMsg.textContent = err.message; cloneMsg.classList.add("err"); cloneMsg.hidden = false;
  } finally {
    cloneCreate.disabled = false; cloneCreate.textContent = "Tạo giọng";
  }
});

/* ---------- Danh sách video ---------- */
const jobList = $("#jobList"), jobsEmpty = $("#jobsEmpty");

const STATUS_META = {
  queued: { label: "Chờ đến lượt", cls: "st-queued" },
  running: { label: "Đang xử lý", cls: "st-running" },
  cancelling: { label: "Đang dừng…", cls: "st-cancelling" },
  cancelled: { label: "Đã hủy", cls: "st-cancelled" },
  done: { label: "Hoàn tất", cls: "st-done" },
  error: { label: "Lỗi", cls: "st-error" },
};

function voiceName(id) {
  const v = state.voices.find(x => x.id === id);
  return v ? v.display_name : id;
}

function renderJobs() {
  const jobs = state.jobs;
  jobsEmpty.hidden = jobs.length > 0;
  jobList.textContent = "";
  jobs.forEach(j => {
    const card = document.createElement("div");
    card.className = "job-card" + (j.degraded ? " degraded" : "");
    card.dataset.jobId = j.job_id;
    card.dataset.status = j.status;

    /* hàng đầu: tên + trạng thái */
    const top = document.createElement("div"); top.className = "jc-top";
    const left = document.createElement("div");
    const name = document.createElement("div"); name.className = "jc-name"; name.textContent = j.filename;
    const voice = document.createElement("div"); voice.className = "jc-voice"; voice.textContent = "Giọng: " + voiceName(j.voice_id);
    left.append(name, voice);
    const st = document.createElement("span");
    const meta = STATUS_META[j.status] || STATUS_META.queued;
    st.className = "jc-status " + (j.degraded && j.status === "done" ? "st-degraded" : meta.cls);
    st.textContent = j.degraded && j.status === "done" ? "Hoàn tất — có vấn đề" : meta.label;
    top.append(left, st);
    card.append(top);

    /* các bước + tiến trình */
    if (j.status === "running" || j.status === "cancelling") {
      const stages = document.createElement("div"); stages.className = "jc-stages";
      const idx = STAGES.indexOf(j.stage);
      STAGES.forEach((s, i) => {
        const seg = document.createElement("div");
        seg.className = "jc-stage" + (i < idx ? " donestep" : i === idx ? " active" : "");
        seg.title = STAGE_LABELS[s];
        stages.append(seg);
      });
      card.append(stages);

      const mid = document.createElement("div"); mid.className = "jc-mid";
      const msg = document.createElement("div"); msg.className = "jc-msg";
      msg.textContent = j.message || STAGE_LABELS[j.stage] || "";
      const pct = document.createElement("div"); pct.className = "jc-pct";
      pct.textContent = Math.round(j.percent || 0) + "%";
      mid.append(msg, pct);
      card.append(mid);

      const bar = document.createElement("div"); bar.className = "bar jc-bar";
      const fill = document.createElement("div"); fill.className = "bar-fill";
      fill.style.width = (j.percent || 0) + "%";
      bar.append(fill);
      card.append(bar);
    } else if (j.status === "queued") {
      const msg = document.createElement("div"); msg.className = "jc-msg"; msg.style.marginTop = "12px";
      msg.textContent = "Chờ đến lượt — sẽ tự chạy khi có chỗ trống.";
      card.append(msg);
    } else if (j.status === "error") {
      const err = document.createElement("div"); err.className = "jc-error"; err.textContent = j.message || "Đã xảy ra lỗi.";
      card.append(err);
    } else if (j.status === "cancelled") {
      const msg = document.createElement("div"); msg.className = "jc-msg"; msg.style.marginTop = "12px"; msg.textContent = "Đã hủy.";
      card.append(msg);
    }

    /* cảnh báo degraded */
    if (j.degraded && j.warnings && j.warnings.length) {
      const w = document.createElement("div"); w.className = "jc-warnings";
      j.warnings.forEach(t => {
        const line = document.createElement("span"); line.textContent = "⚠ " + t;
        w.append(line);
      });
      card.append(w);
    }

    /* thống kê khi xong */
    if (j.status === "done" && j.attempted_count) {
      const stats = document.createElement("div"); stats.className = "jc-stats";
      stats.textContent = `Đã đọc ${j.spoken_count}/${j.attempted_count} lượt thoại`;
      card.append(stats);
    }

    /* hành động */
    const actions = document.createElement("div"); actions.className = "jc-actions";
    if (j.status === "queued" || j.status === "running" || j.status === "cancelling") {
      const btn = document.createElement("button");
      btn.className = "jc-btn cancel"; btn.dataset.action = "cancel"; btn.textContent = "Hủy";
      btn.disabled = j.status === "cancelling";
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        try { await api.cancel(j.job_id); refreshJobs(true); } catch (_) { btn.disabled = false; }
      });
      actions.append(btn);
    }
    if (j.status === "done") {
      const dv = document.createElement("a");
      dv.className = "jc-btn dl"; dv.dataset.download = "video";
      dv.textContent = "Tải video"; dv.href = `/api/jobs/${encodeURIComponent(j.job_id)}/download/video`;
      const ds = document.createElement("a");
      ds.className = "jc-btn dl"; ds.dataset.download = "srt";
      ds.textContent = "Tải phụ đề (.srt)"; ds.href = `/api/jobs/${encodeURIComponent(j.job_id)}/download/srt`;
      if (state.mock) {
        [dv, ds].forEach(a => a.addEventListener("click", e => e.preventDefault()));
      }
      actions.append(dv, ds);
    }
    if (actions.children.length) card.append(actions);

    jobList.append(card);
  });
}

let refreshing = false;
async function refreshJobs(immediate) {
  if (refreshing) return;
  refreshing = true;
  try {
    const data = await api.jobs();
    state.jobs = data.jobs || [];
    renderJobs();
  } catch (_) {} finally { refreshing = false; }
  schedulePoll();
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  const active = state.jobs.some(j => ACTIVE_STATUSES.has(j.status));
  if (active) state.pollTimer = setTimeout(() => refreshJobs(), 1500);
}

/* ---------- reveal khi cuộn ---------- */
const io = new IntersectionObserver(entries => {
  entries.forEach(en => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
}, { threshold: 0.08 });
$$(".reveal").forEach(el => io.observe(el));

/* ---------- glow bám chuột ---------- */
const glow = $("#glow");
let gx = innerWidth / 2, gy = innerHeight / 3, tx = gx, ty = gy;
addEventListener("pointermove", e => { tx = e.clientX; ty = e.clientY; }, { passive: true });
(function loop() {
  gx += (tx - gx) * 0.06; gy += (ty - gy) * 0.06;
  glow.style.left = gx + "px"; glow.style.top = gy + "px";
  requestAnimationFrame(loop);
})();

/* ---------- khởi động ---------- */
(async function init() {
  /* dò server: fetch /api/voices thất bại → chế độ xem thử */
  try {
    const r = await fetch("/api/voices");
    if (!r.ok) throw new Error();
    state.voices = await r.json();
  } catch (_) {
    state.mock = true;
    $("#mockBadge").hidden = false;
    state.voices = structuredClone(mock.voices);
  }
  if (state.selectedVoice && !state.voices.some(v => v.id === state.selectedVoice)) state.selectedVoice = null;
  renderVoices();

  const cl = await api.cloning().catch(() => ({ enabled: false }));
  state.cloningEnabled = !!cl.enabled;
  cloneBlock.hidden = !state.cloningEnabled;

  await refreshModels();
  if (!state.mock && state.models.models && state.models.models.some(m => m.status === "downloading")) openModelSSE();

  await refreshJobs(true);
})();
