# Windows Frontend Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy Windows landing-page frontend with the Mac operations dashboard while preserving Windows provider routing and making every visible control functional.

**Architecture:** Keep the existing FastAPI static mount and framework-free HTML/CSS/JavaScript client. Port the Mac dashboard as the visual baseline, add backward-compatible job telemetry and the small control APIs it consumes, then replace Mac-only labels with runtime-driven Windows-safe text.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic v2, vanilla HTML/CSS/JavaScript, Server-Sent Events, pytest, FastAPI TestClient, MCP Playwright.

**Spec:** `docs/superpowers/specs/2026-09-05-m-windows-frontend-parity-design.md`

## Global Constraints

- Preserve the Windows `VieNeu`, `OmniVoice`, `Edge` and `Gemini` provider routing; do not copy the Mac OmniVoice-only migration.
- Do not add a JavaScript framework, bundler or Node runtime requirement.
- Runtime labels must not hard-code `MPS`, Apple Silicon or OmniVoice when another provider is active.
- User-controlled filename, voice and status text must use `textContent` or HTML escaping.
- Settings responses must never contain the Gemini API key.
- Test UI behavior with MCP Playwright at 360 px and 1440 px.

---

### Task 1: Replace the legacy Windows application shell

**Files:**
- Create: `tests/test_windows_frontend_parity.py`
- Modify: `tests/test_api.py:88-92`
- Replace: `web/static/index.html`
- Replace: `web/static/style.css`
- Replace: `web/static/app.js`

**Interfaces:**
- Consumes: FastAPI's existing static mount at `/` and the existing `/api/voices`, `/api/model`, `/api/jobs` routes.
- Produces: Dashboard DOM IDs consumed by `web/static/app.js`, including `jobList`, `selectedContent`, `pipelineGraph`, `uploadForm`, `voiceList`, `geminiSettingsForm` and `shutdownButton`.

- [ ] **Step 1: Add a failing served-page contract**

Create `tests/test_windows_frontend_parity.py`:

```python
import re


def test_root_serves_operations_dashboard_without_legacy_shell(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    for marker in (
        'id="overview"',
        'id="jobList"',
        'id="selectedContent"',
        'id="pipelineGraph"',
        'id="uploadForm"',
        'id="voiceList"',
        'id="geminiSettingsForm"',
    ):
        assert marker in html

    for legacy_marker in ('id="mockBadge"', 'id="fileInput"', 'id="voiceGrid"'):
        assert legacy_marker not in html


def test_frontend_assets_have_matching_cache_keys(client):
    html = client.get("/").text
    css_version = re.search(r'href="style\.css\?v=([^\"]+)"', html).group(1)
    js_version = re.search(r'src="app\.js\?v=([^\"]+)"', html).group(1)
    assert css_version and css_version == js_version
```

Update the existing root-page assertion in `tests/test_api.py` to check the functional dashboard marker and cache-busted script:

```python
def test_index_page_is_served_at_root(client):
    page = client.get("/").text
    assert 'id="uploadForm"' in page
    assert 'src="app.js?v=' in page
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_windows_frontend_parity.py tests/test_api.py::test_index_page_is_served_at_root -q
```

Expected: the new parity tests fail because the legacy shell still exposes `mockBadge`, `fileInput` and `voiceGrid` and lacks dashboard IDs.

- [ ] **Step 3: Replace the three static files with the approved Mac baseline**

Perform a mechanical whole-file replacement from these exact sources:

```text
/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/web/static/index.html
/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/web/static/style.css
/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/web/static/app.js
```

Use the same version value for `style.css?v=...` and `app.js?v=...`. Preserve these Mac fixes in the copied source:

```javascript
updateGraphFlow(snapshot);
renderJobs();
```

```javascript
const completed = job.status === "done" ? index <= currentIndex : index < currentIndex;
```

The copied HTML must omit `selectedProgress` and `.progress-track.large` from `SELECTED JOB`.

- [ ] **Step 4: Run the served-page tests and confirm GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_windows_frontend_parity.py tests/test_api.py::test_index_page_is_served_at_root -q
node --check web/static/app.js
```

Expected: both pytest tests pass and Node reports no syntax error.

- [ ] **Step 5: Commit the shell replacement**

```bash
git add tests/test_windows_frontend_parity.py tests/test_api.py web/static/index.html web/static/style.css web/static/app.js
git commit -m "feat: replace Windows frontend with operations dashboard"
```

---

### Task 2: Add atomic realtime job telemetry and terminal deletion

**Files:**
- Create: `tests/test_job_telemetry.py`
- Modify: `tests/test_job_manager.py`
- Modify: `tests/test_api.py`
- Modify: `backend/job_manager.py:21-291`
- Modify: `backend/routes/jobs.py:112-206`
- Modify: `pipeline/backends.py:35-116`

**Interfaces:**
- Consumes: pipeline callback `progress(stage: str, fraction: float, message: str)` and each TTS synthesizer's optional `engine`, `device` and `batch_size` attributes.
- Produces: `Job.update_progress(...)`, `Job.update_runtime_from_backend(...)`, `Job.mark_finished(...)`, `JobManager.delete(job_id)`, `CompositeBackend.runtime_info()`, `GET /api/jobs/{job_id}/telemetry`, `DELETE /api/jobs/{job_id}` and additive telemetry fields in `Job.snapshot()`.

- [ ] **Step 1: Add failing telemetry behavior tests**

Create `tests/test_job_telemetry.py` with provider-neutral fixtures:

```python
from pathlib import Path

from backend.job_manager import Job


def test_running_snapshot_exposes_live_timing_runtime_and_eta():
    job = Job(
        id="abc",
        filename="clip.mp4",
        workdir=Path("jobs/abc"),
        voice_id="voice",
        status="running",
        stage="synthesize",
        percent=50.0,
        created_at=90.0,
        started_at=100.0,
        stage_started_at=112.0,
        engine="vieneu",
        device="cuda",
        batch_size=16,
    )

    snapshot = job.snapshot(now=130.0)

    assert snapshot["elapsed_seconds"] == 30.0
    assert snapshot["stage_elapsed_seconds"] == 18.0
    assert snapshot["eta_seconds"] == 30.0
    assert snapshot["engine"] == "vieneu"
    assert snapshot["device"] == "cuda"
    assert snapshot["batch_size"] == 16


def test_progress_is_monotonic_and_records_stage_events():
    job = Job(id="abc", filename="clip.mp4", workdir=Path("jobs/abc"), voice_id="voice")
    job.update_progress("transcribe", 1.0, "Xong nhận diện", now=10.0)
    job.update_progress("extract", 0.0, "Tín hiệu trễ", now=11.0)

    snapshot = job.snapshot(now=12.0)
    assert snapshot["percent"] == 30.0
    assert [item["stage"] for item in snapshot["stage_history"]] == [
        "transcribe",
        "extract",
    ]
    assert snapshot["events"][-1]["message"] == "Tín hiệu trễ"
```

Add deletion tests to `tests/test_job_manager.py`:

```python
def test_delete_removes_terminal_job_and_its_workdir(tmp_path):
    manager = JobManager(max_workers=1)
    workdir = tmp_path / "done"
    workdir.mkdir()
    (workdir / "output.mp4").write_bytes(b"video")
    job = Job(id="done", filename="clip.mp4", workdir=workdir, voice_id="voice", status="done")
    manager._jobs[job.id] = job

    assert manager.delete(job.id) == "done"
    assert manager.get(job.id) is None
    assert not workdir.exists()


def test_delete_rejects_active_job(tmp_path):
    manager = JobManager(max_workers=1)
    job = Job(id="live", filename="clip.mp4", workdir=tmp_path, voice_id="voice", status="running")
    manager._jobs[job.id] = job

    with pytest.raises(RuntimeError, match="đang chạy"):
        manager.delete(job.id)
```

Add an HTTP behavior test beside the existing `_put(...)` helper in `tests/test_api.py`:

```python
def test_job_telemetry_route_returns_the_monitor_snapshot(client, tmp_path):
    _put(Job(
        id="telemetry-job",
        filename="clip.mp4",
        workdir=tmp_path,
        voice_id="voice",
        status="running",
        started_at=100.0,
        stage_started_at=110.0,
    ))

    response = client.get("/api/jobs/telemetry-job/telemetry")

    assert response.status_code == 200
    assert response.json()["job_id"] == "telemetry-job"
    assert response.json()["telemetry_only"] is True
```

- [ ] **Step 2: Run the telemetry tests and confirm RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_job_telemetry.py tests/test_job_manager.py -q
```

Expected: construction fails for missing telemetry fields and `JobManager.delete` is absent.

- [ ] **Step 3: Implement telemetry without changing Windows provider routing**

Extend `Job` with these fields and a lock:

```python
started_at: float | None = None
finished_at: float | None = None
stage_started_at: float | None = None
stage_fraction: float = 0.0
stage_history: list[dict] = field(default_factory=list)
events: list[dict] = field(default_factory=list)
engine: str = ""
device: str = ""
batch_size: int = 0
telemetry_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
```

Port the provider-neutral implementations of `_optional_float`, `_safe_int`, `_append_event_locked`, `mark_started`, `update_runtime`, `update_runtime_from_backend`, `update_progress`, `mark_finished`, `mark_persisted_result` and the additive snapshot fields from the Mac `backend/job_manager.py`.

In `_run`, start telemetry before constructing the backend, update runtime immediately after construction, route all progress through `job.update_progress`, and record a terminal event before exposing terminal status. Preserve the existing Windows `backend_factory`, VieNeu fallback and `PipelineOptions` behavior.

Restore all telemetry fields from `job.json` with empty/zero defaults so pre-migration jobs remain readable.

Add `JobManager.delete(job_id)` exactly at the registry boundary: reject `ACTIVE_STATUSES`, remove registry/future entries while holding `_lock`, then remove only `job.workdir`.

Expose non-sensitive runtime facts from `CompositeBackend`:

```python
def runtime_info(self) -> tuple[str, str, int]:
    synthesizer = self._synthesizer
    engine = getattr(synthesizer, "engine", synthesizer.__class__.__name__)
    device = getattr(synthesizer, "device", "")
    batch = getattr(synthesizer, "batch_size", 0)
    engine = engine() if callable(engine) else engine
    device = device() if callable(device) else device
    batch = batch() if callable(batch) else batch
    try:
        normalized_batch = max(0, int(str(batch or 0)))
    except (TypeError, ValueError):
        normalized_batch = 0
    return str(engine), str(device), normalized_batch
```

Register the delete route:

```python
@router.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    try:
        return {"deleted": manager.delete(job_id)}
    except LookupError:
        raise HTTPException(404, "Không tìm thấy công việc này.") from None
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
```

Register the dedicated monitor payload before the generic cancel/events routes:

```python
@router.get("/api/jobs/{job_id}/telemetry")
def job_telemetry(job_id: str) -> dict:
    payload = _require(job_id).snapshot()
    payload["telemetry_only"] = True
    return payload
```

- [ ] **Step 4: Run focused and neighboring tests and confirm GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_job_telemetry.py tests/test_job_manager.py tests/test_progress.py tests/test_cancel.py tests/test_api.py -q
```

Expected: all selected tests pass, including restored-job and existing progress behavior.

- [ ] **Step 5: Commit realtime telemetry**

```bash
git add backend/job_manager.py backend/routes/jobs.py pipeline/backends.py tests/test_job_telemetry.py tests/test_job_manager.py tests/test_api.py
git commit -m "feat: expose realtime job telemetry on Windows"
```

---

### Task 3: Add Gemini settings and cross-platform tool shutdown

**Files:**
- Create: `backend/routes/settings.py`
- Create: `backend/process_control.py`
- Create: `tests/test_runtime_controls.py`
- Modify: `backend/main.py:12-61`

**Interfaces:**
- Consumes: `backend.config.ROOT`, the singleton `settings`, and the local server PID.
- Produces: `GET /api/settings/gemini`, `POST /api/settings/gemini`, `POST /api/shutdown`, `schedule_shutdown(root_pid=None, delay_seconds=0.2)` and `terminate_process_tree(root_pid)`.

- [ ] **Step 1: Add failing settings and shutdown tests**

Create `tests/test_runtime_controls.py`:

```python
from backend.config import settings


def test_gemini_status_is_masked(client, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "secret-value")
    monkeypatch.setattr(settings, "gemini_backend", "vertex")

    response = client.get("/api/settings/gemini")

    assert response.status_code == 200
    assert response.json() == {"configured": True, "backend": "vertex"}
    assert "secret-value" not in response.text


def test_gemini_update_atomically_persists_without_echo(tmp_path, client, monkeypatch):
    import backend.routes.settings as settings_route

    env_path = tmp_path / ".env"
    env_path.write_text("OTHER_SETTING=keep\n", encoding="utf-8")
    monkeypatch.setattr(settings_route, "ENV_PATH", env_path)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "gemini_backend", "developer")

    response = client.post(
        "/api/settings/gemini",
        json={"api_key": "key-with-#-hash", "backend": "vertex"},
    )

    assert response.json() == {"saved": True, "configured": True, "backend": "vertex"}
    assert env_path.read_text(encoding="utf-8") == (
        'OTHER_SETTING=keep\nGEMINI_BACKEND=vertex\nGEMINI_API_KEY="key-with-#-hash"\n'
    )


def test_windows_tree_shutdown_uses_taskkill(monkeypatch):
    import backend.process_control as control

    calls = []
    monkeypatch.setattr(control.subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))

    control._terminate_windows_tree(4321)

    assert calls[0][0] == ["taskkill", "/PID", "4321", "/T", "/F"]


def test_shutdown_route_returns_before_scheduling(client, monkeypatch):
    calls = []
    monkeypatch.setattr("backend.main.schedule_shutdown", lambda: calls.append("scheduled"))

    response = client.post("/api/shutdown")

    assert response.status_code == 200
    assert response.json() == {"shutting_down": True}
    assert calls == ["scheduled"]
```

- [ ] **Step 2: Run the control tests and confirm RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_runtime_controls.py -q
```

Expected: imports or route requests fail because settings and process-control modules are absent.

- [ ] **Step 3: Implement masked settings and Windows-aware shutdown**

Port `backend/routes/settings.py` from the Mac project, preserving:

```python
_ALLOWED_BACKENDS = {"developer", "vertex"}

def _masked_status() -> dict:
    return {"configured": bool(settings.gemini_api_key), "backend": settings.gemini_backend}
```

Keep the atomic temp-file write, JSON dotenv quoting, `0o600` permission on POSIX and `os.replace`. Reject empty keys, newline injection and unknown backends.

Create a cross-platform `backend/process_control.py`. On Windows use:

```python
def _terminate_windows_tree(root_pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(int(root_pid)), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )
```

On POSIX retain the Mac descendant-first SIGTERM/SIGKILL implementation so tests and local development remain safe. `schedule_shutdown` must delay work on a daemon thread so the HTTP response can leave first.

Register `settings.router` and the shutdown endpoint in `backend/main.py` before the static mount:

```python
app.include_router(settings_routes.router)

@app.post("/api/shutdown")
def shutdown_tool() -> dict:
    schedule_shutdown()
    return {"shutting_down": True}
```

- [ ] **Step 4: Run control and API tests and confirm GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_runtime_controls.py tests/test_api.py -q
```

Expected: settings remain masked, the `.env` fixture is updated atomically, Windows taskkill arguments are exact and the route returns `200`.

- [ ] **Step 5: Commit runtime controls**

```bash
git add backend/routes/settings.py backend/process_control.py backend/main.py tests/test_runtime_controls.py
git commit -m "feat: add Windows dashboard runtime controls"
```

---

### Task 4: Secure result downloads and voice previews

**Files:**
- Create: `tests/test_security_paths.py`
- Create: `tests/test_voice_preview_api.py`
- Modify: `backend/routes/jobs.py:169-206`
- Modify: `backend/routes/voices.py:1-128`

**Interfaces:**
- Consumes: terminal `Job.workdir`, `Job.video_path`, `Job.srt_path`, `settings.previews_dir` and the configured voice catalog.
- Produces: `_job_file(job, value)`, `_preview_path(voice_id)` and `GET /api/voices/{voice_id}/preview`.

- [ ] **Step 1: Add failing path-boundary tests**

Port these exact behavior suites from the Mac project:

```text
/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/tests/test_security_paths.py
/Users/nthieuu/Documents/Claude/Code/ToolVietSubMac/tests/test_voice_preview_api.py
```

The result tests must cover regular files, outside paths, symlinks and nested files. The voice tests must cover known preview, missing preview and unknown/path-like voice IDs.

- [ ] **Step 2: Run the security tests and confirm RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_security_paths.py tests/test_voice_preview_api.py -q
```

Expected: outside/nested job paths are served or `_preview_path`/preview route is absent.

- [ ] **Step 3: Implement explicit path ownership checks**

Add `_job_file(job, value)` and call it from both download routes:

```python
def _job_file(job: Job, value: str) -> Path:
    try:
        raw = Path(value)
        root = job.workdir.resolve(strict=True)
        path = raw.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(404, "Không tìm thấy file kết quả.") from None
    if raw.is_symlink() or (path != root and root not in path.parents) or path.parent != root:
        raise HTTPException(404, "Không tìm thấy file kết quả.")
    return path
```

Add `_preview_path(voice_id)` that accepts only a registered custom voice or an ID in `available_voices(...)`, then resolve the preview filename under `settings.previews_dir`. Register:

```python
@router.get("/api/voices/{voice_id}/preview")
def preview_voice(voice_id: str) -> FileResponse:
    path = _preview_path(voice_id)
    if not path.is_file():
        raise HTTPException(404, "Giọng này chưa có file demo.")
    return FileResponse(path, media_type="audio/wav", filename=f"{voice_id}.wav")
```

Use `_preview_path` for list/create/delete preview operations without changing the current Windows VieNeu/OmniVoice clone routing.

- [ ] **Step 4: Run security, voice and API tests and confirm GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_security_paths.py tests/test_voice_preview_api.py tests/test_custom_voices.py tests/test_api.py -q
```

Expected: all selected tests pass and no provider-specific voice test regresses.

- [ ] **Step 5: Commit path-boundary support**

```bash
git add backend/routes/jobs.py backend/routes/voices.py tests/test_security_paths.py tests/test_voice_preview_api.py
git commit -m "fix: constrain Windows dashboard file routes"
```

---

### Task 5: Make the copied dashboard provider-neutral on Windows

**Files:**
- Modify: `tests/test_windows_frontend_parity.py`
- Modify: `web/static/index.html`
- Modify: `web/static/app.js`
- Modify: `web/static/style.css`

**Interfaces:**
- Consumes: optional `job.engine`, `job.device`, `job.batch_size`, `/api/model` model labels and voice metadata.
- Produces: provider-neutral static copy and bounded Voice Lab controls while retaining all Mac dashboard behavior.

- [ ] **Step 1: Add failing Windows-specific UI behavior checks**

Extend `tests/test_windows_frontend_parity.py`:

```python
def test_dashboard_copy_is_provider_neutral_and_keeps_requested_regressions(client):
    html = client.get("/").text
    js = client.get("/app.js").text
    css = client.get("/style.css").text

    assert "Apple Silicon" not in html
    assert "MPS · batch" not in html
    assert 'id="selectedProgress"' not in html
    assert 'class="progress-track large"' not in html
    assert 'job.status === "done" ? index <= currentIndex : index < currentIndex' in js
    assert "updateGraphFlow(snapshot)" in js
    assert "renderJobs();" in js
    assert "min-width: 72px" in css
    assert "text-overflow: ellipsis" in css
```

- [ ] **Step 2: Run the provider-neutral test and confirm RED**

Run:

```bash
./.venv/bin/python -m pytest tests/test_windows_frontend_parity.py -q
```

Expected: the copied Mac HTML still contains at least one Mac-only engine/device label.

- [ ] **Step 3: Replace Mac-only copy while preserving layout and behavior**

Change the static synthesize node title/note to `TTS Engine` / `Theo cấu hình job`, change the Voice Lab badge/copy to `LOCAL VOICE` and provider-neutral Vietnamese, and make clone success say `Đã tạo giọng nhân bản.`.

Keep runtime rendering data-driven:

```javascript
const runtimeEngine = focus?.engine || "Chưa có engine đang chạy";
const runtimeDevice = focus?.device || "—";
```

Do not remove system health model rows; they already display the backend's actual model list. Preserve `.voice-select-button` as a fixed, overflow-hidden flex item:

```css
.voice-select-button {
  min-width: 72px;
  flex: 0 0 72px;
  overflow: hidden;
  text-overflow: ellipsis;
}
```

Increment both asset query strings together after this final static edit.

- [ ] **Step 4: Run frontend contracts and syntax checks and confirm GREEN**

Run:

```bash
./.venv/bin/python -m pytest tests/test_windows_frontend_parity.py -q
node --check web/static/app.js
git diff --check
```

Expected: all checks exit `0`.

- [ ] **Step 5: Commit Windows-specific dashboard copy**

```bash
git add tests/test_windows_frontend_parity.py web/static/index.html web/static/app.js web/static/style.css
git commit -m "fix: adapt operations dashboard copy for Windows"
```

---

### Task 6: Full regression and MCP Playwright verification

**Files:**
- No production files unless verification reveals a reproducible defect.
- Optional evidence: `qa/windows-dashboard-360.png`
- Optional evidence: `qa/windows-dashboard-1440.png`

**Interfaces:**
- Consumes: the complete Windows application at `http://127.0.0.1:8000/`.
- Produces: fresh repository test evidence and browser evidence for the approved acceptance criteria.

- [ ] **Step 1: Run the complete repository suite**

Run:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost ./.venv/bin/python -m pytest -q
node --check web/static/app.js
git diff --check
```

Expected: pytest reports zero failures and both static checks exit `0`.

- [ ] **Step 2: Start the Windows project server locally**

Run:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost ./.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Keep the process in a reusable terminal session for browser verification.

- [ ] **Step 3: Verify the dashboard with MCP Playwright at 1440 px**

Use MCP Playwright to navigate to `http://127.0.0.1:8000/`, resize to `1440x1000`, and assert through the live DOM:

```javascript
({
  title: document.title,
  dashboard: Boolean(document.querySelector("#overview")),
  legacy: Boolean(document.querySelector("#mockBadge, #voiceGrid")),
  selectedBars: document.querySelectorAll("#selectedContent .progress-track").length,
  queueBars: document.querySelectorAll("#jobList .progress-track").length,
  pageOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
})
```

Expected: `dashboard=true`, `legacy=false`, `selectedBars=0`, `pageOverflow=false`. Queue bars may be zero only when the job list is empty.

- [ ] **Step 4: Verify realtime SSE, Export and Voice Lab with MCP Playwright**

Use MCP Playwright request interception to provide complete mock responses for `/api/voices`, `/api/model`, `/api/jobs`, `/api/settings/gemini` and one `/api/jobs/demo/events` stream. The stream must emit a `running/synthesize/48%` snapshot followed by `done/mux/100%`.

Assert observable DOM behavior:

```javascript
({
  queuePercent: document.querySelector('[data-job-id="demo"] .job-progress-row strong')?.textContent,
  exportCompleted: document.querySelector('[data-stage="mux"]')?.classList.contains("completed"),
  selectedProgressBars: document.querySelectorAll("#selectedContent .progress-track").length,
  voiceSelectWidth: document.querySelector(".voice-select-button")?.getBoundingClientRect().width,
})
```

Expected after the terminal snapshot: `queuePercent="100%"`, `exportCompleted=true`, `selectedProgressBars=0` and `voiceSelectWidth=72`.

- [ ] **Step 5: Verify responsive overflow and console output**

Resize with MCP Playwright to `360x800`. Verify:

```javascript
({
  pageOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
  voiceOverflow: [...document.querySelectorAll(".voice-item")].some(
    item => item.scrollWidth > item.clientWidth
  ),
  graphOwnsHorizontalScroll: document.querySelector(".graph-scroll-shell").scrollWidth
    > document.querySelector(".graph-scroll-shell").clientWidth,
})
```

Expected: `pageOverflow=false`, `voiceOverflow=false`, `graphOwnsHorizontalScroll=true`. Read MCP Playwright console messages at `error` level and expect no application runtime errors.

- [ ] **Step 6: Inspect the final diff and commit any verified evidence**

Run:

```bash
git status --short
git diff --stat HEAD~5..HEAD
git log -6 --oneline
```

If screenshots were saved, add only the two explicit QA files and commit them:

```bash
git add qa/windows-dashboard-360.png qa/windows-dashboard-1440.png
git commit -m "test: capture Windows dashboard browser verification"
```

Do not add the pre-existing untracked `.codex/` directory.
