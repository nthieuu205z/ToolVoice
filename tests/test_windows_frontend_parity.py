import json
import re
import subprocess
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

from backend.job_manager import manager
from backend.main import app


class _GraphAffordanceMarkupParser(HTMLParser):
    _VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self._ancestors = []
        self.cue_ancestors = None

    def handle_starttag(self, tag, attrs):
        classes = set(dict(attrs).get("class", "").split())
        if "graph-scroll-cue" in classes:
            self.cue_ancestors = tuple(self._ancestors)
        if tag not in self._VOID_ELEMENTS:
            self._ancestors.append(classes)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID_ELEMENTS:
            self._ancestors.pop()

    def handle_endtag(self, tag):
        if tag not in self._VOID_ELEMENTS:
            self._ancestors.pop()


class _DashboardMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.class_sets = []

    def handle_starttag(self, _tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        self.class_sets.append(set(attributes.get("class", "").split()))


_DASHBOARD_BEHAVIOR_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");

class FakeClassList {
  constructor(owner) { this.owner = owner; }
  _set() { return new Set(this.owner.className.split(/\s+/).filter(Boolean)); }
  add(...names) { const classes = this._set(); names.forEach(name => classes.add(name)); this.owner.className = [...classes].join(" "); }
  remove(...names) { const classes = this._set(); names.forEach(name => classes.delete(name)); this.owner.className = [...classes].join(" "); }
  contains(name) { return this._set().has(name); }
  toggle(name, force) { const classes = this._set(); const enabled = force === undefined ? !classes.has(name) : Boolean(force); enabled ? classes.add(name) : classes.delete(name); this.owner.className = [...classes].join(" "); return enabled; }
}

class FakeElement {
  constructor(tag = "div", id = "") {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.className = "";
    this.classList = new FakeClassList(this);
    this.dataset = {};
    this.style = {};
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.hidden = false;
    this.disabled = false;
    this.value = "";
    this.files = [];
    this.scrollWidth = 820;
    this.clientWidth = 460;
    this.scrollLeft = 0;
    this._textContent = "";
    this._innerHTML = "";
  }
  get textContent() { return this._textContent; }
  set textContent(value) { this._textContent = String(value); this._innerHTML = ""; this.children = []; }
  get innerHTML() { return this._innerHTML; }
  set innerHTML(value) { this._innerHTML = String(value); this._textContent = ""; this.children = []; }
  append(...children) { this.children.push(...children); }
  addEventListener(type, listener) { const listeners = this.listeners.get(type) || []; listeners.push(listener); this.listeners.set(type, listeners); }
  dispatchEvent(event) { event.target = this; return Promise.all((this.listeners.get(event.type) || []).map(listener => listener(event))); }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name); }
  querySelectorAll(selector) {
    if (selector === ".job-card") return this.children.filter(child => child.classList.contains("job-card"));
    return [];
  }
  querySelector(selector) { return document.querySelector(`${this.id || this.tagName}:${selector}`); }
  closest(selector) { return selector === ".graph-scroll-frame" ? graphFrame : null; }
  getBoundingClientRect() { return { height: 96 }; }
  reset() {}
  remove() {}
  scrollIntoView() {}
  click() {}
}

const ids = new Map();
const selectors = new Map();
const graphNodes = ["extract", "transcribe", "translate", "synthesize", "subtitle", "assemble", "mux"].map(stage => {
  const node = new FakeElement();
  node.dataset.stage = stage;
  return node;
});
const graphLinks = Array.from({ length: 6 }, () => new FakeElement());
const graphFrame = new FakeElement();
graphFrame.className = "graph-scroll-frame";
const graphShell = new FakeElement();
graphShell.className = "graph-scroll-shell";

const document = {
  createElement: tag => new FakeElement(tag),
  addEventListener() {},
  querySelector(selector) {
    if (selector === ".graph-scroll-shell") return graphShell;
    if (selector === '[data-stage="mux"]') {
      const queueCards = ids.get("jobList")?.children || [];
      return [...queueCards, ...graphNodes].find(element => element.dataset.stage === "mux") || null;
    }
    if (selector.startsWith("#") && !selector.includes(" ")) {
      const id = selector.slice(1);
      if (!ids.has(id)) ids.set(id, new FakeElement("div", id));
      return ids.get(id);
    }
    if (!selectors.has(selector)) selectors.set(selector, new FakeElement());
    return selectors.get(selector);
  },
  querySelectorAll(selector) {
    if (selector === ".graph-node") return graphNodes;
    if (selector === ".graph-link") return graphLinks;
    return [];
  },
};

const storage = new Map();
const localStorage = {
  getItem: key => storage.get(key) || null,
  setItem: (key, value) => storage.set(key, String(value)),
  removeItem: key => storage.delete(key),
};
class FakeEventSource {
  static instances = [];
  constructor(url) { this.url = url; this.closed = false; FakeEventSource.instances.push(this); }
  close() { this.closed = true; }
}
class FakeEvent { constructor(type) { this.type = type; this.target = null; } preventDefault() {} stopPropagation() {} }
class FakeFormData { append() {} }
class FakeAudio {
  static urls = [];
  constructor(url) { this.src = url; FakeAudio.urls.push(url); }
  addEventListener() {}
  pause() {}
  play() { return Promise.resolve(); }
  removeAttribute(name) { if (name === "src") this.src = ""; }
  load() {}
}
class FakeXMLHttpRequest {}

const initialJob = {
  job_id: "job-1", filename: "demo.mp4", voice_id: "voice-1", status: "running",
  stage: "transcribe", percent: 10, message: "Nhận diện", elapsed_seconds: 5,
  stage_elapsed_seconds: 2, created_at: Date.now() / 1000, engine: "edge",
  device: "cpu", batch_size: 4, events: [],
};
const fetchUrls = [];
let cloneCreated = false;
let previewChecks = 0;
async function fetch(url) {
  fetchUrls.push(url);
  if (url === "/api/voices/custom") cloneCreated = true;
  const voices = [{ id: "voice-1", display_name: "Voice", custom: false, preview_status: "missing", preview_url: "" }];
  if (cloneCreated) voices.push({ id: "clone-1", display_name: "Giọng dài", custom: true, preview_status: previewChecks > 1 ? "ready" : "generating", preview_url: previewChecks > 1 ? "/api/voices/clone-1/preview" : "" });
  const body = url === "/api/voices" ? voices
    : url === "/api/model" ? { models: [] }
    : url === "/api/jobs" ? { jobs: [initialJob] }
    : url === "/api/settings/gemini" ? { configured: false, backend: "developer" }
    : url === "/api/voices/custom" ? { id: "clone-1", preview_status: "generating" }
    : url === "/api/voices/clone-1/preview-status" ? { status: ++previewChecks > 1 ? "ready" : "generating" }
    : {};
  return { ok: true, status: 200, json: async () => body };
}

let timerId = 0;
const timeouts = new Map();
const intervals = new Map();
const window = {
  addEventListener() {},
  setInterval(callback, delay) { const id = ++timerId; intervals.set(id, { callback, delay }); return id; },
  clearInterval(id) { intervals.delete(id); },
  setTimeout(callback, delay) { const id = ++timerId; timeouts.set(id, { callback, delay }); return id; },
  clearTimeout(id) { timeouts.delete(id); },
  confirm: () => true,
};
const context = {
  console, document, window, localStorage, fetch,
  EventSource: FakeEventSource, Event: FakeEvent, FormData: FakeFormData,
  Audio: FakeAudio, XMLHttpRequest: FakeXMLHttpRequest,
  getComputedStyle: () => ({ rowGap: "9" }),
  setTimeout: window.setTimeout,
  clearTimeout: window.clearTimeout,
};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);

(async () => {
  await new Promise(resolve => setImmediate(resolve));
  await new Promise(resolve => setImmediate(resolve));
  let source = FakeEventSource.instances[0];
  if (!source || !source.onmessage) throw new Error("dashboard did not open the selected job SSE stream");
  source.onmessage({ data: JSON.stringify({ ...initialJob, stage: "translate", percent: 73, message: "Dịch 73%" }) });
  const queueAfterProgress = ids.get("jobList").children[0].innerHTML;
  source.onmessage({ data: JSON.stringify({ ...initialJob, status: "done", stage: "mux", percent: 100, message: "Hoàn tất" }) });
  const queueAfterDone = ids.get("jobList").children[0].innerHTML;
  const exportCompleted = document.querySelector('[data-stage="mux"]')?.classList.contains("completed") || false;
  const allFlowsCompleted = graphLinks.every(link => link.classList.contains("flow-complete"));

  await context.refreshJobs();
  source = FakeEventSource.instances.at(-1);
  const intervalBaseline = intervals.size;
  const timeoutBaseline = new Set(timeouts.keys());
  source.onerror();
  source.onerror();
  const firstRetryIds = [...timeouts.keys()].filter(id => !timeoutBaseline.has(id));
  const oneRetryLoop = firstRetryIds.length === 1;
  const onePollingFallback = intervals.size === intervalBaseline + 1;
  const firstRetry = timeouts.get(firstRetryIds[0]);
  timeouts.delete(firstRetryIds[0]);
  await firstRetry.callback();
  source = FakeEventSource.instances.at(-1);
  source.onopen();
  const recoveryStopsPolling = intervals.size === intervalBaseline;

  const retryDelays = [];
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const before = new Set(timeouts.keys());
    source.onerror();
    const retryId = [...timeouts.keys()].find(id => !before.has(id));
    if (!retryId) break;
    const retry = timeouts.get(retryId);
    retryDelays.push(retry.delay);
    timeouts.delete(retryId);
    await retry.callback();
    source = FakeEventSource.instances.at(-1);
  }
  const boundedBackoff = retryDelays.length > 1 && retryDelays.length < 10
    && retryDelays.every((delay, index) => index === 0 || delay > retryDelays[index - 1]);
  const persistentPollingFallback = intervals.size === intervalBaseline + 1;

  document.querySelector("#cloneName").value = "Giọng dài";
  document.querySelector("#cloneAudio").files = [{ name: "voice.wav" }];
  await ids.get("voiceForm").dispatchEvent(new FakeEvent("submit"));
  await new Promise(resolve => setImmediate(resolve));
  const pollsReadiness = fetchUrls.includes("/api/voices/clone-1/preview-status");
  const previewTimerEntry = [...timeouts.entries()].find(([, timer]) => timer.delay < 4200);
  if (previewTimerEntry) {
    timeouts.delete(previewTimerEntry[0]);
    await previewTimerEntry[1].callback();
    await new Promise(resolve => setImmediate(resolve));
  }
  context.togglePreview(
    { id: "clone-1", preview_url: "/api/voices/clone-1/preview" },
    new FakeElement("button")
  );
  const cloneToast = ids.get("toastStack").children.at(-1)?.textContent;
  process.stdout.write(JSON.stringify({
    queueUpdated: queueAfterProgress.includes("73%") && queueAfterProgress.includes("Dịch 73%"),
    queueCompleted: queueAfterDone.includes("100%") && queueAfterDone.includes("Hoàn tất"),
    exportCompleted,
    allFlowsCompleted,
    runtimeMetric: ids.get("metricEngine").textContent,
    runtimeDevice: ids.get("metricDevice").textContent,
    graphRuntime: ids.get("graphTtsNote").textContent,
    cloneToast,
    oneRetryLoop,
    onePollingFallback,
    recoveryStopsPolling,
    boundedBackoff,
    persistentPollingFallback,
    pollsReadiness,
    securePreviewConsumed: FakeAudio.urls.at(-1) === "/api/voices/clone-1/preview",
  }));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""

@pytest.fixture
def client():
    yield TestClient(app)
    manager._jobs.clear()
    manager._futures.clear()


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


def test_mobile_pipeline_scroll_viewport_is_the_constrained_shell(client):
    css = client.get("/style.css").text
    responsive_css = css.split("@media (max-width: 1100px) {", 1)[1].split(
        "@media (max-width: 900px)", 1
    )[0]
    shell_rule = re.search(r"\.graph-scroll-shell\s*\{([^}]*)\}", responsive_css).group(1)
    track_rule = re.search(r"\.graph-track\s*\{([^}]*)\}", responsive_css).group(1)

    assert "overflow-x: auto" in shell_rule
    assert "overflow-x: auto" not in track_rule
    assert "min-width: 820px" in track_rule


def test_pipeline_scroll_affordance_observes_the_shell(client):
    js = client.get("/app.js").text

    assert "const scrollable = shell.scrollWidth > shell.clientWidth + 1;" in js
    assert "shell.scrollLeft + shell.clientWidth >= shell.scrollWidth - 1" in js
    assert 'graphScrollShell?.addEventListener("scroll", syncGraphScrollAffordance' in js


def test_pipeline_scroll_affordance_stays_pinned_and_hides_at_end(client):
    html = client.get("/").text
    css = client.get("/style.css").text
    js = client.get("/app.js").text

    parser = _GraphAffordanceMarkupParser()
    parser.feed(html)
    assert {"graph-scroll-frame"} in parser.cue_ancestors
    assert {"graph-scroll-shell"} not in parser.cue_ancestors

    responsive_css = css.split("@media (max-width: 1100px) {", 1)[1].split(
        "@media (max-width: 900px)", 1
    )[0]
    assert ".graph-scroll-frame.is-scrollable:not(.is-at-end)::after { opacity: 1; }" in responsive_css
    assert ".graph-scroll-frame.is-scrollable:not(.is-at-end) .graph-scroll-cue { opacity: .9; }" in responsive_css
    assert 'const frame = shell.closest(".graph-scroll-frame");' in js
    assert 'frame.classList.toggle("is-at-end"' in js


def test_dashboard_copy_and_selected_job_markup_are_provider_neutral(client):
    html = client.get("/").text
    parser = _DashboardMarkupParser()
    parser.feed(html)

    visible_copy = html.lower()
    assert "apple silicon" not in visible_copy
    assert "mps" not in visible_copy
    assert "omnivoice" not in visible_copy
    assert "TTS Engine" in html
    assert "Theo cấu hình job" in html
    assert "LOCAL VOICE" in html
    assert "selectedProgress" not in parser.ids
    assert {"progress-track", "large"} not in parser.class_sets


def test_voice_select_button_stays_bounded(client):
    css = client.get("/style.css").text
    rule = re.search(r"\.voice-select-button\s*\{([^}]*)\}", css).group(1)
    declarations = {
        name.strip(): value.strip()
        for name, value in (declaration.split(":", 1) for declaration in rule.split(";") if ":" in declaration)
    }

    assert declarations["min-width"] == "72px"
    assert declarations["flex"] == "0 0 72px"
    assert declarations["overflow"] == "hidden"
    assert declarations["text-overflow"] == "ellipsis"


def test_selected_file_text_wrapper_can_shrink_for_ellipsis(client):
    html = client.get("/").text
    css = client.get("/style.css").text
    parser = _DashboardMarkupParser()
    parser.feed(html)

    assert {"selected-file-copy"} in parser.class_sets
    rule = re.search(r"\.selected-file-copy\s*\{([^}]*)\}", css).group(1)
    declarations = {
        name.strip(): value.strip()
        for name, value in (declaration.split(":", 1) for declaration in rule.split(";") if ":" in declaration)
    }
    assert declarations["min-width"] == "0"
    assert declarations["flex"] == "1 1 auto"


def test_realtime_snapshot_repaints_queue_and_completed_graph(client, tmp_path):
    script_path = tmp_path / "app.js"
    script_path.write_text(client.get("/app.js").text, encoding="utf-8")
    result = subprocess.run(
        ["node", "-e", _DASHBOARD_BEHAVIOR_HARNESS, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    behavior = json.loads(result.stdout)

    assert behavior["queueUpdated"] is True
    assert behavior["queueCompleted"] is True
    assert behavior["exportCompleted"] is True
    assert behavior["allFlowsCompleted"] is True
    assert "edge" in behavior["runtimeMetric"]
    assert behavior["runtimeDevice"] == "cpu"
    assert "edge" in behavior["graphRuntime"]
    assert "cpu" in behavior["graphRuntime"]
    assert behavior["cloneToast"] == "Đã tạo giọng nhân bản."
    assert behavior["oneRetryLoop"] is True
    assert behavior["onePollingFallback"] is True
    assert behavior["recoveryStopsPolling"] is True
    assert behavior["boundedBackoff"] is True
    assert behavior["persistentPollingFallback"] is True
    assert behavior["pollsReadiness"] is True
    assert behavior["securePreviewConsumed"] is True
