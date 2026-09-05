import re
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
