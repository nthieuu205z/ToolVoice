import re

import pytest
from fastapi.testclient import TestClient

from backend.job_manager import manager
from backend.main import app


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
