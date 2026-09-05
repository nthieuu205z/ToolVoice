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
