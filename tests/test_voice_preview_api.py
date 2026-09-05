import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from backend.routes import voices as voice_routes
from pipeline import custom_voices
from pipeline.audio import write_wav


@pytest.fixture
def client():
    return TestClient(app)


def test_voice_preview_endpoint_serves_a_known_preview(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    preview = tmp_path / "vi-VN-HoaiMyNeural.wav"
    preview.write_bytes(b"RIFFfake")

    response = client.get("/api/voices/vi-VN-HoaiMyNeural/preview")

    assert response.status_code == 200
    assert response.content == b"RIFFfake"
    assert response.headers["content-type"] == "audio/wav"


def test_voice_preview_endpoint_serves_a_registered_custom_preview(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    voice_id = "clone-safe"
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    custom_voices.register(voice_id, "Safe")
    (tmp_path / f"{voice_id}.wav").write_bytes(b"RIFFcustom")

    response = client.get(f"/api/voices/{voice_id}/preview")

    assert response.status_code == 200
    assert response.content == b"RIFFcustom"


def test_voice_preview_endpoint_rejects_a_known_voice_without_a_preview(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))

    response = client.get("/api/voices/vi-VN-HoaiMyNeural/preview")

    assert response.status_code == 404
    assert response.json()["detail"] == "Giọng này chưa có file demo."


@pytest.mark.parametrize(
    "voice_id",
    ["not-a-real-voice", "clone-not-registered", "clone-../outside", "clone-a/b", "clone-a\\b"],
)
def test_preview_path_rejects_unknown_and_path_like_voice_ids(voice_id):
    with pytest.raises(HTTPException) as excinfo:
        voice_routes._preview_path(voice_id)

    assert excinfo.value.status_code == 404
