import asyncio
import logging
import os
import threading
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from backend.routes import voices as voice_routes
from pipeline import custom_voices
from pipeline.audio import write_wav


async def _response_body(response) -> bytes:
    messages = []
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/preview",
        "raw_path": b"/preview",
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("test", 1),
        "server": ("test", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await response(scope, receive, send)
    return b"".join(message.get("body", b"") for message in messages)


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
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == "8"
    assert response.headers["last-modified"]
    assert response.headers["etag"]
    assert response.headers["content-disposition"] == (
        'attachment; filename="vi-VN-HoaiMyNeural.wav"'
    )


def test_voice_catalog_advertises_only_the_secure_preview_endpoint(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    voice_id = "vi-VN-HoaiMyNeural"
    (tmp_path / f"{voice_id}.wav").write_bytes(b"RIFFsafe")

    voice = next(item for item in client.get("/api/voices").json() if item["id"] == voice_id)

    assert voice["preview_url"] == f"/api/voices/{voice_id}/preview"
    assert voice["preview_status"] == "ready"
    assert client.get(f"/previews/{voice_id}.wav").status_code == 404


def test_legacy_preview_path_cannot_fall_through_the_static_mount(client):
    legacy_root = settings.static_dir / "previews"
    legacy_root.mkdir(exist_ok=True)
    planted = legacy_root / "security-regression.wav"
    planted.write_bytes(b"must not be public")
    try:
        response = client.get("/previews/security-regression.wav")
    finally:
        planted.unlink(missing_ok=True)
        try:
            legacy_root.rmdir()
        except OSError:
            pass

    assert response.status_code == 404
    assert response.content != b"must not be public"


def test_voice_preview_endpoint_serves_a_byte_range(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    preview = tmp_path / "vi-VN-HoaiMyNeural.wav"
    preview.write_bytes(b"RIFF0123456789")

    response = client.get(
        "/api/voices/vi-VN-HoaiMyNeural/preview",
        headers={"Range": "bytes=4-7"},
    )

    assert response.status_code == 206
    assert response.content == b"0123"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"] == "bytes 4-7/14"
    assert response.headers["content-length"] == "4"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["content-disposition"] == (
        'attachment; filename="vi-VN-HoaiMyNeural.wav"'
    )


def test_voice_preview_holds_the_validated_file_when_the_path_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    preview = tmp_path / "vi-VN-HoaiMyNeural.wav"
    preview.write_bytes(b"RIFFsafe")
    replacement = tmp_path / "outside.wav"
    replacement.write_bytes(b"RIFFoutside")

    response = voice_routes.preview_voice("vi-VN-HoaiMyNeural")
    try:
        os.replace(replacement, preview)
    except PermissionError:
        # Windows holds the validated file without delete sharing, so replacement is denied.
        pass

    assert asyncio.run(_response_body(response)) == b"RIFFsafe"
    assert response.file.closed is True


def test_voice_preview_rejects_a_symlink_to_another_managed_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "tts_provider", "edge")
    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: tmp_path))
    target = tmp_path / "other.wav"
    target.write_bytes(b"RIFFother voice")
    preview = tmp_path / "vi-VN-HoaiMyNeural.wav"
    try:
        preview.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable on this platform: {exc}")

    response = client.get("/api/voices/vi-VN-HoaiMyNeural/preview")

    assert response.status_code == 404
    assert response.json()["detail"] == "Giọng này chưa có file demo."


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


def test_preview_generation_rejects_a_planted_symlink_and_cleans_temp_files(
    tmp_path, monkeypatch, caplog
):
    previews = tmp_path / "previews"
    previews.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside bytes")
    voice_id = "clone-safe"
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    custom_voices.register(voice_id, "Safe")
    planted = previews / f"{voice_id}.wav"
    try:
        planted.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"Symlinks unavailable on this platform: {exc}")

    class Synthesizer:
        def synthesize(self, text, requested_voice_id):
            return b"\x00\x00" * 2400

    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: previews))
    monkeypatch.setattr(voice_routes, "_clone_synthesizer", Synthesizer)

    with caplog.at_level(logging.WARNING, logger=voice_routes.__name__):
        voice_routes._generate_preview(voice_id)

    assert outside.read_bytes() == b"outside bytes"
    assert planted.is_symlink()
    assert list(previews.iterdir()) == [planted]
    assert "Không tạo được nghe thử" in caplog.text


def test_preview_generation_creates_a_missing_preview_root(tmp_path, monkeypatch):
    previews = tmp_path / "fresh" / "previews"
    voice_id = "clone-fresh"
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    custom_voices.register(voice_id, "Fresh")

    class Synthesizer:
        def synthesize(self, text, requested_voice_id):
            return b"\x00\x00" * 2400

    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: previews))
    monkeypatch.setattr(voice_routes, "_clone_synthesizer", Synthesizer)

    voice_routes._generate_preview(voice_id)

    assert (previews / f"{voice_id}.wav").is_file()
    assert voice_routes.preview_status(voice_id) == {"status": "ready"}


def test_delete_waits_for_generation_then_removes_the_published_preview(
    tmp_path, monkeypatch
):
    previews = tmp_path / "previews"
    voice_id = "clone-race"
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    custom_voices.register(voice_id, "Race")
    synthesis_started = threading.Event()
    allow_synthesis = threading.Event()
    deletion_finished = threading.Event()

    class Synthesizer:
        def synthesize(self, text, requested_voice_id):
            synthesis_started.set()
            assert allow_synthesis.wait(timeout=2)
            return b"\x00\x00" * 2400

    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: previews))
    monkeypatch.setattr(voice_routes, "_clone_synthesizer", Synthesizer)
    monkeypatch.setattr("pipeline.vieneu_speech.forget_clone", lambda voice: None)
    monkeypatch.setattr("pipeline.omnivoice_speech.forget_clone", lambda voice: None)

    generator = threading.Thread(target=voice_routes._generate_preview, args=(voice_id,))
    deleter = threading.Thread(
        target=lambda: (voice_routes.delete_custom_voice(voice_id), deletion_finished.set())
    )
    generator.start()
    assert synthesis_started.wait(timeout=2)
    deleter.start()
    assert deletion_finished.wait(timeout=0.1) is False
    allow_synthesis.set()
    generator.join(timeout=2)
    deleter.join(timeout=2)

    assert deletion_finished.is_set()
    assert custom_voices.get(voice_id) is None
    assert (previews / f"{voice_id}.wav").exists() is False


def test_delete_keeps_voice_retryable_when_preview_handle_blocks_unlink(
    tmp_path, monkeypatch
):
    previews = tmp_path / "previews"
    previews.mkdir()
    voice_id = "clone-busy"
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    custom_voices.register(voice_id, "Busy")
    preview = previews / f"{voice_id}.wav"
    preview.write_bytes(b"RIFFbusy")
    real_unlink = Path.unlink

    def fail_preview_unlink(path, *args, **kwargs):
        if path == preview:
            raise PermissionError("sharing violation")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(settings), "previews_dir", property(lambda self: previews))
    monkeypatch.setattr(Path, "unlink", fail_preview_unlink)

    with pytest.raises(HTTPException) as excinfo:
        voice_routes.delete_custom_voice(voice_id)

    assert excinfo.value.status_code == 409
    assert custom_voices.get(voice_id) is not None
    assert preview.is_file()


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
