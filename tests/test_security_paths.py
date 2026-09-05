from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from backend.routes import jobs as job_routes
from pipeline import custom_voices
from pipeline.audio import write_wav


def _scope(spec_version: str = "2.4", headers: dict[str, str] | None = None) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/download",
        "raw_path": b"/download",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or {}).items()
        ],
        "client": ("test", 1),
        "server": ("test", 80),
    }


async def _response_messages(response, headers: dict[str, str] | None = None) -> list[dict]:
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await response(_scope(headers=headers), receive, send)
    return messages


async def _response_body(response, headers: dict[str, str] | None = None) -> bytes:
    messages = await _response_messages(response, headers)
    return b"".join(message.get("body", b"") for message in messages)


def _response_result(response, headers: dict[str, str] | None = None):
    messages = asyncio.run(_response_messages(response, headers))
    start = next(message for message in messages if message["type"] == "http.response.start")
    response_headers = {
        name.decode("latin-1"): value.decode("latin-1") for name, value in start["headers"]
    }
    body = b"".join(message.get("body", b"") for message in messages)
    return start["status"], response_headers, body


def _voice(voice_id: str = "clone-safe"):
    write_wav(custom_voices.sample_path(voice_id), np.zeros(2400, dtype="<i2"), 24000)
    return custom_voices.register(voice_id, "Safe")


def test_custom_voice_ids_are_restricted_to_clone_slugs():
    assert custom_voices.is_custom("clone-safe") is True
    assert custom_voices.is_custom("clone-../outside") is False
    assert custom_voices.is_custom("clone-a/b") is False
    assert custom_voices.is_custom("clone-a\\b") is False


def test_sample_path_rejects_a_non_clone_id():
    with pytest.raises(ValueError):
        custom_voices.sample_path("clone-../outside")


def test_register_rejects_invalid_id_and_missing_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    with pytest.raises(ValueError):
        custom_voices.register("clone-../outside", "Bad")
    with pytest.raises(FileNotFoundError):
        custom_voices.register("clone-valid", "Missing")


def test_malformed_voice_metadata_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    (tmp_path / "clone-safe.json").write_text(
        json.dumps({"id": "clone-../outside", "display_name": "bad"}),
        encoding="utf-8",
    )
    (tmp_path / "outside.wav").write_bytes(b"not a voice")

    assert custom_voices.list_custom() == []


def test_registered_voice_id_must_match_metadata_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_voices, "_dir", tmp_path)
    (tmp_path / "clone-safe.wav").write_bytes(b"sample")
    (tmp_path / "clone-safe.json").write_text(
        json.dumps({"id": "clone-other", "display_name": "bad"}),
        encoding="utf-8",
    )

    assert custom_voices.list_custom() == []


def _job_class(job_dir: Path, video_path: Path):
    target = str(video_path)

    class Job:
        status = "done"
        workdir = job_dir
        video_path = target
        filename = "video.mp4"

    return Job


def test_download_rejects_a_path_outside_the_job_directory(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, outside)())

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_video("job")

    assert excinfo.value.status_code == 404


def test_download_rejects_a_symlink_inside_job_directory(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    link = job_dir / "output.mp4"
    link.symlink_to(outside)
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, link)())

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_video("job")

    assert excinfo.value.status_code == 404


def test_download_accepts_a_regular_result_file(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"safe")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    response = job_routes.download_video("job")

    assert response.path == video.resolve()
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == "4"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["content-disposition"] == 'attachment; filename="video_vi.mp4"'
    assert response.headers["last-modified"]
    assert response.headers["etag"]
    response.close()


def test_download_holds_the_validated_file_when_the_path_is_replaced(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"safe result")
    replacement = tmp_path / "outside.mp4"
    replacement.write_bytes(b"outside bytes")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    response = job_routes.download_video("job")
    try:
        os.replace(replacement, video)
    except PermissionError:
        # Windows holds the validated file without delete sharing, so replacement is denied.
        pass

    assert asyncio.run(_response_body(response)) == b"safe result"
    assert response.file.closed is True


@pytest.mark.parametrize(
    ("range_header", "expected_body", "expected_content_range"),
    [
        pytest.param("bytes=2-5", b"2345", "bytes 2-5/10", id="bounded"),
        pytest.param("bytes=6-", b"6789", "bytes 6-9/10", id="open-ended"),
        pytest.param("bytes=-4", b"6789", "bytes 6-9/10", id="suffix"),
    ],
)
def test_download_serves_byte_ranges_from_the_held_file(
    tmp_path, monkeypatch, range_header, expected_body, expected_content_range
):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"0123456789")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    status, headers, body = _response_result(
        job_routes.download_video("job"), {"range": range_header}
    )

    assert status == 206
    assert body == expected_body
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-range"] == expected_content_range
    assert headers["content-length"] == "4"


def test_download_rejects_an_unsatisfiable_byte_range(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"0123456789")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    status, headers, body = _response_result(
        job_routes.download_video("job"), {"range": "bytes=10-20"}
    )

    assert status == 416
    assert headers["content-range"] == "bytes */10"
    assert body == b""


def test_download_rejects_a_malformed_byte_range(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"0123456789")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    status, _, _ = _response_result(
        job_routes.download_video("job"), {"range": "items=2-5"}
    )

    assert status == 400


def test_download_honors_a_matching_if_range_validator(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"0123456789")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())
    response = job_routes.download_video("job")

    status, headers, body = _response_result(
        response,
        {"range": "bytes=2-5", "if-range": response.headers["etag"]},
    )

    assert status == 206
    assert headers["content-range"] == "bytes 2-5/10"
    assert body == b"2345"


def test_download_ignores_range_for_a_mismatching_if_range_validator(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"0123456789")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    status, headers, body = _response_result(
        job_routes.download_video("job"),
        {"range": "bytes=2-5", "if-range": '"different-etag"'},
    )

    assert status == 200
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-length"] == "10"
    assert "content-range" not in headers
    assert body == b"0123456789"


def test_download_closes_the_held_file_when_sending_raises(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"safe result")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())
    response = job_routes.download_video("job")

    async def fail_send():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.body":
                raise RuntimeError("send failed")

        await response(_scope(), receive, send)

    with pytest.raises(RuntimeError, match="send failed"):
        asyncio.run(fail_send())
    assert response.file.closed is True


def test_download_closes_the_held_file_on_disconnect(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = job_dir / "output.mp4"
    video.write_bytes(b"safe result")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())
    response = job_routes.download_video("job")

    async def disconnect():
        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        await response(_scope("2.3"), receive, send)

    asyncio.run(disconnect())
    assert response.file.closed is True


def test_download_rejects_a_nested_result_file(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    nested = job_dir / "nested"
    nested.mkdir(parents=True)
    video = nested / "output.mp4"
    video.write_bytes(b"safe")
    monkeypatch.setattr(job_routes, "_require_done", lambda job_id: _job_class(job_dir, video)())

    with pytest.raises(HTTPException) as excinfo:
        job_routes.download_video("job")

    assert excinfo.value.status_code == 404
