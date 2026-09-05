"""Runtime-safe settings endpoints for the local control room."""

from __future__ import annotations

import json
import os
import tempfile

from fastapi import APIRouter, HTTPException, Request

from backend.config import ROOT, settings

router = APIRouter()
ENV_PATH = ROOT / ".env"
_ALLOWED_BACKENDS = {"developer", "vertex"}
_MAX_API_KEY_LENGTH = 512


def _dotenv_quote(value: str) -> str:
    """Quote a value so special characters cannot alter dotenv parsing."""
    return json.dumps(value, ensure_ascii=False)


def _masked_status() -> dict:
    """Never include the actual API key in an HTTP response."""
    return {"configured": bool(settings.gemini_api_key), "backend": settings.gemini_backend}


@router.get("/api/settings/gemini")
def gemini_settings() -> dict:
    return _masked_status()


@router.post("/api/settings/gemini")
async def update_gemini_settings(request: Request) -> dict:
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(400, "Dữ liệu cấu hình Gemini không hợp lệ.") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "Dữ liệu cấu hình Gemini không hợp lệ.")

    raw_api_key = payload.get("api_key", "")
    raw_backend = payload.get("backend", "developer")
    if not isinstance(raw_api_key, str) or not isinstance(raw_backend, str):
        raise HTTPException(400, "Dữ liệu cấu hình Gemini không hợp lệ.")

    backend = raw_backend.strip().lower()
    if backend not in _ALLOWED_BACKENDS:
        raise HTTPException(400, "Gemini backend phải là developer hoặc vertex.")
    if len(raw_api_key) > _MAX_API_KEY_LENGTH:
        raise HTTPException(400, "Gemini API key không được dài quá 512 ký tự.")
    if any(char in raw_api_key for char in "\r\n"):
        raise HTTPException(400, "Gemini API key không được chứa ký tự xuống dòng.")
    api_key = raw_api_key.strip()
    if not api_key:
        raise HTTPException(400, "Hãy điền Gemini API key.")

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.is_file() else []
    kept = [
        line for line in lines
        if not line.startswith("GEMINI_API_KEY=") and not line.startswith("GEMINI_BACKEND=")
    ]
    kept.extend([
        f"GEMINI_BACKEND={backend}",
        f"GEMINI_API_KEY={_dotenv_quote(api_key)}",
    ])
    content = "\n".join(kept).rstrip() + "\n"

    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".env.", dir=ENV_PATH.parent)
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, ENV_PATH)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    settings.gemini_api_key = api_key
    settings.gemini_backend = backend
    return {"saved": True, **_masked_status()}
