"""Minimal HTTP adapter for ChatKodi 1.9 generation scenarios.

Run from the repository root:
    pip install -r requirements-web.txt
    uvicorn integrations.fastapi_example:app --host 127.0.0.1 --port 8080

For production, replace ``build_llm`` with the site's model provider and add
authentication, rate limits and a persistent session store.
"""

from __future__ import annotations

import re
from pathlib import Path
from threading import Lock
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.config import get_config
from core.generation.service import GenerationService


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)


class SaveRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    artifact_id: str = Field(min_length=1, max_length=128)


def build_llm():
    """Return a compatible local/provider model here; None is honest offline mode."""
    return None


app = FastAPI(title="ChatKodi API", version="2.0.0")
_services: Dict[str, GenerationService] = {}
_lock = Lock()
_downloads = Path("runtime") / "web_downloads"


def _safe_session_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", value)[:128]


def _service(session_id: str) -> GenerationService:
    key = _safe_session_id(session_id)
    with _lock:
        if key not in _services:
            config = get_config().data.get("generation", {})
            _services[key] = GenerationService(build_llm(), config)
        return _services[key]


@app.get("/health")
def health():
    return {"ok": True, "version": "2.0.0"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    result = _service(request.session_id).handle(request.message)
    if result is None:
        raise HTTPException(422, "Request was not routed to a generation scenario")
    artifact = result.artifact
    return {
        "text": result.text,
        "generation_kind": result.generation_kind.value,
        "sources": result.sources,
        "clarification_fields": result.clarification_fields,
        "artifact": None if artifact is None else {
            "id": artifact.artifact_id,
            "suggested_name": artifact.suggested_name,
            "can_save": artifact.can_save,
            "validation_errors": artifact.validation_errors,
        },
    }


@app.post("/api/artifacts/save")
def save_artifact(request: SaveRequest):
    service = _service(request.session_id)
    artifact = service.artifacts.get(request.artifact_id)
    if artifact is None:
        raise HTTPException(404, "Artifact was not found in this session")
    if not artifact.can_save:
        raise HTTPException(409, "Artifact did not pass validation")
    destination = _downloads / _safe_session_id(request.session_id) / artifact.suggested_name
    saved = service.save_artifact(request.artifact_id, destination)
    return FileResponse(
        saved,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=saved.name,
    )
