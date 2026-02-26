import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

try:
    import apple_fm_sdk as fm
except ImportError:  # pragma: no cover
    fm = None


BASE_DIR = Path(__file__).parent
app = FastAPI(title="Local LLM Chat")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    system_prompt: str | None = None


class ResetRequest(BaseModel):
    conversation_id: str


class SessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, Any] = {}

    async def get_or_create(self, conversation_id: str, system_prompt: str | None = None):
        async with self._lock:
            if conversation_id in self._sessions:
                return self._sessions[conversation_id]

            if fm is None:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "python-apple-fm-sdk is not installed. "
                        "Install it from https://github.com/apple/python-apple-fm-sdk"
                    ),
                )

            model = fm.SystemLanguageModel()
            available, reason = model.is_available()
            if not available:
                raise HTTPException(status_code=503, detail=f"Model unavailable: {reason}")

            kwargs: dict[str, Any] = {"model": model}
            if system_prompt:
                kwargs["instructions"] = system_prompt
            session = fm.LanguageModelSession(**kwargs)
            self._sessions[conversation_id] = session
            return session

    async def reset(self, conversation_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(conversation_id, None) is not None


store = SessionStore()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"conversation_id": str(uuid4())},
    )


@app.get("/api/status")
async def status() -> JSONResponse:
    if fm is None:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "detail": "python-apple-fm-sdk not installed"},
        )
    model = fm.SystemLanguageModel()
    available, reason = model.is_available()
    return JSONResponse(content={"ok": bool(available), "detail": reason})


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest):
    conversation_id = payload.conversation_id or str(uuid4())
    session = await store.get_or_create(conversation_id, payload.system_prompt)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id})}\n\n"
        try:
            async for snapshot in session.stream_response(payload.message):
                # SDK emits full snapshots; send snapshot verbatim.
                yield f"data: {json.dumps({'type': 'chunk', 'content': snapshot})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:  # pragma: no cover
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/reset")
async def reset_chat(payload: ResetRequest) -> JSONResponse:
    removed = await store.reset(payload.conversation_id)
    return JSONResponse(content={"ok": True, "removed": removed})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
