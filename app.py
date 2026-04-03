import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    import mcp.types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover
    mcp_types = None
    ClientSession = None
    streamable_http_client = None


BASE_DIR = Path(__file__).parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("localLLM")
app = FastAPI(title="Local LLM Chat")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")
CURRENT_OLLAMA_MODEL = DEFAULT_OLLAMA_MODEL
TOOL_LOOP_MAX_STEPS = 8


def normalize_endpoint(endpoint: str) -> str:
    return endpoint.strip().rstrip("/")


def make_config_signature(system_prompt: str | None, mcp_endpoints: list[str]) -> str:
    normalized_endpoints = [normalize_endpoint(endpoint) for endpoint in mcp_endpoints]
    return json.dumps(
        {
            "system_prompt": system_prompt or "",
            "mcp_endpoints": normalized_endpoints,
        },
        sort_keys=True,
    )


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def parse_json_object(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    if not candidate:
        return {}

    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to a JSON object.")
    return parsed


def normalize_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return parse_json_object(raw)
    raise ValueError("Tool arguments must be a JSON object or JSON-encoded object string.")


def render_tool_content(block: Any) -> str:
    if isinstance(block, getattr(mcp_types, "TextContent", ())):
        return block.text

    if isinstance(block, getattr(mcp_types, "ImageContent", ())):
        return f"[Image content omitted: {block.mimeType}]"

    if isinstance(block, getattr(mcp_types, "AudioContent", ())):
        return f"[Audio content omitted: {block.mimeType}]"

    if isinstance(block, getattr(mcp_types, "EmbeddedResource", ())):
        resource = block.resource
        text = getattr(resource, "text", None)
        if text:
            return f"[Embedded resource]\n{text}"
        blob = getattr(resource, "blob", None)
        mime_type = getattr(resource, "mimeType", "application/octet-stream")
        return f"[Embedded resource omitted: {mime_type}, {len(blob or '')} bytes]"

    if isinstance(block, getattr(mcp_types, "ResourceLink", ())):
        label = block.title or block.name
        return f"[Resource link] {label}: {block.uri}"

    return compact_json(block.model_dump(mode="json"))


async def list_all_tools(client: "ClientSession") -> list["mcp_types.Tool"]:
    tools: list[mcp_types.Tool] = []
    cursor: str | None = None

    while True:
        response = await client.list_tools(cursor=cursor) if cursor else await client.list_tools()
        tools.extend(response.tools)
        if not response.nextCursor:
            break
        cursor = response.nextCursor

    return tools


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    system_prompt: str | None = None
    mcp_endpoints: list[str] = Field(default_factory=list)


class ResetRequest(BaseModel):
    conversation_id: str


class MCPInspectRequest(BaseModel):
    endpoints: list[str] = Field(default_factory=list)


class ModelSelectRequest(BaseModel):
    model: str = Field(..., min_length=1)


@dataclass
class EndpointConnection:
    endpoint: str
    client: "ClientSession"
    tools: list["mcp_types.Tool"]
    lock: asyncio.Lock


@dataclass
class ManagedSession:
    signature: str
    history: list[dict[str, str]]
    mcp_bridge: "MCPBridge | None" = None

    async def close(self) -> None:
        if self.mcp_bridge is not None:
            await self.mcp_bridge.close()


class MCPBridge:
    def __init__(self, stack: contextlib.AsyncExitStack, endpoints: list[EndpointConnection]) -> None:
        self._stack = stack
        self._endpoints = {connection.endpoint: connection for connection in endpoints}

    @classmethod
    async def create(cls, endpoints: list[str]) -> "MCPBridge":
        if streamable_http_client is None or ClientSession is None or mcp_types is None:
            raise HTTPException(
                status_code=500,
                detail="MCP support is unavailable. Install the 'mcp' Python package first.",
            )

        normalized = [normalize_endpoint(endpoint) for endpoint in endpoints if normalize_endpoint(endpoint)]
        if not normalized:
            raise HTTPException(status_code=400, detail="At least one MCP endpoint is required.")

        stack = contextlib.AsyncExitStack()
        connections: list[EndpointConnection] = []

        try:
            for endpoint in normalized:
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(endpoint)
                )
                client = ClientSession(read_stream, write_stream)
                await stack.enter_async_context(client)
                await client.initialize()
                tools = await list_all_tools(client)
                connections.append(
                    EndpointConnection(
                        endpoint=endpoint,
                        client=client,
                        tools=tools,
                        lock=asyncio.Lock(),
                    )
                )
            return cls(stack, connections)
        except Exception:
            await stack.aclose()
            raise

    def summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for connection in self._endpoints.values():
            summaries.append(
                {
                    "endpoint": connection.endpoint,
                    "tool_count": len(connection.tools),
                    "tools": [
                        {
                            "name": tool.name,
                            "title": tool.title,
                            "description": tool.description,
                        }
                        for tool in connection.tools
                    ],
                }
            )
        return summaries

    def tool_catalog(self) -> list[dict[str, Any]]:
        catalog: list[dict[str, Any]] = []
        for connection in self._endpoints.values():
            for tool in connection.tools:
                catalog.append(
                    {
                        "endpoint": connection.endpoint,
                        "name": tool.name,
                        "title": tool.title,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                )
        return catalog

    def endpoint_for_tool(self, tool_name: str) -> str | None:
        for connection in self._endpoints.values():
            if any(tool.name == tool_name for tool in connection.tools):
                return connection.endpoint
        return None

    async def call_tool(self, endpoint: str, name: str, arguments: dict[str, Any]) -> str:
        connection = self._endpoints[endpoint]

        async with connection.lock:
            result = await connection.client.call_tool(name=name, arguments=arguments)

        parts: list[str] = []
        if result.structuredContent is not None:
            parts.append("Structured result:\n" + compact_json(result.structuredContent))
        elif result.content:
            rendered = [render_tool_content(block) for block in result.content]
            parts.append("\n\n".join(part for part in rendered if part))

        if not parts:
            parts.append("Tool completed without returning content.")

        final = "\n\n".join(parts)
        if result.isError:
            return f"Tool error from {name}:\n{final}"
        return final

    async def close(self) -> None:
        await self._stack.aclose()


class SessionStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, ManagedSession] = {}

    async def get_or_create(
        self,
        conversation_id: str,
        system_prompt: str | None = None,
        mcp_endpoints: list[str] | None = None,
    ) -> ManagedSession:
        normalized_endpoints = [normalize_endpoint(endpoint) for endpoint in (mcp_endpoints or []) if normalize_endpoint(endpoint)]
        signature = make_config_signature(system_prompt, normalized_endpoints)

        async with self._lock:
            existing = self._sessions.get(conversation_id)
            if existing and existing.signature == signature:
                return existing

            mcp_bridge = None
            if normalized_endpoints:
                mcp_bridge = await MCPBridge.create(normalized_endpoints)
            managed = ManagedSession(signature=signature, history=[], mcp_bridge=mcp_bridge)

            if existing is not None:
                await existing.close()

            self._sessions[conversation_id] = managed
            return managed

    async def reset(self, conversation_id: str) -> bool:
        async with self._lock:
            managed = self._sessions.pop(conversation_id, None)

        if managed is None:
            return False

        await managed.close()
        return True

    async def close_all(self) -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for managed in sessions:
            await managed.close()


store = SessionStore()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await store.close_all()


def _ollama_request(path: str, payload: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        f"{OLLAMA_BASE_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _ollama_get(path: str) -> urllib.request.Request:
    return urllib.request.Request(f"{OLLAMA_BASE_URL}{path}", method="GET")


def current_ollama_model() -> str:
    return CURRENT_OLLAMA_MODEL


def set_current_ollama_model(model: str) -> None:
    global CURRENT_OLLAMA_MODEL
    CURRENT_OLLAMA_MODEL = model


def fetch_ollama_models() -> list[str]:
    with urllib.request.urlopen(_ollama_get("/api/tags"), timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [model.get("name") for model in payload.get("models", []) if model.get("name")]


async def ollama_status() -> tuple[bool, str]:
    def _fetch() -> tuple[bool, str]:
        try:
            names = set(fetch_ollama_models())
        except urllib.error.URLError as exc:
            return False, f"Ollama unavailable: {exc.reason}"
        except Exception as exc:  # pragma: no cover
            return False, f"Ollama unavailable: {exc}"

        model = current_ollama_model()
        if model in names:
            return True, f"Ollama ready ({model})"
        return False, f"Ollama is running but model '{model}' is not installed"

    return await asyncio.to_thread(_fetch)


async def stop_ollama_service() -> tuple[bool, str]:
    def _stop() -> tuple[bool, str]:
        brew_result = subprocess.run(
            ["brew", "services", "stop", "ollama"],
            capture_output=True,
            text=True,
            check=False,
        )
        if brew_result.returncode == 0:
            detail = (brew_result.stdout or brew_result.stderr).strip() or "Ollama service stopped."
            return True, detail

        pkill_result = subprocess.run(
            ["pkill", "-f", "ollama serve"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pkill_result.returncode == 0:
            return True, "Stopped Ollama process."

        if pkill_result.returncode == 1:
            return True, "Ollama was not running."

        detail = (brew_result.stderr or brew_result.stdout or pkill_result.stderr or pkill_result.stdout).strip()
        return False, detail or "Unable to stop Ollama."

    return await asyncio.to_thread(_stop)


async def start_ollama_service() -> tuple[bool, str]:
    def _start() -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(_ollama_get("/api/tags"), timeout=2):
                return True, "Ollama is already running."
        except Exception:
            pass

        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        for _ in range(10):
            try:
                with urllib.request.urlopen(_ollama_get("/api/tags"), timeout=2):
                    return True, "Started Ollama service."
            except Exception:
                pass
            import time

            time.sleep(0.5)

        return False, f"Started Ollama process {process.pid}, but the server did not become ready in time."

    return await asyncio.to_thread(_start)


async def ollama_stream_chat(messages: list[dict[str, str]]):
    def _iter_lines():
        request = _ollama_request(
            "/api/chat",
            {"model": current_ollama_model(), "messages": messages, "stream": True},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield line

    iterator = await asyncio.to_thread(lambda: iter(_iter_lines()))
    sentinel = object()

    def _next_or_sentinel():
        return next(iterator, sentinel)

    while True:
        line = await asyncio.to_thread(_next_or_sentinel)
        if line is sentinel:
            break
        yield json.loads(line)


async def ollama_chat_once(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def _fetch() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": current_ollama_model(),
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = tools
        request = _ollama_request("/api/chat", payload)
        with urllib.request.urlopen(request, timeout=240) as response:
            body = json.loads(response.read().decode("utf-8"))
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama returned an invalid chat response.")
        return message

    return await asyncio.to_thread(_fetch)


def build_ollama_tools(tool_catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in tool_catalog:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


async def chat_with_mcp_tools(
    messages: list[dict[str, Any]],
    bridge: MCPBridge,
    conversation_id: str,
    on_progress: Any | None = None,
) -> str:
    async def report_progress(message: str) -> None:
        if on_progress is None:
            return
        maybe = on_progress(message)
        if asyncio.iscoroutine(maybe):
            await maybe

    ollama_tools = build_ollama_tools(bridge.tool_catalog())
    await report_progress("Letting the model choose from the connected MCP tools...")

    for _ in range(TOOL_LOOP_MAX_STEPS):
        assistant_message = await ollama_chat_once(messages, tools=ollama_tools)
        logger.info(
            "model=%s conversation_id=%s Received assistant message from Ollama: %s",
            current_ollama_model(),
            conversation_id,
            json.dumps(assistant_message, ensure_ascii=True),
        )
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls:
            return str(assistant_message.get("content") or "")

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            tool_name = function.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                continue

            endpoint = bridge.endpoint_for_tool(tool_name)
            if endpoint is None:
                tool_result = f"Tool error from {tool_name}: tool is not available on any configured endpoint."
            else:
                await report_progress(f"Running MCP tool {tool_name}...")
                try:
                    arguments = normalize_tool_arguments(function.get("arguments"))
                    logger.info(
                        "model=%s conversation_id=%s Executing MCP tool call: %s",
                        current_ollama_model(),
                        conversation_id,
                        json.dumps(
                            {
                                "tool_name": tool_name,
                                "arguments": arguments,
                            },
                            ensure_ascii=True,
                        ),
                    )
                    tool_result = await bridge.call_tool(endpoint, tool_name, arguments)
                except Exception as exc:
                    tool_result = f"Tool error from {tool_name}: {exc}"
                    await report_progress(f"{tool_name} returned an error. Letting the model recover...")

            tool_message = {
                "role": "tool",
                "name": tool_name,
                "tool_name": tool_name,
                "content": tool_result[:12000],
            }
            logger.info(
                "model=%s conversation_id=%s Sending tool message back to Ollama: %s",
                current_ollama_model(),
                conversation_id,
                json.dumps(tool_message, ensure_ascii=True),
            )
            messages.append(tool_message)

        await report_progress("Tool results received. Asking the model to continue...")

    raise RuntimeError("The model did not finish after the maximum number of tool rounds.")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"conversation_id": str(uuid4())},
    )


@app.get("/api/status")
async def status() -> JSONResponse:
    available, detail = await ollama_status()
    return JSONResponse(
        content={
            "ok": bool(available),
            "detail": detail,
            "mcp_available": mcp_types is not None,
            "model": current_ollama_model(),
        }
    )


@app.get("/api/models")
async def list_models() -> JSONResponse:
    try:
        models = await asyncio.to_thread(fetch_ollama_models)
        return JSONResponse(content={"ok": True, "models": models, "current_model": current_ollama_model()})
    except urllib.error.URLError as exc:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "detail": f"Ollama unavailable: {exc.reason}",
                "models": [],
                "current_model": current_ollama_model(),
            },
        )
    except Exception as exc:  # pragma: no cover
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "detail": f"Ollama unavailable: {exc}",
                "models": [],
                "current_model": current_ollama_model(),
            },
        )


@app.post("/api/models/select")
async def select_model(payload: ModelSelectRequest) -> JSONResponse:
    model = payload.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model name is required.")

    try:
        models = await asyncio.to_thread(fetch_ollama_models)
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama unavailable: {exc.reason}") from exc

    if model not in models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' is not installed in Ollama.")

    set_current_ollama_model(model)
    return JSONResponse(content={"ok": True, "model": model})


@app.post("/api/ollama/stop")
async def stop_ollama() -> JSONResponse:
    ok, detail = await stop_ollama_service()
    status_code = 200 if ok else 500
    return JSONResponse(status_code=status_code, content={"ok": ok, "detail": detail})


@app.post("/api/ollama/start")
async def start_ollama() -> JSONResponse:
    ok, detail = await start_ollama_service()
    status_code = 200 if ok else 500
    return JSONResponse(status_code=status_code, content={"ok": ok, "detail": detail})


@app.post("/api/mcp/inspect")
async def inspect_mcp(payload: MCPInspectRequest) -> JSONResponse:
    bridge = await MCPBridge.create(payload.endpoints)
    try:
        return JSONResponse(content={"ok": True, "servers": bridge.summaries()})
    finally:
        await bridge.close()


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest):
    conversation_id = payload.conversation_id or str(uuid4())
    managed = await store.get_or_create(
        conversation_id,
        payload.system_prompt,
        payload.mcp_endpoints,
    )

    async def event_stream():
        meta: dict[str, Any] = {"type": "meta", "conversation_id": conversation_id}
        if managed.mcp_bridge is not None:
            meta["mcp_servers"] = managed.mcp_bridge.summaries()
        yield f"data: {json.dumps(meta)}\n\n"
        try:
            if managed.mcp_bridge is not None:
                progress_queue: asyncio.Queue[str] = asyncio.Queue()

                async def on_progress(message: str) -> None:
                    await progress_queue.put(message)

                messages: list[dict[str, Any]] = []
                if payload.system_prompt:
                    messages.append({"role": "system", "content": payload.system_prompt})
                messages.extend(managed.history)
                messages.append({"role": "user", "content": payload.message})

                task = asyncio.create_task(
                    chat_with_mcp_tools(
                        messages,
                        managed.mcp_bridge,
                        conversation_id,
                        on_progress=on_progress,
                    )
                )
                while True:
                    if task.done() and progress_queue.empty():
                        break
                    try:
                        progress = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
                        yield f"data: {json.dumps({'type': 'progress', 'content': progress})}\n\n"
                    except asyncio.TimeoutError:
                        pass
                analytics_answer = await task
            else:
                analytics_answer = None
            if analytics_answer is not None:
                yield f"data: {json.dumps({'type': 'chunk', 'content': analytics_answer})}\n\n"
                managed.history.append({"role": "user", "content": payload.message})
                managed.history.append({"role": "assistant", "content": analytics_answer})
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            messages: list[dict[str, str]] = []
            if payload.system_prompt:
                messages.append({"role": "system", "content": payload.system_prompt})
            messages.extend(managed.history)
            messages.append({"role": "user", "content": payload.message})

            response_text = ""
            async for event in ollama_stream_chat(messages):
                chunk = event.get("message", {}).get("content", "")
                if chunk:
                    response_text += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': response_text})}\n\n"

                if event.get("done"):
                    managed.history.append({"role": "user", "content": payload.message})
                    managed.history.append({"role": "assistant", "content": response_text})
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
