# localLLM

FastAPI chat UI backed by Ollama, with optional MCP endpoint discovery and direct MCP tool use by the selected local model.

## Prerequisites

- Ollama running locally
- A chat model pulled into Ollama
- Python 3 with `venv`

## 1) Start Ollama

If Ollama is not already running, start it first:

```bash
ollama serve
```

In another terminal, check which models you have:

```bash
ollama list
```

If you want to use Gemma 4 with the app default, pull the matching Ollama tag:

```bash
ollama pull gemma4:e2b
```

If you do not set `OLLAMA_MODEL`, the app defaults to `gemma4:e2b`.

If you want to use a different Ollama model, point `localLLM` at that exact tag:

```bash
export OLLAMA_MODEL=<your-model-tag>
```

You can also override the Ollama host if needed:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

## 2) Install Python dependencies

From this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## 3) Run the app

From the repo root, with the virtualenv still activated:

```bash
python3 app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 4) Use MCP endpoints

In the left sidebar:

1. Open the `MCP` tab.
2. Add one or more Streamable HTTP MCP endpoints.
3. Click `Refresh MCP tools` to verify the server and inspect available tools.
4. Ask your question in chat.

The app will connect to those MCP servers, discover their tools, and pass them directly to Ollama as callable tools.

## Useful env vars

- `OLLAMA_MODEL`: model tag to send chat/tool-planning requests to. Defaults to `gemma4:e2b`.
- `OLLAMA_BASE_URL`: Ollama base URL, defaults to `http://127.0.0.1:11434`
## Notes

- `localLLM` now uses Ollama, not Apple Foundation Models.
- The configured `OLLAMA_MODEL` must exist in `ollama list` or `/api/status` will report it as unavailable.
- MCP support uses the Python [`mcp`](https://pypi.org/project/mcp/) client and expects Streamable HTTP endpoints.
- When MCP endpoints are configured, the app exposes their tool schemas directly to the Ollama model instead of routing through a custom local planning layer.
