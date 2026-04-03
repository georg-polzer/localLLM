const messagesEl = document.getElementById('messages');
const composer = document.getElementById('composer');
const promptEl = document.getElementById('prompt');
const statusEl = document.getElementById('status');
const modelSelectEl = document.getElementById('model-select');
const stopOllamaBtn = document.getElementById('stop-ollama');
const newChatBtn = document.getElementById('new-chat');
const sidebarTabs = Array.from(document.querySelectorAll('.sidebar-tab'));
const sidebarPanels = Array.from(document.querySelectorAll('.sidebar-panel'));
const mcpForm = document.getElementById('mcp-form');
const mcpEndpointEl = document.getElementById('mcp-endpoint');
const mcpListEl = document.getElementById('mcp-list');
const mcpEmptyEl = document.getElementById('mcp-empty');
const refreshMcpBtn = document.getElementById('refresh-mcp');
const mcpStatusEl = document.getElementById('mcp-status');

const MCP_STORAGE_KEY = 'local-llm-mcp-endpoints';

let conversationId = window.__CONVERSATION_ID__;
let mcpEndpoints = loadMcpEndpoints();
let ollamaIsReady = false;
let currentModel = '';

function loadMcpEndpoints() {
  try {
    const raw = localStorage.getItem(MCP_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((value) => typeof value === 'string') : [];
  } catch {
    return [];
  }
}

function saveMcpEndpoints() {
  localStorage.setItem(MCP_STORAGE_KEY, JSON.stringify(mcpEndpoints));
}

function escapeHtml(text) {
  return text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderAssistantMarkdown(text) {
  const escaped = escapeHtml(text).replace(/\r\n/g, '\n');
  const lines = escaped.split('\n');
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (!line.trim()) {
      i += 1;
      continue;
    }

    if (line.startsWith('### ')) {
      blocks.push(`<h3>${line.slice(4)}</h3>`);
      i += 1;
      continue;
    }

    if (/^\s*-\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*-\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*-\s+/, ''));
        i += 1;
      }
      blocks.push(`<ul>${items.map((item) => `<li>${item}</li>`).join('')}</ul>`);
      continue;
    }

    const paragraph = [];
    while (i < lines.length && lines[i].trim() && !lines[i].startsWith('### ') && !/^\s*-\s+/.test(lines[i])) {
      paragraph.push(lines[i]);
      i += 1;
    }
    blocks.push(`<p>${paragraph.join('<br>')}</p>`);
  }

  return blocks.join('');
}

function applyInlineMarkdown(html) {
  return html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function renderMessageContent(el, role, text) {
  if (role === 'assistant') {
    el.innerHTML = applyInlineMarkdown(renderAssistantMarkdown(text));
    return;
  }
  el.textContent = text;
}

function renderAssistantProgress(el, text) {
  el.innerHTML = `<p><em>${escapeHtml(text)}</em></p>`;
}

function appendMessage(role, text = '') {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  renderMessageContent(el, role, text);
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function setActiveSidebarTab(tabName) {
  for (const tab of sidebarTabs) {
    const isActive = tab.dataset.tab === tabName;
    tab.classList.toggle('active', isActive);
    tab.setAttribute('aria-selected', isActive ? 'true' : 'false');
  }

  for (const panel of sidebarPanels) {
    const isActive = panel.dataset.panel === tabName;
    panel.classList.toggle('active', isActive);
    panel.hidden = !isActive;
  }
}

function renderMcpEndpoints() {
  mcpListEl.innerHTML = '';
  mcpEmptyEl.style.display = mcpEndpoints.length ? 'none' : 'block';

  for (const endpoint of mcpEndpoints) {
    const row = document.createElement('div');
    row.className = 'mcp-item';

    const label = document.createElement('div');
    label.className = 'mcp-item-label';
    label.textContent = endpoint;

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'mcp-remove';
    removeBtn.textContent = 'Remove';
    removeBtn.addEventListener('click', () => {
      mcpEndpoints = mcpEndpoints.filter((value) => value !== endpoint);
      saveMcpEndpoints();
      renderMcpEndpoints();
      mcpStatusEl.textContent = 'Endpoint removed.';
    });

    row.append(label, removeBtn);
    mcpListEl.appendChild(row);
  }
}

function updateOllamaButton() {
  stopOllamaBtn.textContent = ollamaIsReady ? 'Stop Ollama server' : 'Start Ollama service';
}

function renderModelOptions(models, selectedModel) {
  modelSelectEl.innerHTML = '';

  if (!models.length) {
    const option = document.createElement('option');
    option.textContent = 'No models available';
    option.value = '';
    modelSelectEl.appendChild(option);
    modelSelectEl.disabled = true;
    return;
  }

  for (const model of models) {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    option.selected = model === selectedModel;
    modelSelectEl.appendChild(option);
  }

  modelSelectEl.disabled = false;
}

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const body = await res.json();
    currentModel = body.current_model || currentModel;
    renderModelOptions(body.models || [], currentModel);
  } catch {
    renderModelOptions([], currentModel);
  }
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const body = await res.json();
    ollamaIsReady = Boolean(body.ok);
    currentModel = body.model || currentModel;
    const modelState = body.ok ? 'Model ready' : `Unavailable: ${body.detail}`;
    const mcpState = body.mcp_available ? 'MCP support installed' : 'MCP support not installed';
    statusEl.textContent = `${modelState} · ${mcpState}`;
  } catch {
    ollamaIsReady = false;
    statusEl.textContent = 'Status unavailable';
  } finally {
    updateOllamaButton();
  }
}

async function stopOllama() {
  stopOllamaBtn.disabled = true;
  stopOllamaBtn.textContent = ollamaIsReady ? 'Stopping server...' : 'Starting service...';

  try {
    const res = await fetch(ollamaIsReady ? '/api/ollama/stop' : '/api/ollama/start', { method: 'POST' });
    const body = await res.json();
    if (body.ok) {
      statusEl.textContent = ollamaIsReady ? 'Ollama stopped' : 'Ollama started';
    } else {
      statusEl.textContent = `${ollamaIsReady ? 'Stop' : 'Start'} failed: ${body.detail}`;
    }
  } catch {
    statusEl.textContent = `${ollamaIsReady ? 'Stop' : 'Start'} failed`;
  } finally {
    stopOllamaBtn.disabled = false;
    setTimeout(checkStatus, 400);
  }
}

async function selectModel() {
  const nextModel = modelSelectEl.value;
  if (!nextModel || nextModel === currentModel) return;

  modelSelectEl.disabled = true;

  try {
    const res = await fetch('/api/models/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: nextModel }),
    });
    const body = await res.json();

    if (!res.ok) {
      statusEl.textContent = body.detail || 'Failed to switch model.';
      modelSelectEl.value = currentModel;
      return;
    }

    currentModel = body.model;
    statusEl.textContent = `Switched to ${currentModel}`;
  } catch {
    statusEl.textContent = 'Failed to switch model.';
    modelSelectEl.value = currentModel;
  } finally {
    await checkStatus();
    await loadModels();
  }
}

async function inspectMcpEndpoints() {
  if (!mcpEndpoints.length) {
    mcpStatusEl.textContent = 'Add an endpoint to preview its MCP tools.';
    return;
  }

  mcpStatusEl.textContent = 'Inspecting MCP endpoints...';

  try {
    const res = await fetch('/api/mcp/inspect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ endpoints: mcpEndpoints }),
    });
    const body = await res.json();

    if (!res.ok) {
      mcpStatusEl.textContent = body.detail || 'Failed to inspect MCP endpoints.';
      return;
    }

    const serverSummaries = body.servers.map((server) => {
      const names = server.tools.slice(0, 5).map((tool) => tool.name).join(', ');
      return `${server.endpoint} (${server.tool_count} tools${names ? `: ${names}` : ''})`;
    });

    mcpStatusEl.textContent = serverSummaries.join(' | ');
  } catch {
    mcpStatusEl.textContent = 'Failed to inspect MCP endpoints.';
  }
}

async function streamAssistantReply(message) {
  const assistantMsg = appendMessage('assistant', '');

  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      mcp_endpoints: mcpEndpoints,
    }),
  });

  if (!res.ok || !res.body) {
    let detail = 'Failed to contact local model.';
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      // Ignore JSON parsing failures here.
    }
    assistantMsg.textContent = detail;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';

    for (const event of events) {
      if (!event.startsWith('data: ')) continue;
      const payload = JSON.parse(event.slice(6));

      if (payload.type === 'meta') {
        conversationId = payload.conversation_id;
        if (payload.mcp_servers?.length) {
          const toolCount = payload.mcp_servers.reduce((sum, server) => sum + server.tool_count, 0);
          mcpStatusEl.textContent = `Active in this chat: ${payload.mcp_servers.length} endpoint(s), ${toolCount} tool(s).`;
        }
      } else if (payload.type === 'progress') {
        renderAssistantProgress(assistantMsg, payload.content);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } else if (payload.type === 'chunk') {
        renderMessageContent(assistantMsg, 'assistant', payload.content);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      } else if (payload.type === 'error') {
        assistantMsg.textContent = `Error: ${payload.content}`;
      }
    }
  }
}

composer.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = promptEl.value.trim();
  if (!text) return;

  appendMessage('user', text);
  promptEl.value = '';
  promptEl.style.height = 'auto';

  await streamAssistantReply(text);
});

promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = `${Math.min(promptEl.scrollHeight, 220)}px`;
});

promptEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

newChatBtn.addEventListener('click', async () => {
  await fetch('/api/chat/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ conversation_id: conversationId }),
  });

  conversationId = crypto.randomUUID();
  messagesEl.innerHTML = '';
  mcpStatusEl.textContent = mcpEndpoints.length
    ? 'New chat started. MCP endpoints will be available on the next message.'
    : 'MCP tools are optional.';
});

stopOllamaBtn.addEventListener('click', stopOllama);
modelSelectEl.addEventListener('change', selectModel);

for (const tab of sidebarTabs) {
  tab.addEventListener('click', () => {
    setActiveSidebarTab(tab.dataset.tab);
  });
}

mcpForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const endpoint = mcpEndpointEl.value.trim().replace(/\/$/, '');
  if (!endpoint) return;

  if (mcpEndpoints.includes(endpoint)) {
    mcpStatusEl.textContent = 'That endpoint is already configured.';
    return;
  }

  mcpEndpoints = [...mcpEndpoints, endpoint];
  saveMcpEndpoints();
  renderMcpEndpoints();
  mcpEndpointEl.value = '';
  mcpStatusEl.textContent = 'Endpoint added. Refresh MCP tools to verify connectivity.';
});

refreshMcpBtn.addEventListener('click', inspectMcpEndpoints);

appendMessage('assistant', 'Hi. I am your local on-device assistant. Add MCP endpoints in the sidebar if you want me to use external tools.');
setActiveSidebarTab('chat');
renderMcpEndpoints();
checkStatus();
loadModels();
