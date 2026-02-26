const messagesEl = document.getElementById('messages');
const composer = document.getElementById('composer');
const promptEl = document.getElementById('prompt');
const statusEl = document.getElementById('status');
const newChatBtn = document.getElementById('new-chat');

let conversationId = window.__CONVERSATION_ID__;

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

function appendMessage(role, text = '') {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  renderMessageContent(el, role, text);
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const body = await res.json();
    statusEl.textContent = body.ok ? 'Model ready' : `Unavailable: ${body.detail}`;
  } catch {
    statusEl.textContent = 'Status unavailable';
  }
}

async function streamAssistantReply(message) {
  const assistantMsg = appendMessage('assistant', '');

  const res = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });

  if (!res.ok || !res.body) {
    assistantMsg.textContent = 'Failed to contact local model.';
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
});

appendMessage('assistant', 'Hi. I am your local on-device assistant.');
checkStatus();
