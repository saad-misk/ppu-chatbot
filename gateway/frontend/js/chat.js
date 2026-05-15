/**
 * chat.js — Main chat page logic.
 * Depends on: js/api.js, js/i18n.js (loaded as ES modules via index.html)
 */
import { apiFetch, getToken, clearToken } from './api.js';
import { t, getLang, setLang } from './i18n.js';

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  currentUser:  null,
  sessionId:    localStorage.getItem('guest_session_id') || null,
  sessions:     [],
  sending:      false,
};

// ─── Utilities ────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/\n/g, '<br>');
}

function formatTime() {
  return new Date().toLocaleTimeString(getLang() === 'ar' ? 'ar' : 'en', {
    hour: '2-digit',
    minute: '2-digit',
  });
}

function $(id) {
  return document.getElementById(id);
}

// ─── User Panel ───────────────────────────────────────────────────────────────
async function loadCurrentUser() {
  if (!getToken()) {
    renderUserPanel();
    return;
  }
  try {
    state.currentUser = await apiFetch('/api/auth/me');
  } catch {
    state.currentUser = null;
  }
  renderUserPanel();
}

function renderUserPanel() {
  const user = state.currentUser;

  $('user-name').textContent   = user ? (user.full_name || user.email || t('guestName')) : t('guestName');
  $('user-role').textContent   = user ? (user.is_admin ? t('adminRole') : t('userRole')) : t('guestRole');
  $('user-avatar').textContent = user
    ? (user.is_admin ? '🛠' : (user.full_name?.[0]?.toUpperCase() || 'U'))
    : '👤';

  const adminBtn    = $('admin-btn');
  const loginArea   = $('login-prompt-area');
  const logoutBtn   = $('logout-btn');

  if (adminBtn)  adminBtn.classList.toggle('hidden', !user?.is_admin);
  if (loginArea) loginArea.classList.toggle('hidden', !!user);
  if (logoutBtn) logoutBtn.classList.toggle('hidden', !user);
}

// ─── Sessions ─────────────────────────────────────────────────────────────────
async function loadSessions() {
  if (!getToken()) {
    if (state.sessionId) {
      try {
        await loadSessionHistory(state.sessionId);
      } catch {
        state.sessionId = null;
        localStorage.removeItem('guest_session_id');
        await startNewSession();
      }
    } else {
      await startNewSession();
    }
    return;
  }

  try {
    const data = await apiFetch('/api/sessions');
    state.sessions = (data.sessions || []).map(s => ({
      id:      s.session_id,
      preview: s.preview || t('untitled'),
    }));
    renderSessions();

    if (state.sessions.length) {
      state.sessionId = state.sessions[0].id;
      setActiveSession(state.sessionId);
      await loadSessionHistory(state.sessionId);
    } else {
      await startNewSession();
    }
  } catch {
    await startNewSession();
  }
}

async function startNewSession() {
  const data = await apiFetch('/api/sessions/new?channel=web', { method: 'POST' });
  state.sessionId = data.session_id;

  if (!getToken()) {
    localStorage.setItem('guest_session_id', state.sessionId);
  } else {
    state.sessions.unshift({ id: state.sessionId, preview: t('untitled') });
    renderSessions();
  }

  setActiveSession(state.sessionId);
  showWelcome();
  updateSendButton();
}

function renderSessions() {
  const container = $('sessions-list');
  container.innerHTML = '';

  state.sessions.forEach(session => {
    const wrap = document.createElement('div');
    wrap.className = 'session-item-wrap';

    const item = document.createElement('button');
    item.type        = 'button';
    item.className   = 'session-item';
    item.dataset.id  = session.id;
    item.textContent = session.preview;
    item.addEventListener('click',   () => switchSession(session.id));
    item.addEventListener('dblclick', e => {
      e.stopPropagation();
      startRename(session.id, item, session.preview);
    });

    const del = document.createElement('button');
    del.type        = 'button';
    del.className   = 'session-delete-btn';
    del.title       = t('deleteConfirm');
    del.textContent = '✕';
    del.addEventListener('click', e => {
      e.stopPropagation();
      if (confirm(t('deleteConfirm'))) deleteSession(session.id);
    });

    wrap.append(item, del);
    container.appendChild(wrap);
  });
}

function startRename(id, itemEl, currentPreview) {
  const wrap  = itemEl.parentElement;
  const input = document.createElement('input');
  input.type      = 'text';
  input.className = 'session-rename-input';
  input.value     = currentPreview;
  wrap.replaceChild(input, itemEl);
  input.focus();
  input.select();

  async function commit() {
    const newName = input.value.trim() || currentPreview;
    const session = state.sessions.find(s => s.id === id);
    if (session) session.preview = newName;
    try {
      await apiFetch(`/api/sessions/${id}/rename`, {
        method: 'PATCH',
        body: JSON.stringify({ preview: newName }),
      });
    } catch { /* best-effort */ }
    renderSessions();
    setActiveSession(state.sessionId);
  }

  input.addEventListener('blur', commit);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { e.preventDefault(); input.blur(); }
    if (e.key === 'Escape') {
      const s = state.sessions.find(s => s.id === id);
      if (s) s.preview = currentPreview;
      renderSessions();
      setActiveSession(state.sessionId);
    }
  });
}

async function deleteSession(id) {
  try {
    await apiFetch(`/api/sessions/${id}`, { method: 'DELETE' });
    state.sessions = state.sessions.filter(s => s.id !== id);

    if (state.sessionId === id) {
      if (state.sessions.length) {
        await switchSession(state.sessions[0].id);
      } else {
        await startNewSession();
      }
    } else {
      renderSessions();
      setActiveSession(state.sessionId);
    }
  } catch {
    alert('Failed to delete session. Please try again.');
  }
}

function setActiveSession(id) {
  document.querySelectorAll('.session-item').forEach(item => {
    item.classList.toggle('active', item.dataset.id === id);
  });
}

async function switchSession(id) {
  state.sessionId = id;
  setActiveSession(id);
  await loadSessionHistory(id);
  updateSendButton();
}

async function loadSessionHistory(id) {
  try {
    const data = await apiFetch(`/api/chat/history/${id}`);
    clearMessages();
    const turns = data.turns || [];
    if (!turns.length) { showWelcome(); return; }
    turns.forEach(turn =>
      appendMessage(turn.role === 'assistant' ? 'bot' : 'user', turn.content, turn.id)
    );
  } catch {
    showWelcome();
  }
}

// ─── Welcome Screen ───────────────────────────────────────────────────────────
function showWelcome() {
  const messages = $('messages');
  messages.innerHTML = `
    <div id="welcome">
      <img src="static/images/ppu-logo.png" alt="PPU Logo" class="welcome-logo" />
      <div class="welcome-title">${escapeHtml(t('welcomeTitle'))}</div>
      <div class="welcome-sub">${escapeHtml(t('welcomeSub'))}</div>
      <div class="suggestion-chips">
        ${t('suggestions')
          .map(text => `<button class="suggestion-chip" type="button">${escapeHtml(text)}</button>`)
          .join('')}
      </div>
    </div>
  `;

  messages.querySelectorAll('.suggestion-chip').forEach(btn => {
    btn.addEventListener('click', () => sendMessage(btn.textContent));
  });
}

function clearMessages() {
  $('messages').innerHTML = '';
}

function removeWelcome() {
  $('welcome')?.remove();
}

// ─── Messaging ────────────────────────────────────────────────────────────────
async function sendMessage(text) {
  const message = text.trim();
  if (!message || state.sending) return;

  if (!state.sessionId) await startNewSession();

  state.sending = true;
  updateSendButton();
  removeWelcome();
  appendMessage('user', message);
  updateSessionPreview(message);

  const typingId = appendTyping();
  try {
    const params = new URLSearchParams({ session_id: state.sessionId, message });
    const data   = await apiFetch(`/api/chat/message?${params}`, { method: 'POST' });
    removeTyping(typingId);
    if (data.low_confidence) appendWarning(t('lowConf'));
    appendMessage('bot', data.reply || '', data.turn_id, data.sources || []);
  } catch {
    removeTyping(typingId);
    appendMessage('bot', t('connectionError'));
  } finally {
    state.sending = false;
    updateSendButton();
  }
}

function updateSessionPreview(message) {
  const session = state.sessions.find(s => s.id === state.sessionId);
  if (!session) return;
  const defaultPreviews = [t('untitled'), 'New Chat', 'محادثة جديدة'];
  if (defaultPreviews.includes(session.preview)) {
    session.preview = message.slice(0, 36) + (message.length > 36 ? '…' : '');
    renderSessions();
    setActiveSession(state.sessionId);
  }
}

function appendMessage(role, text, turnId = null, sources = []) {
  const container = $('messages');
  const row       = document.createElement('div');
  row.className   = `message-row ${role}`;

  const sourcesHtml = sources.length
    ? `<div class="msg-sources">${sources
        .map(s => {
          const page = s.page ? ` · p.${escapeHtml(String(s.page))}` : '';
          return `<span class="source-chip">📄 ${escapeHtml(s.doc_name || 'source')}${page}</span>`;
        })
        .join('')}</div>`
    : '';

  const feedbackHtml = role === 'bot' && turnId
    ? `<div class="feedback-btns">
         <button class="fb-btn" type="button" data-rating="up"   data-turn="${escapeHtml(turnId)}" title="Helpful">👍</button>
         <button class="fb-btn" type="button" data-rating="down" data-turn="${escapeHtml(turnId)}" title="Not helpful">👎</button>
       </div>`
    : '';

  const avatarLabel = role === 'bot' ? 'B' : 'U';

  row.innerHTML = `
    <div class="msg-avatar">${avatarLabel}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escapeHtml(text)}</div>
      ${sourcesHtml}
      <div class="msg-meta">
        <span class="msg-time">${formatTime()}</span>
        ${feedbackHtml}
      </div>
    </div>
  `;

  row.querySelectorAll('.fb-btn').forEach(btn => {
    btn.addEventListener('click', () => submitFeedback(btn.dataset.turn, btn.dataset.rating, btn));
  });

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function appendWarning(text) {
  const container = $('messages');
  const w = document.createElement('div');
  w.className   = 'confidence-warning';
  w.textContent = `⚠️ ${text}`;
  container.appendChild(w);
  container.scrollTop = container.scrollHeight;
}

let _typingCounter = 0;

function appendTyping() {
  const id        = `typing-${_typingCounter++}`;
  const container = $('messages');
  const row       = document.createElement('div');
  row.className   = 'message-row bot';
  row.id          = id;
  row.innerHTML   = `
    <div class="msg-avatar">B</div>
    <div class="msg-content">
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTyping(id) {
  $(id)?.remove();
}

async function submitFeedback(turnId, rating, clickedBtn) {
  try {
    await apiFetch('/api/chat/feedback', {
      method: 'POST',
      body: JSON.stringify({
        session_id: state.sessionId,
        message_id: turnId,
        rating,
      }),
    });
    // Visual confirmation
    const allBtns = document.querySelectorAll(`.fb-btn[data-turn="${turnId}"]`);
    allBtns.forEach(btn => btn.classList.remove('active-up', 'active-down'));
    clickedBtn.classList.add(rating === 'up' ? 'active-up' : 'active-down');
  } catch { /* best-effort */ }
}

// ─── Input Handling ───────────────────────────────────────────────────────────
function handleSend() {
  const input = $('msg-input');
  const text  = input.value.trim();
  if (!text) return;
  input.value       = '';
  input.style.height = 'auto';
  updateSendButton();
  sendMessage(text);
}

function updateSendButton() {
  const input   = $('msg-input');
  const sendBtn = $('send-btn');
  sendBtn.disabled = state.sending || !input.value.trim();
}

// ─── Language ─────────────────────────────────────────────────────────────────
function applyLangToPage() {
  const newChatBtn     = $('new-chat-btn');
  const sessionsLabel  = $('sessions-label');
  const botStatus      = $('bot-status');
  const msgInput       = $('msg-input');
  const inputHint      = $('input-hint');
  const loginPromptTxt = $('login-prompt-text');
  const sidebarLoginBtn= $('sidebar-login-btn');
  const logoutBtn      = $('logout-btn');

  if (newChatBtn)      newChatBtn.textContent      = t('newChat');
  if (sessionsLabel)   sessionsLabel.textContent   = t('sessions');
  if (botStatus)       botStatus.textContent       = t('status');
  if (msgInput)        msgInput.placeholder        = t('placeholder');
  if (inputHint)       inputHint.textContent       = t('hint');
  if (loginPromptTxt)  loginPromptTxt.textContent  = t('loginPrompt');
  if (sidebarLoginBtn) sidebarLoginBtn.textContent = t('loginBtn');
  if (logoutBtn)       logoutBtn.title             = t('logoutBtn');

  renderUserPanel();
  if ($('welcome')) showWelcome();
}

// ─── Auth / Navigation ────────────────────────────────────────────────────────
function logout() {
  clearToken();
  state.currentUser = null;
  state.sessions    = [];
  renderUserPanel();
  renderSessions();
  startNewSession();
}

function goToAdminDashboard() {
  window.location.href = '/admin.html';
}

function toggleSidebar() {
  const sidebar = $('sidebar');
  const btn     = $('sidebar-toggle');
  const collapsed = sidebar.classList.toggle('collapsed');
  btn.textContent = collapsed ? '▶' : '◀';
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Wire event listeners
  $('send-btn').addEventListener('click', handleSend);
  $('new-chat-btn').addEventListener('click', startNewSession);

  const clearBtn = $('clear-chat-btn');
  if (clearBtn) clearBtn.addEventListener('click', startNewSession);

  $('sidebar-toggle').addEventListener('click', toggleSidebar);

  $('logout-btn')?.addEventListener('click', logout);
  $('sidebar-login-btn')?.addEventListener('click', () => { window.location.href = '/login'; });
  $('admin-btn')?.addEventListener('click', goToAdminDashboard);

  // Language buttons
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => setLang(btn.dataset.lang, applyLangToPage));
  });

  // Input auto-resize + send on Enter
  const input = $('msg-input');
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    updateSendButton();
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Apply initial language from storage
  setLang(getLang(), applyLangToPage);

  // Bootstrap data
  await loadCurrentUser();
  try {
    await loadSessions();
  } catch {
    await startNewSession();
  }

  updateSendButton();
});