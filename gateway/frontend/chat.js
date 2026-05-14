// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  token: localStorage.getItem("token") || null,
  currentUser: null,
  sessionId: localStorage.getItem("guest_session_id") || null,
  sessions: [],
  lang: localStorage.getItem("chat_lang") || "ar",   // Arabic by default
  sending: false,
};

// ─── i18n ─────────────────────────────────────────────────────────────────────
const i18n = {
  ar: {
    newChat: "+ محادثة جديدة",
    untitled: "محادثة جديدة",
    sessions: "المحادثات الأخيرة",
    placeholder: "اسأل عن الرسوم، الجداول، التسجيل...",
    hint: "اضغط Enter للإرسال · Shift+Enter لسطر جديد",
    status: "متصل",
    welcomeTitle: "مساعد جامعة PPU",
    welcomeSub: "دليلك الذكي لجامعة بوليتكنك فلسطين. اسأل عن الرسوم أو الجداول أو التسجيل أو الأقسام.",
    suggestions: [
      "ما هي رسوم تخصص الحاسوب؟",
      "متى يفتح باب التسجيل؟",
      "من هو عميد كلية الهندسة؟",
      "أين قسم تكنولوجيا المعلومات؟",
    ],
    connectionError: "حدث خطأ في الاتصال. حاول مرة أخرى.",
    lowConf: "لست متأكداً تماماً من هذه الإجابة. يرجى التحقق مع الجامعة.",
    userRole: "مستخدم",
    adminRole: "مدير النظام",
    guestName: "زائر",
    guestRole: "غير مسجّل",
    loginPrompt: "سجّل الدخول لحفظ محادثاتك",
    loginBtn: "تسجيل الدخول",
    logoutBtn: "تسجيل الخروج",
    deleteConfirm: "هل أنت متأكد من حذف هذه المحادثة؟",
  },
  en: {
    newChat: "+ New Chat",
    untitled: "New Chat",
    sessions: "Recent Chats",
    placeholder: "Ask about fees, schedules, registration...",
    hint: "Press Enter to send · Shift+Enter for new line",
    status: "Online",
    welcomeTitle: "PPU Assistant",
    welcomeSub: "Your smart guide to Palestine Polytechnic University. Ask about fees, schedules, registration, or departments.",
    suggestions: [
      "What are the CS tuition fees?",
      "When does registration open?",
      "Who is the dean of Engineering?",
      "Where is the IT department?",
    ],
    connectionError: "Connection error. Please try again.",
    lowConf: "I am not fully certain about this answer. Please verify with the university.",
    userRole: "User",
    adminRole: "Administrator",
    guestName: "Guest",
    guestRole: "Not signed in",
    loginPrompt: "Sign in to save your chats",
    loginBtn: "Sign In",
    logoutBtn: "Logout",
    deleteConfirm: "Delete this conversation?",
  },
};

function t(key) { return i18n[state.lang][key] || key; }

// ─── API helpers ──────────────────────────────────────────────────────────────
function authHeaders() {
  const h = { "Content-Type": "application/json" };
  if (state.token) h["Authorization"] = `Bearer ${state.token}`;
  return h;
}

async function apiFetch(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { ...(options.headers || {}), ...authHeaders() },
  });

  if (res.status === 401 || res.status === 403) {
    // Token expired – clear but stay on chat page as guest
    localStorage.removeItem("token");
    state.token = null;
    state.currentUser = null;
    renderUserPanel();
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `Request failed: ${res.status}`);
  }
  return res.json();
}

// ─── User panel ───────────────────────────────────────────────────────────────
async function loadCurrentUser() {
  if (!state.token) { renderUserPanel(); return; }
  try {
    state.currentUser = await apiFetch("/api/auth/me");
  } catch (_) {
    state.currentUser = null;
  }
  renderUserPanel();
}

function renderUserPanel() {
  const user = state.currentUser;
  const nameEl  = document.getElementById("user-name");
  const roleEl  = document.getElementById("user-role");
  const avEl    = document.getElementById("user-avatar");
  const adminBtn = document.getElementById("admin-btn");
  const loginArea = document.getElementById("login-prompt-area");
  const logoutBtn = document.getElementById("logout-btn");

  if (user) {
    nameEl.textContent  = user.full_name || user.email || t("guestName");
    roleEl.textContent  = user.is_admin ? t("adminRole") : t("userRole");
    avEl.textContent    = user.is_admin ? "A" : (user.full_name ? user.full_name[0].toUpperCase() : "U");
    if (adminBtn) adminBtn.style.display = user.is_admin ? "flex" : "none";
    if (loginArea) loginArea.style.display = "none";
    if (logoutBtn) logoutBtn.style.display = "flex";
  } else {
    nameEl.textContent  = t("guestName");
    roleEl.textContent  = t("guestRole");
    avEl.textContent    = "👤";
    if (adminBtn) adminBtn.style.display = "none";
    if (loginArea) loginArea.style.display = "flex";
    if (logoutBtn) logoutBtn.style.display = "none";
  }
}

// ─── Sessions ─────────────────────────────────────────────────────────────────
async function loadSessions() {
  if (!state.token) {
    // Guest: restore saved session or create a new one
    if (state.sessionId) {
      try {
        await loadSessionHistory(state.sessionId);
      } catch (_) {
        state.sessionId = null;
        localStorage.removeItem("guest_session_id");
        await startNewSession();
      }
    } else {
      await startNewSession();
    }
    return;
  }

  // Authenticated user: load server-side session list
  const data = await apiFetch("/api/sessions");
  state.sessions = (data.sessions || []).map((s) => ({
    id: s.session_id,
    preview: s.preview || t("untitled"),
  }));
  renderSessions();

  if (state.sessions.length) {
    state.sessionId = state.sessions[0].id;
    setActiveSession(state.sessionId);
    await loadSessionHistory(state.sessionId);
  } else {
    await startNewSession();
  }
}

async function startNewSession() {
  const data = await apiFetch("/api/sessions/new?channel=web", { method: "POST" });
  state.sessionId = data.session_id;

  if (!state.token) {
    localStorage.setItem("guest_session_id", state.sessionId);
  } else {
    state.sessions.unshift({ id: state.sessionId, preview: t("untitled") });
    renderSessions();
  }

  setActiveSession(state.sessionId);
  showWelcome();
  updateSendButton();
}

function renderSessions() {
  const container = document.getElementById("sessions-list");
  container.innerHTML = "";

  state.sessions.forEach((session) => {
    const wrap = document.createElement("div");
    wrap.className = "session-item-wrap";

    const item = document.createElement("button");
    item.type = "button";
    item.className = "session-item";
    item.dataset.id = session.id;
    item.textContent = session.preview;
    item.addEventListener("click", () => switchSession(session.id));
    item.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      startRename(session.id, item, session.preview);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "session-delete-btn";
    deleteBtn.title = "Delete";
    deleteBtn.textContent = "✕";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (confirm(t("deleteConfirm"))) deleteSession(session.id);
    });

    wrap.appendChild(item);
    wrap.appendChild(deleteBtn);
    container.appendChild(wrap);
  });
}

function startRename(id, itemEl, currentPreview) {
  const wrap = itemEl.parentElement;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "session-rename-input";
  input.value = currentPreview;
  wrap.replaceChild(input, itemEl);
  input.focus();
  input.select();

  async function commitRename() {
    const newName = input.value.trim() || currentPreview;
    const session = state.sessions.find((s) => s.id === id);
    if (session) session.preview = newName;
    try {
      await apiFetch(`/api/sessions/${id}/rename`, {
        method: "PATCH",
        body: JSON.stringify({ preview: newName }),
      });
    } catch (_) {}
    renderSessions();
    setActiveSession(state.sessionId);
  }

  input.addEventListener("blur", commitRename);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    if (e.key === "Escape") {
      const s = state.sessions.find((s) => s.id === id);
      if (s) s.preview = currentPreview;
      renderSessions();
      setActiveSession(state.sessionId);
    }
  });
}

async function deleteSession(id) {
  try {
    await apiFetch(`/api/sessions/${id}`, { method: "DELETE" });
    state.sessions = state.sessions.filter((s) => s.id !== id);
    if (state.sessionId === id) {
      if (state.sessions.length) await switchSession(state.sessions[0].id);
      else await startNewSession();
    } else {
      renderSessions();
      setActiveSession(state.sessionId);
    }
  } catch (err) {
    console.error("Delete session failed:", err);
  }
}

function setActiveSession(id) {
  document.querySelectorAll(".session-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.id === id);
  });
}

async function switchSession(id) {
  state.sessionId = id;
  setActiveSession(id);
  await loadSessionHistory(id);
  updateSendButton();
}

async function loadSessionHistory(id) {
  const data = await apiFetch(`/api/chat/history/${id}`);
  clearMessages();
  const turns = data.turns || [];
  if (!turns.length) { showWelcome(); return; }
  turns.forEach((turn) => appendMessage(turn.role === "assistant" ? "bot" : "user", turn.content, turn.id));
}

// ─── Welcome screen ───────────────────────────────────────────────────────────
function showWelcome() {
  const messages = document.getElementById("messages");
  messages.innerHTML = `
    <div id="welcome">
      <div class="welcome-icon">
        <img src="static/images/ppu-logo.png" alt="" style="width:180px;height:90px;object-fit:contain;" />
      </div>
      <div class="welcome-title">${escapeHtml(t("welcomeTitle"))}</div>
      <div class="welcome-sub">${escapeHtml(t("welcomeSub"))}</div>
      <div class="suggestion-chips">
        ${t("suggestions").map((text) =>
          `<button class="suggestion-chip" type="button" data-suggestion="${escapeHtml(text)}">${escapeHtml(text)}</button>`
        ).join("")}
      </div>
    </div>
  `;
  messages.querySelectorAll(".suggestion-chip").forEach((btn) => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.suggestion));
  });
}

function clearMessages() { document.getElementById("messages").innerHTML = ""; }
function removeWelcome() { const w = document.getElementById("welcome"); if (w) w.remove(); }

// ─── Messaging ────────────────────────────────────────────────────────────────
async function sendMessage(text) {
  const message = text.trim();
  if (!message || state.sending) return;

  if (!state.sessionId) await startNewSession();

  state.sending = true;
  updateSendButton();
  removeWelcome();
  appendMessage("user", message);
  updateSessionPreview(message);

  const typingId = appendTyping();
  try {
    const params = new URLSearchParams({ session_id: state.sessionId, message });
    const data = await apiFetch(`/api/chat/message?${params}`, { method: "POST" });
    removeTyping(typingId);
    if (data.low_confidence) appendWarning(t("lowConf"));
    appendMessage("bot", data.reply || "", data.turn_id, data.sources || []);
  } catch (err) {
    console.error("Send failed:", err);
    removeTyping(typingId);
    appendMessage("bot", t("connectionError"));
  } finally {
    state.sending = false;
    updateSendButton();
  }
}

function updateSessionPreview(message) {
  const session = state.sessions.find((s) => s.id === state.sessionId);
  if (!session) return;
  if (session.preview === t("untitled") || session.preview === "New Chat" || session.preview === "محادثة جديدة") {
    session.preview = message.slice(0, 36) + (message.length > 36 ? "…" : "");
    renderSessions();
    setActiveSession(state.sessionId);
  }
}

function appendMessage(role, text, turnId = null, sources = []) {
  const container = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const time = new Date().toLocaleTimeString(state.lang === "ar" ? "ar" : "en", { hour: "2-digit", minute: "2-digit" });

  const sourcesHtml = sources.length
    ? `<div class="msg-sources">${sources.map((s) => {
        const page = s.page ? ` · p.${escapeHtml(String(s.page))}` : "";
        return `<span class="source-chip">${escapeHtml(s.doc_name || "source")}${page}</span>`;
      }).join("")}</div>`
    : "";

  const feedbackHtml = role === "bot" && turnId
    ? `<div class="feedback-btns">
        <button class="fb-btn" type="button" data-rating="up" data-turn="${escapeHtml(turnId)}" title="Helpful">+</button>
        <button class="fb-btn" type="button" data-rating="down" data-turn="${escapeHtml(turnId)}" title="Not helpful">-</button>
      </div>`
    : "";

  row.innerHTML = `
    <div class="msg-avatar">${role === "bot" ? "B" : "U"}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escapeHtml(text)}</div>
      ${sourcesHtml}
      <div class="msg-meta">
        <span class="msg-time">${time}</span>
        ${feedbackHtml}
      </div>
    </div>
  `;

  row.querySelectorAll(".fb-btn").forEach((btn) => {
    btn.addEventListener("click", () => submitFeedback(btn.dataset.turn, btn.dataset.rating));
  });

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function appendWarning(text) {
  const container = document.getElementById("messages");
  const w = document.createElement("div");
  w.className = "confidence-warning";
  w.textContent = text;
  container.appendChild(w);
  container.scrollTop = container.scrollHeight;
}

let typingCounter = 0;
function appendTyping() {
  const id = `typing-${typingCounter++}`;
  const container = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = "message-row bot";
  row.id = id;
  row.innerHTML = `
    <div class="msg-avatar">B</div>
    <div class="msg-content">
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return id;
}
function removeTyping(id) { const el = document.getElementById(id); if (el) el.remove(); }

async function submitFeedback(turnId, rating) {
  try {
    await apiFetch("/api/chat/feedback", {
      method: "POST",
      body: JSON.stringify({ session_id: state.sessionId, message_id: turnId, rating }),
    });
  } catch (_) {}
}

// ─── Input ────────────────────────────────────────────────────────────────────
function handleSend() {
  const input = document.getElementById("msg-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  updateSendButton();
  sendMessage(text);
}

function updateSendButton() {
  const input = document.getElementById("msg-input");
  const btn = document.getElementById("send-btn");
  btn.disabled = state.sending || !input.value.trim();
}

// ─── Language ─────────────────────────────────────────────────────────────────
function setLang(lang) {
  state.lang = lang;
  localStorage.setItem("chat_lang", lang);
  document.documentElement.lang = lang;
  document.body.classList.toggle("rtl", lang === "ar");
  document.querySelectorAll(".lang-btn").forEach((b) => b.classList.toggle("active", b.dataset.lang === lang));

  const newChatBtn = document.getElementById("new-chat-btn");
  const sessionsLabel = document.getElementById("sessions-label");
  if (newChatBtn) newChatBtn.textContent = t("newChat");
  if (sessionsLabel) sessionsLabel.textContent = t("sessions");
  document.getElementById("bot-status").textContent = t("status");
  document.getElementById("msg-input").placeholder = t("placeholder");
  document.getElementById("input-hint").textContent = t("hint");

  const loginPromptEl = document.getElementById("login-prompt-text");
  if (loginPromptEl) loginPromptEl.textContent = t("loginPrompt");
  const loginBtnEl = document.getElementById("sidebar-login-btn");
  if (loginBtnEl) loginBtnEl.textContent = t("loginBtn");
  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) logoutBtn.title = t("logoutBtn");

  renderUserPanel();
  if (document.getElementById("welcome")) showWelcome();
}

// ─── Auth modal ───────────────────────────────────────────────────────────────
function openAuthModal() {
  window.location.href = "/login";
}

function goToAdminDashboard() { window.location.href = "/admin.html"; }

function logout() {
  localStorage.removeItem("token");
  state.token = null;
  state.currentUser = null;
  state.sessions = [];
  renderUserPanel();
  renderSessions();
  // Start a fresh guest session
  startNewSession();
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, "<br>");
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const btn = document.getElementById("sidebar-toggle");
  sidebar.classList.toggle("collapsed");
  btn.textContent = sidebar.classList.contains("collapsed") ? "▶" : "◀";
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Wire up controls
  document.getElementById("send-btn").addEventListener("click", handleSend);
  document.getElementById("new-chat-btn").addEventListener("click", startNewSession);

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) logoutBtn.addEventListener("click", logout);

  const sidebarLoginBtn = document.getElementById("sidebar-login-btn");
  if (sidebarLoginBtn) sidebarLoginBtn.addEventListener("click", openAuthModal);

  const input = document.getElementById("msg-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
    updateSendButton();
  });

  // Apply saved language (Arabic default)
  setLang(state.lang);

  // Load user info (works even if not logged in)
  await loadCurrentUser();

  // Load sessions (works as guest too)
  try {
    await loadSessions();
  } catch (err) {
    console.error("Init failed:", err);
    await startNewSession();
  }

  updateSendButton();
});
