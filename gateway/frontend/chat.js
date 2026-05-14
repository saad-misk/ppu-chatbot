const state = {
  token: null,
  currentUser: null,
  sessionId: null,
  sessions: [],
  lang: localStorage.getItem("chat_lang") || "ar",
  sending: false,
};

const i18n = {
  en: {
    newChat: "+ New Chat",
    untitled: "New Chat",
    sessions: "Recent Chats",
    placeholder: "Ask about fees, schedules, registration...",
    hint: "Press Enter to send · Shift+Enter for new line",
    status: "Online",
    welcomeTitle: "PPU Assistant",
    welcomeSub:
      "Your smart guide to Palestine Polytechnic University. Ask about fees, schedules, registration, or departments.",
    suggestions: [
      "What are the CS tuition fees?",
      "When does registration open?",
      "Who is the dean of Engineering?",
      "Where is the IT department?",
    ],
    connectionError: "Connection error. Please try again.",
    lowConf:
      "I am not fully certain about this answer. Please verify with the university.",
    userRole: "User",
    adminRole: "Administrator",
  },
  ar: {
    newChat: "+ محادثة جديدة",
    untitled: "محادثة جديدة",
    sessions: "المحادثات الأخيرة",
    placeholder: "اسأل عن الرسوم، الجداول، التسجيل...",
    hint: "اضغط Enter للإرسال · Shift+Enter لسطر جديد",
    status: "متصل",
    welcomeTitle: "مساعد جامعة PPU",
    welcomeSub:
      "دليلك الذكي لجامعة بوليتكنك فلسطين. اسأل عن الرسوم أو الجداول أو التسجيل أو الأقسام.",
    suggestions: [
      "ما هي رسوم تخصص الحاسوب؟",
      "متى يفتح باب التسجيل؟",
      "من هو عميد كلية الهندسة؟",
      "أين قسم تكنولوجيا المعلومات؟",
    ],
    connectionError: "حدث خطأ في الاتصال. حاول مرة أخرى.",
    lowConf: "لست متأكدا تماما من هذه الإجابة. يرجى التحقق مع الجامعة.",
    userRole: "مستخدم",
    adminRole: "مدير النظام",
  },
};

function t(key) {
  return i18n[state.lang][key] || key;
}

function authHeaders() {
  return { Authorization: `Bearer ${state.token}` };
}

async function apiFetch(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...authHeaders(),
    },
  });

  if (res.status === 401 || res.status === 403) {
    localStorage.removeItem("token");
    window.location.href = "/";
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    const body = await res.text();
    throw new Error(body || `Request failed: ${res.status}`);
  }

  return res.json();
}

async function loadCurrentUser() {
  state.currentUser = await apiFetch("/api/auth/me");
  document.getElementById("user-name").textContent =
    state.currentUser.full_name || state.currentUser.email || "User";
  document.getElementById("user-role").textContent = state.currentUser.is_admin
    ? t("adminRole")
    : t("userRole");
  document.getElementById("user-avatar").textContent = state.currentUser
    .is_admin
    ? "A"
    : "U";
  document.getElementById("admin-btn").style.display = state.currentUser
    .is_admin
    ? "flex"
    : "none";
}

async function loadSessions() {
  const data = await apiFetch("/api/sessions");
  state.sessions = (data.sessions || []).map((session) => ({
    id: session.session_id,
    preview: session.preview || t("untitled"),
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
  const data = await apiFetch("/api/sessions/new?channel=web", {
    method: "POST",
  });
  state.sessionId = data.session_id;
  state.sessions.unshift({ id: state.sessionId, preview: t("untitled") });
  renderSessions();
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
    deleteBtn.title = "Delete chat";
    deleteBtn.textContent = "✕";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteSession(session.id);
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preview: newName }),
      });
    } catch (e) {
      console.warn("Rename API not available, saved locally only");
    }
    renderSessions();
    setActiveSession(state.sessionId);
  }

  input.addEventListener("blur", commitRename);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      input.blur();
    }
    if (e.key === "Escape") {
      const session = state.sessions.find((s) => s.id === id);
      if (session) session.preview = currentPreview;
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
      if (state.sessions.length) {
        await switchSession(state.sessions[0].id);
      } else {
        await startNewSession();
      }
    } else {
      renderSessions();
      setActiveSession(state.sessionId);
    }
  } catch (error) {
    console.error("Delete session failed:", error);
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
  if (!turns.length) {
    showWelcome();
    return;
  }

  turns.forEach((turn) => {
    appendMessage(
      turn.role === "assistant" ? "bot" : "user",
      turn.content,
      turn.id
    );
  });
}

function showWelcome() {
  const messages = document.getElementById("messages");
  messages.innerHTML = `
    <div id="welcome">
      <div class="welcome-icon">
        <img src="static/images/ppu-logo.png" alt="" style="width:200px;height:100px;object-fit:contain;" />
      </div>
      <div class="welcome-title">${escapeHtml(t("welcomeTitle"))}</div>
      <div class="welcome-sub">${escapeHtml(t("welcomeSub"))}</div>
      <div class="suggestion-chips">
        ${t("suggestions")
          .map(
            (text) =>
              `<button class="suggestion-chip" type="button" data-suggestion="${escapeHtml(
                text
              )}">${escapeHtml(text)}</button>`
          )
          .join("")}
      </div>
    </div>
  `;

  messages.querySelectorAll(".suggestion-chip").forEach((button) => {
    button.addEventListener("click", () =>
      sendMessage(button.dataset.suggestion)
    );
  });
}

function clearMessages() {
  document.getElementById("messages").innerHTML = "";
}

function removeWelcome() {
  const welcome = document.getElementById("welcome");
  if (welcome) welcome.remove();
}

async function sendMessage(text) {
  const message = text.trim();
  if (!message || state.sending) return;

  if (!state.sessionId) {
    await startNewSession();
  }

  state.sending = true;
  updateSendButton();
  removeWelcome();
  appendMessage("user", message);
  updateSessionPreview(message);

  const typingId = appendTyping();

  try {
    const params = new URLSearchParams({
      session_id: state.sessionId,
      message,
    });
    const data = await apiFetch(`/api/chat/message?${params}`, {
      method: "POST",
    });
    removeTyping(typingId);

    if (data.low_confidence) appendWarning(t("lowConf"));
    appendMessage("bot", data.reply || "", data.turn_id, data.sources || []);
  } catch (error) {
    console.error("Send message failed:", error);
    removeTyping(typingId);
    appendMessage("bot", t("connectionError"));
  } finally {
    state.sending = false;
    updateSendButton();
  }
}

function updateSessionPreview(message) {
  const session = state.sessions.find((item) => item.id === state.sessionId);
  if (!session) return;

  if (session.preview === t("untitled") || session.preview === "New Chat") {
    session.preview = message.slice(0, 36) + (message.length > 36 ? "..." : "");
    renderSessions();
    setActiveSession(state.sessionId);
  }
}

function appendMessage(role, text, turnId = null, sources = []) {
  const container = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const time = new Date().toLocaleTimeString(
    state.lang === "ar" ? "ar" : "en",
    {
      hour: "2-digit",
      minute: "2-digit",
    }
  );

  const sourcesHtml = sources.length
    ? `<div class="msg-sources">${sources
        .map((source) => {
          const page = source.page
            ? ` · p.${escapeHtml(String(source.page))}`
            : "";
          return `<span class="source-chip">${escapeHtml(
            source.doc_name || "source"
          )}${page}</span>`;
        })
        .join("")}</div>`
    : "";

  const feedbackHtml =
    role === "bot" && turnId
      ? `<div class="feedback-btns">
          <button class="fb-btn" type="button" data-rating="up" data-turn="${escapeHtml(
            turnId
          )}" title="Helpful">+</button>
          <button class="fb-btn" type="button" data-rating="down" data-turn="${escapeHtml(
            turnId
          )}" title="Not helpful">-</button>
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

  row.querySelectorAll(".fb-btn").forEach((button) => {
    button.addEventListener("click", () =>
      submitFeedback(button.dataset.turn, button.dataset.rating)
    );
  });

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function appendWarning(text) {
  const container = document.getElementById("messages");
  const warning = document.createElement("div");
  warning.className = "confidence-warning";
  warning.textContent = text;
  container.appendChild(warning);
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
  const el = document.getElementById(id);
  if (el) el.remove();
}

async function submitFeedback(turnId, rating) {
  await apiFetch("/api/chat/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: state.sessionId,
      message_id: turnId,
      rating,
    }),
  });
}

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
  const sendBtn = document.getElementById("send-btn");
  sendBtn.disabled = state.sending || !state.sessionId || !input.value.trim();
}

function setLang(lang) {
  state.lang = lang;
  localStorage.setItem("chat_lang", lang);
  document.body.classList.toggle("rtl", lang === "ar");

  document.querySelectorAll(".lang-btn").forEach((button) => {
    button.classList.toggle("active", button.dataset.lang === lang);
  });

  document.getElementById("new-chat-btn").textContent = t("newChat");
  document.getElementById("sessions-label").textContent = t("sessions");
  document.getElementById("bot-status").textContent = t("status");
  document.getElementById("msg-input").placeholder = t("placeholder");
  document.getElementById("input-hint").textContent = t("hint");

  if (state.currentUser) {
    document.getElementById("user-role").textContent = state.currentUser
      .is_admin
      ? t("adminRole")
      : t("userRole");
  }

  if (document.getElementById("welcome")) showWelcome();
}

function goToAdminDashboard() {
  window.location.href = "/admin.html";
}

function logout() {
  localStorage.removeItem("token");
  window.location.href = "/";
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, "<br>");
}

document.addEventListener("DOMContentLoaded", async () => {
  state.token = localStorage.getItem("token");
  if (!state.token) {
    window.location.href = "/";
    return;
  }

  document.getElementById("send-btn").addEventListener("click", handleSend);
  document.getElementById("logout-btn").addEventListener("click", logout);
  document
    .getElementById("new-chat-btn")
    .addEventListener("click", startNewSession);

  const input = document.getElementById("msg-input");
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
    updateSendButton();
  });

  setLang(state.lang);

  try {
    await loadCurrentUser();
    await loadSessions();
  } catch (error) {
    console.error("Chat init failed:", error);
    localStorage.removeItem("token");
    window.location.href = "/";
  }

  updateSendButton();
});

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  const btn = document.getElementById("sidebar-toggle");
  sidebar.classList.toggle("collapsed");
  btn.textContent = sidebar.classList.contains("collapsed") ? "▶" : "◀";
}
