const state = {
  sessionId: null,
  token: null,
  lang: "en",
  sessions: [],
  turnIds: {},
};

const i18n = {
  en: {
    newChat: "+ New Chat",
    placeholder: "Ask about fees, schedules, registration…",
    welcomeTitle: "PPU Assistant",
    welcomeSub:
      "Your smart guide to Palestine Polytechnic University. Ask me anything about fees, schedules, registration, or departments.",
    suggestions: [
      "What are the CS tuition fees?",
      "When does registration open?",
      "Who is the dean of Engineering?",
      "Where is the IT department?",
    ],
    status: "Online",
    hint: "Press Enter to send · Shift+Enter for new line",
    lowConf:
      "⚠️ I'm not fully certain about this answer. Please verify with the university.",
    sessions: "Recent Chats",
    you: "You",
    bot: "PPU Bot",
    ppu: "PPU © 2026",
  },
  ar: {
    newChat: "+ محادثة جديدة",
    placeholder: "اسأل عن الرسوم، الجداول، التسجيل…",
    welcomeTitle: "مساعد جامعة PPU",
    welcomeSub:
      "دليلك الذكي لجامعة بوليتكنك فلسطين. اسألني عن أي شيء يخص الرسوم أو الجداول أو التسجيل أو الأقسام.",
    suggestions: [
      "ما هي رسوم تخصص الحاسوب؟",
      "متى يفتح باب التسجيل؟",
      "من هو عميد كلية الهندسة؟",
      "أين قسم تكنولوجيا المعلومات؟",
    ],
    status: "متصل",
    hint: "اضغط Enter للإرسال · Shift+Enter لسطر جديد",
    lowConf: "⚠️ لست متأكداً تماماً من هذه الإجابة. يرجى التحقق مع الجامعة.",
    sessions: "المحادثات الأخيرة",
    you: "أنت",
    bot: "مساعد PPU",
    ppu: "PPU © 2026",
  },
};

function t(key) {
  return i18n[state.lang][key] || key;
}

async function apiGetToken() {
  const res = await fetch("/api/auth/token?username=student", {
    method: "POST",
  });
  const data = await res.json();
  return data.token;
}

async function apiNewSession() {
  const res = await fetch("/api/sessions/new?channel=web", {
    method: "POST",
    headers: { Authorization: `Bearer ${state.token}` },
  });
  return res.json();
}

async function apiSendMessage(sessionId, message) {
  const params = new URLSearchParams({ session_id: sessionId, message });
  const res = await fetch(`/api/chat/message?${params}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${state.token}` },
  });
  return res.json();
}

async function apiFeedback(sessionId, messageId, rating) {
  await fetch("/api/chat/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${state.token}`,
    },
    body: JSON.stringify({
      session_id: sessionId,
      message_id: messageId,
      rating,
    }),
  });
}

async function startNewSession() {
  const data = await apiNewSession();
  state.sessionId = data.session_id;

  const preview = t("newChat").replace("+ ", "");
  state.sessions.unshift({ id: state.sessionId, preview });
  renderSessions();
  setActiveSession(state.sessionId);

  clearMessages();
  showWelcome();
}

function renderSessions() {
  const container = document.getElementById("sessions-list");
  container.innerHTML = "";
  state.sessions.forEach((s) => {
    const el = document.createElement("div");
    el.className = "session-item";
    el.dataset.id = s.id;
    el.textContent = s.preview;
    el.onclick = () => switchSession(s.id);
    container.appendChild(el);
  });
}

function setActiveSession(id) {
  document.querySelectorAll(".session-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.id === id);
  });
}

async function switchSession(id) {
  state.sessionId = id;
  setActiveSession(id);
  clearMessages();
}

function showWelcome() {
  const msgs = document.getElementById("messages");
  msgs.innerHTML = `
    <div id="welcome">
      <div class="welcome-icon"><img src="static/images/ppu-logo.png" alt="" style="width:200px;height:100px;object-fit:contain;" /></div>
      <div class="welcome-title">${t("welcomeTitle")}</div>
      <div class="welcome-sub">${t("welcomeSub")}</div>
      <div class="suggestion-chips">
        ${t("suggestions")
          .map(
            (s) =>
              `<button class="suggestion-chip" onclick="sendSuggestion(this)">${s}</button>`
          )
          .join("")}
      </div>
    </div>
  `;
}

function clearMessages() {
  document.getElementById("messages").innerHTML = "";
}

function removeWelcome() {
  const w = document.getElementById("welcome");
  if (w) w.remove();
}

async function sendMessage(text) {
  if (!text.trim() || !state.sessionId) return;

  removeWelcome();

  const s = state.sessions.find((x) => x.id === state.sessionId);
  if (
    s &&
    s.preview ===
      (t("newChat").replace("+ ", "") || "New Chat" || "محادثة جديدة")
  ) {
    s.preview = text.slice(0, 36) + (text.length > 36 ? "…" : "");
    renderSessions();
    setActiveSession(state.sessionId);
  }

  appendMessage("user", text, null, null);

  const typingId = appendTyping();
  document.getElementById("send-btn").disabled = true;

  try {
    const data = await apiSendMessage(state.sessionId, text);
    removeTyping(typingId);

    if (data.low_confidence) {
      appendWarning(t("lowConf"));
    }

    appendMessage("bot", data.reply, data.turn_id, data.sources || []);
  } catch (e) {
    removeTyping(typingId);
    appendMessage("bot", "⚠️ Connection error. Please try again.", null, []);
  }

  document.getElementById("send-btn").disabled = false;
}

function sendSuggestion(btn) {
  const text = btn.textContent;
  document.getElementById("msg-input").value = text;
  handleSend();
}

function appendMessage(role, text, turnId, sources) {
  const container = document.getElementById("messages");
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const time = new Date().toLocaleTimeString(
    state.lang === "ar" ? "ar" : "en",
    { hour: "2-digit", minute: "2-digit" }
  );

  const sourcesHTML =
    sources && sources.length
      ? `<div class="msg-sources">${sources
          .map(
            (s) =>
              `<span class="source-chip">📄 ${s.doc_name}${
                s.page ? ` · p.${s.page}` : ""
              }</span>`
          )
          .join("")}</div>`
      : "";

  const feedbackHTML =
    role === "bot" && turnId
      ? `<div class="feedback-btns">
        <button class="fb-btn" id="up-${turnId}" onclick="submitFeedback('${turnId}','up')" title="Helpful">👍</button>
        <button class="fb-btn" id="down-${turnId}" onclick="submitFeedback('${turnId}','down')" title="Not helpful">👎</button>
      </div>`
      : "";

  row.innerHTML = `
    <div class="msg-avatar">${role === "bot" ? "🎓" : "👤"}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escapeHtml(text)}</div>
      ${sourcesHTML}
      <div class="msg-meta">
        <span class="msg-time">${time}</span>
        ${feedbackHTML}
      </div>
    </div>
  `;

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function appendWarning(text) {
  const container = document.getElementById("messages");
  const el = document.createElement("div");
  el.className = "confidence-warning";
  el.textContent = text;
  container.appendChild(el);
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
    <div class="msg-avatar">🎓</div>
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
  await apiFeedback(state.sessionId, turnId, rating);
  const upBtn = document.getElementById(`up-${turnId}`);
  const downBtn = document.getElementById(`down-${turnId}`);
  if (upBtn && downBtn) {
    upBtn.classList.toggle("active-up", rating === "up");
    downBtn.classList.toggle("active-down", rating === "down");
    upBtn.disabled = true;
    downBtn.disabled = true;
  }
}

function handleSend() {
  const input = document.getElementById("msg-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  input.style.height = "auto";
  sendMessage(text);
}

function setLang(lang) {
  state.lang = lang;
  document.body.classList.toggle("rtl", lang === "ar");
  document.querySelectorAll(".lang-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });

  document.getElementById("new-chat-btn").textContent = t("newChat");
  document.getElementById("sessions-label").textContent = t("sessions");
  document.getElementById("ppu-badge").textContent = t("ppu");
  document.getElementById("bot-status").textContent = t("status");
  document.getElementById("msg-input").placeholder = t("placeholder");
  document.getElementById("input-hint").textContent = t("hint");
  showWelcome();
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");
}

async function init() {
  const input = document.getElementById("msg-input");
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });

  document.getElementById("send-btn").addEventListener("click", handleSend);
  document
    .getElementById("new-chat-btn")
    .addEventListener("click", startNewSession);

  try {
    state.token = await apiGetToken();
    await startNewSession();
  } catch (e) {
    console.error("Init failed:", e);
  }
}

document.addEventListener("DOMContentLoaded", init);
