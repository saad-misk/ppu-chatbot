(function () {
  if (document.getElementById("ppu-embed-root")) return;

  const script = document.currentScript;
  const scriptUrl = new URL(script.src);
  const baseUrl = script.dataset.baseUrl || scriptUrl.origin;
  const chatUrl = script.dataset.chatUrl || `${baseUrl}/chat`;
  const title = script.dataset.title || "PPU Assistant";

  // ── Inject styles ────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    #ppu-embed-root {
      position: fixed;
      top: 0;
      right: 0;
      width: 30vw;
      min-width: 320px;
      max-width: 520px;
      height: 100vh;
      z-index: 2147483647;
      display: flex;
      flex-direction: column;
      font-family: 'Segoe UI', Arial, sans-serif;
      transform: translateX(0);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    #ppu-embed-root.ppu-hidden {
      transform: translateX(100%);
    }

    /* Toggle tab — sticks out on the left side */
    #ppu-toggle-tab {
      position: fixed;
      top: 50%;
      transform: translateY(-50%);
      right: 30vw;
      min-right: 320px;
      z-index: 2147483647;
      width: 36px;
      height: 64px;
      background: #1B3A6B;
      border: none;
      border-radius: 8px 0 0 8px;
      color: white;
      font-size: 16px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: -4px 0 16px rgba(27,58,107,0.25);
      transition: right 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                  background 0.15s;
    }

    #ppu-toggle-tab:hover { background: #2a4f8f; }

    #ppu-toggle-tab.ppu-tab-hidden {
      right: 0;
    }

    /* Panel */
    #ppu-embed-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      background: white;
      box-shadow: -4px 0 32px rgba(0,0,0,0.15);
      overflow: hidden;
    }

    /* Header */
    #ppu-embed-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      background: #1B3A6B;
      flex-shrink: 0;
    }

    .ppu-embed-header-left {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .ppu-embed-header-logo {
      width: 30px;
      height: 30px;
      object-fit: contain;
      background: white;
      border-radius: 7px;
      padding: 3px;
    }

    .ppu-embed-header-title {
      font-size: 13px;
      font-weight: 700;
      color: white;
    }

    .ppu-embed-header-status {
      font-size: 11px;
      color: rgba(255,255,255,0.6);
      display: flex;
      align-items: center;
      gap: 4px;
      margin-top: 1px;
    }

    .ppu-embed-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #4ade80;
      animation: ppu-blink 2s infinite;
    }

    @keyframes ppu-blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .ppu-embed-actions {
      display: flex;
      gap: 6px;
    }

    .ppu-embed-btn {
      width: 28px;
      height: 28px;
      border: none;
      border-radius: 6px;
      background: rgba(255,255,255,0.12);
      color: white;
      cursor: pointer;
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background 0.15s;
    }

    .ppu-embed-btn:hover { background: rgba(255,255,255,0.22); }

    /* iframe */
    #ppu-embed-frame {
      flex: 1;
      border: none;
      width: 100%;
      display: block;
      background: #F4F6F9;
    }

    /* Mobile: full screen */
    @media (max-width: 600px) {
      #ppu-embed-root {
        width: 100vw;
        min-width: unset;
        max-width: unset;
      }
      #ppu-toggle-tab {
        right: 100vw;
      }
      #ppu-toggle-tab.ppu-tab-hidden {
        right: 0;
      }
    }
  `;
  document.head.appendChild(style);

  // ── Build HTML ───────────────────────────────────────────
  const root = document.createElement("div");
  root.id = "ppu-embed-root";

  root.innerHTML = `
    <div id="ppu-embed-panel">
      <div id="ppu-embed-header">
        <div class="ppu-embed-header-left">
          <img class="ppu-embed-header-logo" src="${esc(
            baseUrl
          )}/static/images/ppu-logo.png" alt="" onerror="this.style.display='none'" />
          <div>
            <div class="ppu-embed-header-title">${esc(title)}</div>
            <div class="ppu-embed-header-status">
              <span class="ppu-embed-dot"></span> Online
            </div>
          </div>
        </div>
        <div class="ppu-embed-actions">
          <button class="ppu-embed-btn" id="ppu-new-chat" title="New chat">✏️</button>
          <button class="ppu-embed-btn" id="ppu-open-full" title="Open full page">⤢</button>
        </div>
      </div>
      <iframe id="ppu-embed-frame" src="${esc(chatUrl)}" title="${esc(
    title
  )}"></iframe>
    </div>
  `;

  // Toggle tab (outside root so it's always visible)
  const tab = document.createElement("button");
  tab.id = "ppu-toggle-tab";
  tab.title = "Toggle PPU Assistant";
  tab.innerHTML = "◀";

  document.body.appendChild(root);
  document.body.appendChild(tab);

  // ── Logic ────────────────────────────────────────────────
  const frame = document.getElementById("ppu-embed-frame");
  let isOpen = true;

  function hide() {
    isOpen = false;
    root.classList.add("ppu-hidden");
    tab.classList.add("ppu-tab-hidden");
    tab.innerHTML = "▶";
  }

  function show() {
    isOpen = true;
    root.classList.remove("ppu-hidden");
    tab.classList.remove("ppu-tab-hidden");
    tab.innerHTML = "◀";
  }

  tab.addEventListener("click", () => (isOpen ? hide() : show()));

  document.getElementById("ppu-new-chat").addEventListener("click", () => {
    frame.src = "about:blank";
    setTimeout(() => {
      frame.src = chatUrl;
    }, 50);
  });

  document.getElementById("ppu-open-full").addEventListener("click", () => {
    window.open(chatUrl, "_blank");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && isOpen) hide();
  });

  function esc(v) {
    return String(v)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
})();
