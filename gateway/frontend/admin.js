const adminState = {
  token: null,
  selectedFile: null,
  currentTab: "stats",
};

const PAGE_META = {
  stats: { title: "Statistics", sub: "Overview of chatbot usage and feedback" },
  upload: { title: "Upload PDF", sub: "Add documents to the knowledge base" },
  documents: {
    title: "Documents",
    sub: "Files currently indexed in the knowledge base",
  },
};

// ── Auth ──────────────────────────────────────────────────────────────────────

async function getToken() {
  const res = await fetch("/api/auth/token?username=admin", { method: "POST" });
  const data = await res.json();
  return data.token;
}

function authHeaders() {
  return { Authorization: `Bearer ${adminState.token}` };
}

// ── API calls ─────────────────────────────────────────────────────────────────

async function fetchStats() {
  const res = await fetch("/api/admin/stats", { headers: authHeaders() });
  return res.json();
}

async function fetchDocuments() {
  const res = await fetch("/api/admin/documents", { headers: authHeaders() });
  return res.json();
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/admin/upload", {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });
  return res.json();
}

// ── Tab switching ─────────────────────────────────────────────────────────────

function switchTab(name) {
  adminState.currentTab = name;

  // Sidebar buttons
  document.querySelectorAll(".sb-item").forEach((btn) => {
    const targets = ["stats", "upload", "documents"];
    btn.classList.toggle("active", btn.getAttribute("onclick").includes(name));
  });

  // Panels
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });

  // Topbar text
  const meta = PAGE_META[name] || {};
  document.getElementById("page-title").textContent = meta.title || "";
  document.getElementById("page-sub").textContent = meta.sub || "";

  if (name === "stats") loadStats();
  if (name === "documents") loadDocuments();
}

function refreshCurrent() {
  const tab = adminState.currentTab;
  if (tab === "stats") loadStats();
  if (tab === "documents") loadDocuments();
}

// ── Stats ─────────────────────────────────────────────────────────────────────

async function loadStats() {
  try {
    const data = await fetchStats();

    document.getElementById("stat-queries").textContent = (
      data.total_queries ?? 0
    ).toLocaleString();
    document.getElementById("stat-positive").textContent = (
      data.feedback?.positive ?? 0
    ).toLocaleString();
    document.getElementById("stat-negative").textContent = (
      data.feedback?.negative ?? 0
    ).toLocaleString();

    const ratio = data.feedback?.ratio ?? 0;
    document.getElementById("stat-ratio").textContent = `${Math.round(
      ratio * 100
    )}%`;

    renderIntents(data.intent_breakdown || {});
  } catch (e) {
    document.getElementById("intent-list").innerHTML =
      '<div class="no-data"><i class="ti ti-alert-triangle" aria-hidden="true"></i>Failed to load stats.</div>';
    document.getElementById("intent-tag").textContent = "Error";
  }
}

function renderIntents(breakdown) {
  const container = document.getElementById("intent-list");
  const tag = document.getElementById("intent-tag");
  const entries = Object.entries(breakdown);

  if (entries.length === 0) {
    container.innerHTML =
      '<div class="no-data"><i class="ti ti-database-off" aria-hidden="true"></i>No intent data yet.</div>';
    tag.textContent = "0 intents";
    return;
  }

  tag.textContent = `${entries.length} intent${
    entries.length !== 1 ? "s" : ""
  }`;

  const sorted = entries.sort((a, b) => b[1] - a[1]);
  const max = sorted[0][1];

  container.innerHTML = sorted
    .map(
      ([intent, count], idx) => `
      <div class="intent-row">
        <div class="intent-rank">${idx + 1}</div>
        <div class="intent-name">${escapeHtml(intent)}</div>
        <div class="intent-bar-wrap">
          <div class="intent-bar" style="width:${Math.round(
            (count / max) * 100
          )}%"></div>
        </div>
        <div class="intent-count">${count.toLocaleString()}</div>
      </div>
    `
    )
    .join("");
}

// ── Documents ─────────────────────────────────────────────────────────────────

async function loadDocuments() {
  const container = document.getElementById("docs-container");
  const countEl = document.getElementById("docs-count");

  container.innerHTML =
    '<div class="no-data"><span class="spinner"></span></div>';
  countEl.textContent = "Loading…";

  try {
    const data = await fetchDocuments();
    const docs = data.documents || [];

    countEl.textContent =
      docs.length === 0
        ? "No documents yet"
        : `${docs.length} document${docs.length !== 1 ? "s" : ""} indexed`;

    if (docs.length === 0) {
      container.innerHTML = `
        <div class="no-data">
          <i class="ti ti-file-off" aria-hidden="true"></i>
          No documents indexed yet. Upload a PDF to get started.
        </div>`;
      return;
    }

    container.innerHTML = `<div class="docs-grid">${docs
      .map((doc) => {
        const name = doc.name || doc;
        const chunks = doc.chunks ? `${doc.chunks} chunks` : "Indexed";
        return `
          <div class="doc-card">
            <div class="doc-icon">
              <i class="ti ti-file-type-pdf" aria-hidden="true"></i>
            </div>
            <div class="doc-info">
              <div class="doc-name" title="${escapeHtml(name)}">${escapeHtml(
          name
        )}</div>
              <div class="doc-meta">
                <i class="ti ti-database" aria-hidden="true"></i>
                ${escapeHtml(chunks)}
              </div>
            </div>
          </div>`;
      })
      .join("")}</div>`;
  } catch (e) {
    countEl.textContent = "Error loading documents";
    container.innerHTML = `
      <div class="no-data">
        <i class="ti ti-alert-triangle" aria-hidden="true"></i>
        Failed to load documents.
      </div>`;
  }
}

// ── Upload ────────────────────────────────────────────────────────────────────

function setupUpload() {
  const input = document.getElementById("pdf-file-input");
  const area = document.getElementById("upload-area");

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    setSelectedFile(file);
  });

  area.addEventListener("dragover", (e) => {
    e.preventDefault();
    area.classList.add("drag-over");
  });

  area.addEventListener("dragleave", () => area.classList.remove("drag-over"));

  area.addEventListener("drop", (e) => {
    e.preventDefault();
    area.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type === "application/pdf") {
      setSelectedFile(file);
    }
  });
}

function setSelectedFile(file) {
  adminState.selectedFile = file;
  document.getElementById("selected-file-name").textContent = file.name;
  document.getElementById("selected-file").style.display = "flex";
  document.getElementById("upload-btn").disabled = false;
  hideStatus();
}

function clearFile() {
  adminState.selectedFile = null;
  document.getElementById("pdf-file-input").value = "";
  document.getElementById("selected-file").style.display = "none";
  document.getElementById("upload-btn").disabled = true;
  hideStatus();
}

function hideStatus() {
  const s = document.getElementById("upload-status");
  s.className = "upload-status";
  s.style.display = "none";
}

async function uploadPDF() {
  if (!adminState.selectedFile) return;

  const btn = document.getElementById("upload-btn");
  const status = document.getElementById("upload-status");

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Uploading…';
  hideStatus();

  try {
    const data = await uploadFile(adminState.selectedFile);
    status.className = "upload-status success";
    status.innerHTML = `<i class="ti ti-circle-check" aria-hidden="true"></i> ${escapeHtml(
      data.message || "Upload successful."
    )}`;
    status.style.display = "flex";
    clearFile();
  } catch (e) {
    status.className = "upload-status error";
    status.innerHTML =
      '<i class="ti ti-alert-circle" aria-hidden="true"></i> Upload failed. Please try again.';
    status.style.display = "flex";
  }

  btn.disabled = false;
  btn.innerHTML =
    '<i class="ti ti-cloud-upload" aria-hidden="true"></i> Upload & Index';
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Init ──────────────────────────────────────────────────────────────────────

async function init() {
  try {
    adminState.token = await getToken();
    setupUpload();
    loadStats();
  } catch (e) {
    console.error("Admin init failed:", e);
  }
}

document.addEventListener("DOMContentLoaded", init);
