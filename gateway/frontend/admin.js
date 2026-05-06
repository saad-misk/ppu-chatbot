const adminState = {
  token: null,
  selectedFile: null,
};

async function getToken() {
  const res = await fetch("/api/auth/token?username=admin", { method: "POST" });
  const data = await res.json();
  return data.token;
}

async function fetchStats() {
  const res = await fetch("/api/admin/stats", {
    headers: { Authorization: `Bearer ${adminState.token}` },
  });
  return res.json();
}

async function fetchDocuments() {
  const res = await fetch("/api/admin/documents", {
    headers: { Authorization: `Bearer ${adminState.token}` },
  });
  return res.json();
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/admin/upload", {
    method: "POST",
    headers: { Authorization: `Bearer ${adminState.token}` },
    body: formData,
  });
  return res.json();
}

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((btn, i) => {
    const names = ["stats", "upload", "documents"];
    btn.classList.toggle("active", names[i] === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });

  if (name === "stats") loadStats();
  if (name === "documents") loadDocuments();
}

async function loadStats() {
  try {
    const data = await fetchStats();

    document.getElementById("stat-queries").textContent =
      data.total_queries ?? 0;
    document.getElementById("stat-positive").textContent =
      data.feedback?.positive ?? 0;
    document.getElementById("stat-negative").textContent =
      data.feedback?.negative ?? 0;

    const ratio = data.feedback?.ratio ?? 0;
    document.getElementById("stat-ratio").textContent = `${Math.round(
      ratio * 100
    )}%`;

    renderIntents(data.intent_breakdown || {});
  } catch (e) {
    document.getElementById("intent-list").innerHTML =
      '<div class="no-data">Failed to load stats.</div>';
  }
}

function renderIntents(breakdown) {
  const container = document.getElementById("intent-list");
  const entries = Object.entries(breakdown);

  if (entries.length === 0) {
    container.innerHTML = '<div class="no-data">No intent data yet.</div>';
    return;
  }

  const max = Math.max(...entries.map(([, v]) => v));
  container.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(
      ([intent, count]) => `
      <div class="intent-row">
        <div class="intent-name">${intent}</div>
        <div class="intent-bar-wrap">
          <div class="intent-bar" style="width:${Math.round(
            (count / max) * 100
          )}%"></div>
        </div>
        <div class="intent-count">${count}</div>
      </div>
    `
    )
    .join("");
}

async function loadDocuments() {
  const container = document.getElementById("docs-container");
  container.innerHTML =
    '<div class="no-data"><span class="spinner"></span></div>';

  try {
    const data = await fetchDocuments();
    const docs = data.documents || [];

    if (docs.length === 0) {
      container.innerHTML =
        '<div class="no-data">No documents indexed yet. Upload a PDF to get started.</div>';
      return;
    }

    container.innerHTML = `<div class="docs-grid">${docs
      .map(
        (doc) => `
        <div class="doc-card">
          <div class="doc-icon">📄</div>
          <div class="doc-info">
            <div class="doc-name">${doc.name || doc}</div>
            <div class="doc-meta">${
              doc.chunks ? `${doc.chunks} chunks` : "Indexed"
            }</div>
          </div>
        </div>
      `
      )
      .join("")}</div>`;
  } catch (e) {
    container.innerHTML =
      '<div class="no-data">Failed to load documents.</div>';
  }
}

function setupUpload() {
  const input = document.getElementById("pdf-file-input");
  const area = document.getElementById("upload-area");

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    adminState.selectedFile = file;
    document.getElementById("selected-file-name").textContent = file.name;
    document.getElementById("selected-file").style.display = "block";
    document.getElementById("upload-btn").disabled = false;
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
      adminState.selectedFile = file;
      document.getElementById("selected-file-name").textContent = file.name;
      document.getElementById("selected-file").style.display = "block";
      document.getElementById("upload-btn").disabled = false;
    }
  });
}

async function uploadPDF() {
  if (!adminState.selectedFile) return;

  const btn = document.getElementById("upload-btn");
  const status = document.getElementById("upload-status");

  btn.disabled = true;
  btn.textContent = "Uploading…";
  status.className = "upload-status";
  status.style.display = "none";

  try {
    const data = await uploadFile(adminState.selectedFile);
    status.className = "upload-status success";
    status.innerHTML = `✅ ${data.message}`;
    status.style.display = "flex";

    adminState.selectedFile = null;
    document.getElementById("pdf-file-input").value = "";
    document.getElementById("selected-file").style.display = "none";
  } catch (e) {
    status.className = "upload-status error";
    status.innerHTML = "❌ Upload failed. Please try again.";
    status.style.display = "flex";
  }

  btn.disabled = false;
  btn.textContent = "Upload & Index";
}

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
