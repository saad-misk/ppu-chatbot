/**
 * admin.js — Admin panel logic.
 * Loaded as a regular script (not a module) so it works without
 * a bundler, but uses no globals beyond the minimal set needed.
 */

// ─── State ────────────────────────────────────────────────────────────────────
const adminState = {
  token:       localStorage.getItem('token'),
  selectedFile: null,
  currentTab:  'stats',
};

const PAGE_META = {
  stats:     { title: 'Statistics',  sub: 'Overview of chatbot usage and feedback' },
  upload:    { title: 'Upload PDF',  sub: 'Add documents to the knowledge base' },
  documents: { title: 'Documents',   sub: 'Files currently indexed in the knowledge base' },
};

// ─── API helpers ──────────────────────────────────────────────────────────────
function authHeaders() {
  return { Authorization: `Bearer ${adminState.token}` };
}

async function adminFetch(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: { ...authHeaders(), ...(options.headers || {}) },
  });

  if (res.status === 401 || res.status === 403) {
    localStorage.removeItem('token');
    window.location.href = '/';
    throw new Error('Unauthorized');
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(body || `HTTP ${res.status}`);
  }

  return res.json();
}

function fetchStats()     { return adminFetch('/api/admin/stats'); }
function fetchDocuments() { return adminFetch('/api/admin/documents'); }

async function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch('/api/admin/upload', {
    method: 'POST',
    headers: authHeaders(),   // no Content-Type — let browser set multipart boundary
    body: formData,
  });

  if (!res.ok) throw new Error(`Upload failed: HTTP ${res.status}`);
  return res.json();
}

// ─── Utilities ────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function $(id) { return document.getElementById(id); }

// ─── Tab Switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  adminState.currentTab = name;

  document.querySelectorAll('.sb-item').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });

  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.id === `tab-${name}`);
  });

  const meta = PAGE_META[name] || {};
  $('page-title').textContent = meta.title || '';
  $('page-sub').textContent   = meta.sub   || '';

  if (name === 'stats')     loadStats();
  if (name === 'documents') loadDocuments();
}

function refreshCurrent() {
  const tab = adminState.currentTab;
  if (tab === 'stats')     loadStats();
  if (tab === 'documents') loadDocuments();
}

// ─── Stats ────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const data = await fetchStats();

    $('stat-queries').textContent  = (data.total_queries      ?? 0).toLocaleString();
    $('stat-positive').textContent = (data.feedback?.positive  ?? 0).toLocaleString();
    $('stat-negative').textContent = (data.feedback?.negative  ?? 0).toLocaleString();
    $('stat-ratio').textContent    = `${Math.round((data.feedback?.ratio ?? 0) * 100)}%`;

    renderIntents(data.intent_breakdown || {});
  } catch {
    $('intent-list').innerHTML = `
      <div class="no-data">
        <i class="ti ti-alert-triangle" aria-hidden="true"></i>
        Failed to load stats.
      </div>`;
    $('intent-tag').textContent = 'Error';
  }
}

function renderIntents(breakdown) {
  const container = $('intent-list');
  const tag       = $('intent-tag');
  const entries   = Object.entries(breakdown);

  if (!entries.length) {
    container.innerHTML = `
      <div class="no-data">
        <i class="ti ti-database-off" aria-hidden="true"></i>
        No intent data yet.
      </div>`;
    tag.textContent = '0 intents';
    return;
  }

  tag.textContent = `${entries.length} intent${entries.length !== 1 ? 's' : ''}`;

  const sorted = entries.sort((a, b) => b[1] - a[1]);
  const max    = sorted[0][1];

  container.innerHTML = sorted
    .map(([intent, count], idx) => `
      <div class="intent-row">
        <div class="intent-rank">${idx + 1}</div>
        <div class="intent-name">${escapeHtml(intent)}</div>
        <div class="intent-bar-wrap">
          <div class="intent-bar" style="width:${Math.round((count / max) * 100)}%"></div>
        </div>
        <div class="intent-count">${count.toLocaleString()}</div>
      </div>`)
    .join('');
}

// ─── Documents ────────────────────────────────────────────────────────────────
async function loadDocuments() {
  const container = $('docs-container');
  const countEl   = $('docs-count');

  container.innerHTML = '<div class="no-data"><span class="spinner"></span></div>';
  countEl.textContent = 'Loading…';

  try {
    const data = await fetchDocuments();
    const docs = data.documents || [];

    countEl.textContent = docs.length
      ? `${docs.length} document${docs.length !== 1 ? 's' : ''} indexed`
      : 'No documents yet';

    if (!docs.length) {
      container.innerHTML = `
        <div class="no-data">
          <i class="ti ti-file-off" aria-hidden="true"></i>
          No documents indexed yet. Upload a PDF to get started.
        </div>`;
      return;
    }

    container.innerHTML = `<div class="docs-grid">${docs
      .map(doc => {
        const name   = doc.name || String(doc);
        const chunks = doc.chunks ? `${doc.chunks} chunks` : 'Indexed';
        return `
          <div class="doc-card">
            <div class="doc-icon">
              <i class="ti ti-file-type-pdf" aria-hidden="true"></i>
            </div>
            <div class="doc-info">
              <div class="doc-name" title="${escapeHtml(name)}">${escapeHtml(name)}</div>
              <div class="doc-meta">
                <i class="ti ti-database" aria-hidden="true"></i>
                ${escapeHtml(chunks)}
              </div>
            </div>
          </div>`;
      })
      .join('')}</div>`;
  } catch {
    countEl.textContent = 'Error loading documents';
    container.innerHTML = `
      <div class="no-data">
        <i class="ti ti-alert-triangle" aria-hidden="true"></i>
        Failed to load documents.
      </div>`;
  }
}

// ─── Upload ───────────────────────────────────────────────────────────────────
function setupUpload() {
  const input    = $('pdf-file-input');
  const area     = $('upload-area');
  const uploadBtn = $('upload-btn');

  input.addEventListener('change', () => {
    if (input.files[0]) setSelectedFile(input.files[0]);
  });

  area.addEventListener('click', () => input.click());

  area.addEventListener('dragover', e => {
    e.preventDefault();
    area.classList.add('drag-over');
  });

  area.addEventListener('dragleave', () => area.classList.remove('drag-over'));

  area.addEventListener('drop', e => {
    e.preventDefault();
    area.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file?.type === 'application/pdf') setSelectedFile(file);
  });

  uploadBtn.addEventListener('click', doUpload);

  $('remove-file-btn').addEventListener('click', clearFile);
}

function setSelectedFile(file) {
  adminState.selectedFile = file;
  $('selected-file-name').textContent = file.name;
  $('selected-file').style.display    = 'flex';
  $('upload-btn').disabled            = false;
  hideUploadStatus();
}

function clearFile() {
  adminState.selectedFile        = null;
  $('pdf-file-input').value      = '';
  $('selected-file').style.display = 'none';
  $('upload-btn').disabled       = true;
  hideUploadStatus();
}

function hideUploadStatus() {
  const s = $('upload-status');
  s.className = 'upload-status';
}

async function doUpload() {
  if (!adminState.selectedFile) return;

  const btn    = $('upload-btn');
  const status = $('upload-status');

  btn.disabled   = true;
  btn.innerHTML  = '<span class="spinner"></span> Uploading…';
  hideUploadStatus();

  try {
    const data = await uploadFile(adminState.selectedFile);
    status.className = 'upload-status success';
    status.innerHTML = `<i class="ti ti-circle-check" aria-hidden="true"></i> ${escapeHtml(data.message || 'Upload successful.')}`;
    clearFile();
  } catch {
    status.className = 'upload-status error';
    status.innerHTML = '<i class="ti ti-alert-circle" aria-hidden="true"></i> Upload failed. Please try again.';
  } finally {
    btn.disabled  = false;
    btn.innerHTML = '<i class="ti ti-cloud-upload" aria-hidden="true"></i> Upload &amp; Index';
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
async function init() {
  if (!adminState.token) {
    window.location.href = '/';
    return;
  }

  try {
    const res  = await fetch('/api/auth/me', { headers: authHeaders() });
    if (!res.ok) throw new Error('Auth check failed');
    const user = await res.json();
    if (!user.is_admin) { window.location.href = '/'; return; }
  } catch {
    localStorage.removeItem('token');
    window.location.href = '/';
    return;
  }

  // Wire sidebar nav buttons (use data-tab attribute, no onclick)
  document.querySelectorAll('.sb-item[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Topbar refresh
  $('refresh-btn')?.addEventListener('click', refreshCurrent);

  setupUpload();
  loadStats();
}

document.addEventListener('DOMContentLoaded', init);