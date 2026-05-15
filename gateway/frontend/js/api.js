/**
 * api.js — Lightweight API helper shared across all pages.
 */

export function getToken() {
  return localStorage.getItem('token');
}

export function setToken(token) {
  localStorage.setItem('token', token);
}

export function clearToken() {
  localStorage.removeItem('token');
}

/**
 * Fetch wrapper that attaches the auth header and handles
 * 401/403 by clearing the token.
 *
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<any>} parsed JSON
 */
export async function apiFetch(url, options = {}) {
  const token = getToken();

  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  let res;
  try {
    res = await fetch(url, { ...options, headers });
  } catch {
    throw new Error('network_error');
  }

  if (res.status === 401 || res.status === 403) {
    clearToken();
    throw new Error('unauthorized');
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch { /* ignore */ }
    throw new Error(detail);
  }

  return res.json();
}