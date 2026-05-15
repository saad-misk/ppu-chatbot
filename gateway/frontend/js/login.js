/**
 * login.js — Authentication page logic.
 * Depends on: js/api.js
 */
import { apiFetch, setToken, getToken } from './api.js';

// ─── State ────────────────────────────────────────────────────────────────────
let pendingEmail  = '';
let currentLang   = 'ar';

// ─── i18n ─────────────────────────────────────────────────────────────────────
const T = {
  en: {
    brandSub:          'Palestine Polytechnic University',
    tabLogin:          'Sign In',
    tabRegister:       'Create Account',
    loginTitle:        'Welcome back',
    loginSub:          'Sign in with your PPU email',
    lblEmailL:         'Email',
    lblPassL:          'Password',
    loginBtn:          'Sign In',
    registerTitle:     'Create Account',
    registerSub:       'Use your official PPU email',
    lblName:           'Full Name',
    lblEmailR:         'PPU Email',
    domainHint:        '✉️ Accepted: @ppu.edu.ps or @ppu.edu',
    lblPassR:          'Password',
    lblConfirmPass:    'Confirm Password',
    registerBtn:       'Create Account',
    verifyTitle:       'Verify Your Email',
    verifySub:         'Enter the 6-digit code sent to your PPU email',
    verifyBtn:         'Verify & Continue',
    resendBtn:         'Resend Code',
    footer:            'PPU Assistant © 2026 — Palestine Polytechnic University',
    bottom:            '© 2026 Palestine Polytechnic University — All rights reserved',
    signingIn:         'Signing in…',
    creating:          'Creating account…',
    loginOk:           '✅ Logged in!',
    registerOk:        '✅ Account created! Check your email.',
    verifyOk:          '✅ Verified! You can now sign in.',
    resendOk:          '✅ Code resent.',
    fillAll:           'Please fill all fields',
    fullCode:          'Enter the full 6-digit code',
    connErr:           'Connection error',
    passwordMismatch:  'Passwords do not match',
    passwordTooShort:  'Password must be at least 6 characters',
    registerFailed:    'Registration failed',
    show:              'Show',
    hide:              'Hide',
  },
  ar: {
    brandSub:          'جامعة بوليتكنك فلسطين',
    tabLogin:          'تسجيل الدخول',
    tabRegister:       'إنشاء حساب',
    loginTitle:        'مرحباً بعودتك',
    loginSub:          'سجّل الدخول ببريدك الجامعي',
    lblEmailL:         'البريد الإلكتروني',
    lblPassL:          'كلمة المرور',
    loginBtn:          'تسجيل الدخول',
    registerTitle:     'إنشاء حساب جديد',
    registerSub:       'استخدم بريدك الجامعي الرسمي',
    lblName:           'الاسم الكامل',
    lblEmailR:         'البريد الجامعي',
    domainHint:        '✉️ المقبول: @ppu.edu.ps أو @ppu.edu',
    lblPassR:          'كلمة المرور',
    lblConfirmPass:    'تأكيد كلمة المرور',
    registerBtn:       'إنشاء الحساب',
    verifyTitle:       'تحقق من بريدك',
    verifySub:         'أدخل الرمز المكوّن من 6 أرقام',
    verifyBtn:         'تحقق والمتابعة',
    resendBtn:         'إعادة إرسال الرمز',
    footer:            'مساعد PPU © 2026 — جامعة بوليتكنك فلسطين',
    bottom:            '© 2026 جامعة بوليتكنك فلسطين — جميع الحقوق محفوظة',
    signingIn:         'جاري التحقق…',
    creating:          'جاري الإنشاء…',
    loginOk:           '✅ تم تسجيل الدخول!',
    registerOk:        '✅ تم إنشاء الحساب! تحقق من بريدك.',
    verifyOk:          '✅ تم التحقق! يمكنك الآن تسجيل الدخول.',
    resendOk:          '✅ تم إعادة الإرسال.',
    fillAll:           'يرجى ملء جميع الحقول',
    fullCode:          'أدخل الرمز كاملاً',
    connErr:           'خطأ في الاتصال',
    passwordMismatch:  'كلمات المرور غير متطابقة',
    passwordTooShort:  'كلمة المرور يجب أن تكون 6 أحرف على الأقل',
    registerFailed:    'فشل التسجيل',
    show:              'إظهار',
    hide:              'إخفاء',
  },
};

function t(key) { return T[currentLang]?.[key] ?? key; }

// ─── Language ─────────────────────────────────────────────────────────────────
function setLang(lang) {
  currentLang = lang;
  document.body.classList.toggle('rtl', lang === 'ar');
  document.querySelectorAll('.lang-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.lang === lang)
  );

  const s = T[lang];
  setText('brand-sub',         s.brandSub);
  setText('tab-login',         s.tabLogin);
  setText('tab-register',      s.tabRegister);
  setText('login-title',       s.loginTitle);
  setText('login-sub',         s.loginSub);
  setText('lbl-email-l',       s.lblEmailL);
  setText('lbl-pass-l',        s.lblPassL);
  setText('login-btn',         s.loginBtn);
  setText('register-title',    s.registerTitle);
  setText('register-sub',      s.registerSub);
  setText('lbl-name',          s.lblName);
  setText('lbl-email-r',       s.lblEmailR);
  setText('domain-hint',       s.domainHint);
  setText('lbl-pass-r',        s.lblPassR);
  setText('lbl-confirm-pass',  s.lblConfirmPass);
  setText('register-btn',      s.registerBtn);
  setText('verify-title',      s.verifyTitle);
  setText('verify-sub',        s.verifySub);
  setText('verify-btn',        s.verifyBtn);
  setText('resend-btn',        s.resendBtn);
  setText('card-footer',       s.footer);
  setText('bottom-text',       s.bottom);

  // Update password toggle button labels
  document.querySelectorAll('.password-toggle').forEach(btn => {
    const input = btn.closest('.password-wrap')?.querySelector('input');
    btn.textContent = input?.type === 'password' ? t('show') : t('hide');
  });
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ─── Tab Navigation ───────────────────────────────────────────────────────────
function showTab(tab) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.form-panel').forEach(el => el.classList.remove('active'));
  document.getElementById(`tab-${tab}`)?.classList.add('active');
  document.getElementById(`panel-${tab}`)?.classList.add('active');
}

// ─── Message Banners ──────────────────────────────────────────────────────────
function showMsg(id, text, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className   = `msg-banner ${type}`;
}

// ─── Password Toggle ──────────────────────────────────────────────────────────
function togglePassword(inputId, btn) {
  const input   = document.getElementById(inputId);
  const hidden  = input.type === 'password';
  input.type    = hidden ? 'text' : 'password';
  btn.textContent = hidden ? t('hide') : t('show');
}

// ─── Verify code helpers ──────────────────────────────────────────────────────
function getCode() {
  return Array.from(document.querySelectorAll('.code-slot'))
    .map(s => s.value)
    .join('');
}

function openVerifyPanel(email) {
  pendingEmail = email;
  const hint = document.getElementById('verify-email-hint');
  if (hint) hint.textContent = email;
  document.querySelectorAll('.code-slot').forEach(slot => {
    slot.value = '';
    slot.classList.remove('filled');
  });
  showTab('verify');
  document.querySelector('.code-slot')?.focus();
}

// ─── Auth Actions ─────────────────────────────────────────────────────────────
async function doLogin() {
  const email    = document.getElementById('login-email').value.trim();
  const password = document.getElementById('login-password').value;

  if (!email || !password) return showMsg('login-msg', t('fillAll'), 'error');

  const btn = document.getElementById('login-btn');
  setLoading(btn, true, t('signingIn'), t('loginBtn'));

  try {
    const data = await apiFetch('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    setToken(data.token);
    showMsg('login-msg', t('loginOk'), 'success');
    setTimeout(() => { window.location.href = '/'; }, 800);
  } catch (err) {
    showMsg('login-msg', err.message === 'network_error' ? t('connErr') : err.message, 'error');
    setLoading(btn, false, null, t('loginBtn'));
  }
}

async function doRegister() {
  const fullName        = document.getElementById('full-name').value.trim();
  const email           = document.getElementById('register-email').value.trim();
  const password        = document.getElementById('register-password').value;
  const confirmPassword = document.getElementById('register-confirm-password').value;

  if (!fullName || !email || !password || !confirmPassword)
    return showMsg('register-msg', t('fillAll'), 'error');
  if (password !== confirmPassword)
    return showMsg('register-msg', t('passwordMismatch'), 'error');
  if (password.length < 6)
    return showMsg('register-msg', t('passwordTooShort'), 'error');

  const btn = document.getElementById('register-btn');
  setLoading(btn, true, t('creating'), t('registerBtn'));

  try {
    await apiFetch('/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({ full_name: fullName, email, password }),
    });
    showMsg('register-msg', t('registerOk'), 'success');
    setTimeout(() => openVerifyPanel(email), 600);
  } catch (err) {
    showMsg('register-msg', err.message === 'network_error' ? t('connErr') : (err.message || t('registerFailed')), 'error');
  } finally {
    setLoading(btn, false, null, t('registerBtn'));
  }
}

async function doVerify() {
  if (!pendingEmail) { showTab('register'); return; }
  const code = getCode();
  if (code.length !== 6) return showMsg('verify-msg', t('fullCode'), 'error');

  const btn = document.getElementById('verify-btn');
  setLoading(btn, true, '…', t('verifyBtn'));

  try {
    await apiFetch('/api/auth/verify', {
      method: 'POST',
      body: JSON.stringify({ email: pendingEmail, code }),
    });
    showMsg('verify-msg', t('verifyOk'), 'success');
    setTimeout(() => showTab('login'), 1500);
  } catch (err) {
    showMsg('verify-msg', err.message || 'Invalid code', 'error');
  } finally {
    setLoading(btn, false, null, t('verifyBtn'));
  }
}

async function doResend() {
  if (!pendingEmail) return;
  try {
    await apiFetch(`/api/auth/resend-code?email=${encodeURIComponent(pendingEmail)}`, { method: 'POST' });
    showMsg('verify-msg', t('resendOk'), 'info');
  } catch {
    showMsg('verify-msg', t('connErr'), 'error');
  }
}

function setLoading(btn, loading, loadingText, defaultText) {
  btn.disabled    = loading;
  btn.textContent = loading ? loadingText : defaultText;
}

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Redirect if already logged in
  if (getToken()) { window.location.href = '/'; return; }

  // Default to Arabic
  setLang('ar');

  // Language buttons
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => setLang(btn.dataset.lang));
  });

  // Tab buttons
  document.getElementById('tab-login')?.addEventListener('click', () => showTab('login'));
  document.getElementById('tab-register')?.addEventListener('click', () => showTab('register'));

  // Auth buttons
  document.getElementById('login-btn')?.addEventListener('click', doLogin);
  document.getElementById('register-btn')?.addEventListener('click', doRegister);
  document.getElementById('verify-btn')?.addEventListener('click', doVerify);
  document.getElementById('resend-btn')?.addEventListener('click', doResend);

  // Password toggles
  document.querySelectorAll('.password-toggle').forEach(btn => {
    const inputId = btn.closest('.password-wrap')?.querySelector('input')?.id;
    if (inputId) btn.addEventListener('click', () => togglePassword(inputId, btn));
  });

  // Code slots behaviour
  const slots = document.querySelectorAll('.code-slot');
  slots.forEach((slot, i) => {
    slot.addEventListener('input', () => {
      slot.value = slot.value.replace(/\D/g, '');
      slot.classList.toggle('filled', slot.value !== '');
      if (slot.value && i < slots.length - 1) slots[i + 1].focus();
    });
    slot.addEventListener('keydown', e => {
      if (e.key === 'Backspace' && !slot.value && i > 0) slots[i - 1].focus();
    });
    slot.addEventListener('paste', e => {
      e.preventDefault();
      const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
      pasted.split('').forEach((ch, j) => {
        if (slots[j]) { slots[j].value = ch; slots[j].classList.add('filled'); }
      });
      slots[Math.min(pasted.length, 5)]?.focus();
    });
  });

  // Enter key shortcuts
  document.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    if (document.getElementById('panel-login')?.classList.contains('active'))    doLogin();
    else if (document.getElementById('panel-register')?.classList.contains('active')) doRegister();
    else if (document.getElementById('panel-verify')?.classList.contains('active'))   doVerify();
  });
});