/**
 * i18n.js — Translation strings and language helpers.
 */

const translations = {
  ar: {
    newChat:       '+ محادثة جديدة',
    untitled:      'محادثة جديدة',
    sessions:      'المحادثات الأخيرة',
    placeholder:   'اسأل عن الرسوم، الجداول، التسجيل...',
    hint:          'اضغط Enter للإرسال · Shift+Enter لسطر جديد',
    status:        'متصل',
    welcomeTitle:  'مساعد جامعة PPU',
    welcomeSub:    'دليلك الذكي لجامعة بوليتكنك فلسطين. اسأل عن الرسوم أو الجداول أو التسجيل أو الأقسام.',
    suggestions:   [
      'ما هي رسوم تخصص الحاسوب؟',
      'متى يفتح باب التسجيل؟',
      'من هو عميد كلية الهندسة؟',
      'أين قسم تكنولوجيا المعلومات؟',
    ],
    connectionError: 'حدث خطأ في الاتصال. حاول مرة أخرى.',
    lowConf:         'لست متأكداً تماماً من هذه الإجابة. يرجى التحقق مع الجامعة.',
    userRole:        'مستخدم',
    adminRole:       'مدير النظام',
    guestName:       'زائر',
    guestRole:       'غير مسجّل',
    loginPrompt:     'سجّل الدخول لحفظ محادثاتك',
    loginBtn:        'تسجيل الدخول',
    logoutBtn:       'تسجيل الخروج',
    deleteConfirm:   'هل أنت متأكد من حذف هذه المحادثة؟',
  },
  en: {
    newChat:       '+ New Chat',
    untitled:      'New Chat',
    sessions:      'Recent Chats',
    placeholder:   'Ask about fees, schedules, registration...',
    hint:          'Press Enter to send · Shift+Enter for new line',
    status:        'Online',
    welcomeTitle:  'PPU Assistant',
    welcomeSub:    'Your smart guide to Palestine Polytechnic University. Ask about fees, schedules, registration, or departments.',
    suggestions:   [
      'What are the CS tuition fees?',
      'When does registration open?',
      'Who is the dean of Engineering?',
      'Where is the IT department?',
    ],
    connectionError: 'Connection error. Please try again.',
    lowConf:         'I am not fully certain about this answer. Please verify with the university.',
    userRole:        'User',
    adminRole:       'Administrator',
    guestName:       'Guest',
    guestRole:       'Not signed in',
    loginPrompt:     'Sign in to save your chats',
    loginBtn:        'Sign In',
    logoutBtn:       'Logout',
    deleteConfirm:   'Delete this conversation?',
  },
};

let currentLang = localStorage.getItem('chat_lang') || 'ar';

/** Get translated string */
export function t(key) {
  return translations[currentLang]?.[key] ?? key;
}

/** Get current language code */
export function getLang() {
  return currentLang;
}

/**
 * Apply a language: updates DOM direction, lang-btn states,
 * and calls an optional callback for page-specific updates.
 */
export function setLang(lang, onChanged) {
  if (!translations[lang]) return;
  currentLang = lang;
  localStorage.setItem('chat_lang', lang);
  document.documentElement.lang = lang;
  document.body.classList.toggle('rtl', lang === 'ar');

  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });

  onChanged?.();
}