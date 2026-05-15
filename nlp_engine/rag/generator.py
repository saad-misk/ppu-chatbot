"""
nlp_engine/rag/generator.py
===========================
LLM generation layer for the PPU chatbot.

Fixes over v1
-------------
1. System prompt rewritten — removed "be concise", added explicit instruction
   to use the context and answer in full when information is available.
2. History cleaning — error turns stripped before building the prompt so
   repeated failures don't corrupt the conversation state.
3. _validate_response fixed — refusal check was incorrectly blocking valid
   responses that happened to contain Arabic negation words mid-sentence.
   Now only flags responses that are ONLY a refusal with nothing else.
4. _truncate_context_for_prompt now called inside generate() — it was
   defined but never used in the original.
5. Raw response logging added — makes it easy to distinguish model problems
   from post-processing problems.
6. Minimal response guard — if response is under 20 chars and context is
   non-empty, it's treated as a failed generation and the next model is tried.
7. History window increased to 6 turns (3 exchanges) — 4 was cutting off
   context too aggressively.
8. doc_name truncated to 8 chars in context header — was showing full
   filename which wastes tokens with no benefit to the model.
"""
from __future__ import annotations

import logging
from typing import List, Dict

import requests
from openai import OpenAI

from shared.config.settings import settings
from shared.utils.lang import is_arabic as _is_arabic

logger = logging.getLogger(__name__)

# ── Provider model registry ───────────────────────────────────────────────────
_PROVIDER_MODELS = {
    "openrouter": [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-2-9b-it:free",
        "openrouter/free"       # moved to 3rd — least reliable
    ],
    "nvidia": [
        "qwen/qwen3.5-122b-a10b",
        "meta/llama-3.1-70b-instruct",
    ],
    "gemini": [
        "gemini-2.0-flash",
    ],
}

# ── System prompts ────────────────────────────────────────────────────────────
# FIX: removed "كن مختصراً" (be concise) — this was causing the model to
# give one-word answers when the context was rich.
# Added explicit instruction: answer fully using what is in the context.
_SYSTEM_PROMPT_AR = """\
أنت مساعد أكاديمي ذكي لجامعة بوليتكنك فلسطين.
ستتلقى:
- سياق مستخرج من ملفات ووثائق الجامعة
- سؤال الطالب

التعليمات:
1. اقرأ السياق بعناية واستخرج منه كل المعلومات المتعلقة بالسؤال.
2. أجب بشكل كامل ومفيد بناءً على ما هو موجود في السياق.
   - إذا كان السياق يحتوي على اسم، منصب، بريد إلكتروني، تخصص، أو أي تفاصيل — اذكرها جميعاً.
   - لا تختصر الإجابة إذا كان السياق يحتوي على معلومات إضافية مفيدة.
3. إذا وُجدت المعلومة بشكل جزئي، قدّم أفضل إجابة ممكنة مما هو متاح.
4. قل "لا تتوفر هذه المعلومات في الوثائق المتاحة" فقط إذا كانت المعلومة غائبة تماماً عن السياق.
5. أجب بنفس لغة السؤال (عربي أو إنجليزي).
6. لا تخترع أي معلومات غير موجودة في السياق.
"""

_SYSTEM_PROMPT_EN = """\
You are an intelligent academic assistant for Palestine Polytechnic University (PPU).
You will receive:
- Context extracted from university documents and files
- A student question

Instructions:
1. Read the context carefully and extract all information relevant to the question.
2. Answer fully and helpfully based on what is in the context.
   - If the context contains a name, title, email, specialization, or any details — mention all of them.
   - Do not shorten your answer if the context contains additional useful information.
3. If the information exists partially, provide the best possible answer from what is available.
4. Only say the information is unavailable if it truly does not exist anywhere in the context.
5. Reply in the same language as the question (Arabic or English).
6. Never invent information not present in the context.
"""

# ── Fallback responses ────────────────────────────────────────────────────────
_FALLBACK_AR = (
    "عذراً، حدث خطأ أثناء توليد الإجابة. "
    "يرجى التواصل مع القسم المعني مباشرةً."
)
_FALLBACK_EN = (
    "Sorry, an error occurred while generating the answer. "
    "Please contact the relevant department directly."
)

# ── OpenRouter client cache ───────────────────────────────────────────────────
_openrouter_client: OpenAI | None = None

def _get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        api_key = getattr(settings, "OPENROUTER_API_KEY", None)
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            timeout=30.0,   # increased from 20 — free models are slow
        )
        logger.info("Initialized OpenRouter client")
    return _openrouter_client


# ── NVIDIA helper ─────────────────────────────────────────────────────────────
def _call_nvidia(model: str, messages: list,
                 temperature: float, max_tokens: int) -> str:
    api_key = getattr(settings, "NVIDIA_API_KEY", None)
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Accept": "application/json"},
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens,
              "top_p": 0.95, "stream": False},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ── Gemini helper ─────────────────────────────────────────────────────────────
def _call_gemini(model: str, messages: list,
                 temperature: float, max_tokens: int) -> str:
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    system_text = ""
    user_parts: list[str] = []
    for msg in messages:
        role    = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_text = content
        else:
            user_parts.append(content)

    if not user_parts:
        raise ValueError("No user content provided for Gemini.")

    payload: dict = {
        "contents": [{"role": "user",
                      "parts": [{"text": "\n\n".join(user_parts)}]}],
        "generationConfig": {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        params={"key": api_key},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data       = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini response missing candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Context truncation
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_context(
    context_chunks: List[Dict],
    max_chars: int = 4000,
) -> List[Dict]:
    """
    Keep highest-scored chunks that fit within max_chars.
    FIX: this was defined in v1 but never called inside generate().
    Now called before building the user message.
    """
    sorted_chunks = sorted(
        context_chunks,
        key=lambda x: x.get("score", x.get("hybrid_score", 0.0)),
        reverse=True,
    )
    total     = 0
    truncated = []

    for chunk in sorted_chunks:
        doc_len = len(chunk.get("document", ""))
        if total + doc_len > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                truncated.append({
                    **chunk,
                    "document": chunk["document"][:remaining] + "…",
                })
            break
        total += doc_len
        truncated.append(chunk)
        if len(truncated) >= 8:
            break

    return truncated
_truncate_context_for_prompt = _truncate_context


# ══════════════════════════════════════════════════════════════════════════════
# History cleaning
# ══════════════════════════════════════════════════════════════════════════════

# FIX: error turns in history were confusing the model and causing
# minimal/empty responses. Strip them before building the prompt.
_ERROR_MARKERS = [
    "حدث خطأ", "عذراً، حدث", "error occurred", "sorry, an error",
]

def _clean_history(history: List[Dict]) -> List[Dict]:
    """
    Remove turns that contain error messages.
    Also remove the user turn that immediately precedes an error turn,
    since a question followed by an error is better dropped entirely
    than shown as context to the model.
    """
    if not history:
        return []

    cleaned: List[Dict] = []
    skip_next = False

    for i, turn in enumerate(history):
        content = (turn.get("content") or "").lower()

        if skip_next:
            skip_next = False
            continue

        is_error = any(m in content for m in _ERROR_MARKERS)
        if is_error:
            # Also drop the preceding user turn if present
            if cleaned and cleaned[-1].get("role") == "user":
                cleaned.pop()
            continue

        # Look ahead: if the next turn is an error, skip this one too
        if i + 1 < len(history):
            next_content = (history[i + 1].get("content") or "").lower()
            if any(m in next_content for m in _ERROR_MARKERS):
                skip_next = True
                continue

        cleaned.append(turn)

    # Keep last 6 turns (3 exchanges) — increased from 4
    return cleaned[-6:]


# ══════════════════════════════════════════════════════════════════════════════
# Message builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_user_message(
    query:          str,
    context_chunks: List[Dict],
    history:        List[Dict],
    arabic:         bool,
) -> str:
    if arabic:
        ctx_header  = "السياق"
        hist_header = "سجل المحادثة السابق"
        q_header    = "السؤال"
        user_label  = "الطالب"
        bot_label   = "المساعد"
        no_ctx_msg  = "لا يوجد سياق متاح."
    else:
        ctx_header  = "Context"
        hist_header = "Previous conversation"
        q_header    = "Question"
        user_label  = "Student"
        bot_label   = "Assistant"
        no_ctx_msg  = "No context available."

    # ── Context ───────────────────────────────────────────────────────────
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta    = chunk.get("metadata", {})
        # FIX: use short doc id instead of full filename — saves tokens
        doc_id  = chunk.get("id", meta.get("doc_name", "?"))[:8]
        page    = meta.get("page", "?")
        context_parts.append(
            f"[{i} — {doc_id}, p.{page}]\n{chunk['document']}"
        )

    context_text = (
        "\n\n".join(context_parts) if context_parts else no_ctx_msg
    )

    # ── History ───────────────────────────────────────────────────────────
    history_lines = []
    for turn in history:
        role    = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        prefix  = user_label if role == "user" else bot_label
        history_lines.append(f"{prefix}: {content}")

    history_text = "\n".join(history_lines)

    # ── Assemble ──────────────────────────────────────────────────────────
    msg = f"### {ctx_header}\n{context_text}\n"
    if history_text:
        msg += f"\n### {hist_header}\n{history_text}\n"
    msg += f"\n### {q_header}\n{query}"

    logger.debug("User message length: %d chars", len(msg))
    return msg


# ══════════════════════════════════════════════════════════════════════════════
# Response validation
# ══════════════════════════════════════════════════════════════════════════════

# FIX: original _validate_response was too aggressive — it flagged any response
# containing an Arabic negation word as a refusal, even if the response was
# a full paragraph that happened to contain "لا" or "ليس".
# New approach: only flag if the response is SHORT and matches a refusal pattern,
# meaning it's ONLY a refusal with nothing substantive.
_REFUSAL_PATTERNS_AR = [
    "لا تتوفر", "لا يوجد", "لا أملك", "لا أعرف",
    "غير متوفر", "غير متاح", "ليس لدي معلومات",
    "لا يمكنني الإجابة",
]
_REFUSAL_PATTERNS_EN = [
    "i cannot", "i can't", "not available", "i don't have",
    "no information", "cannot provide", "i am unable",
    "i do not have information",
]

def _is_refusal(response: str, arabic: bool) -> bool:
    """
    Returns True only if the response is short AND purely a refusal.
    A 500-char response that contains "لا يوجد" mid-sentence is NOT a refusal.
    """
    if len(response) > 150:
        # Long responses are almost never pure refusals
        return False
    patterns = _REFUSAL_PATTERNS_AR if arabic else _REFUSAL_PATTERNS_EN
    r_lower  = response.lower()
    return any(p in r_lower for p in patterns)


def _is_minimal_response(response: str, context_chunks: List[Dict]) -> bool:
    """
    Detect suspiciously short responses when context is available.
    A 5-word answer when 3 rich context chunks were provided suggests
    the model failed (timeout, rate limit, etc.) rather than genuinely
    having nothing to say.
    """
    if not context_chunks:
        return False   # no context → short answer may be correct
    return len(response.strip()) < 20


# ══════════════════════════════════════════════════════════════════════════════
# Main generation function
# ══════════════════════════════════════════════════════════════════════════════

def generate(
    query:          str,
    context_chunks: List[Dict],
    history:        List[Dict] | None = None,
    max_new_tokens: int   = 512,
    temperature:    float = 0.3,
) -> str:
    arabic   = _is_arabic(query)
    fallback = _FALLBACK_AR if arabic else _FALLBACK_EN

    system_prompt = _SYSTEM_PROMPT_AR if arabic else _SYSTEM_PROMPT_EN

    provider = getattr(settings, "LLM_PROVIDER", "openrouter").lower().strip()
    models   = _PROVIDER_MODELS.get(provider)
    if not models:
        raise ValueError(f"Unsupported provider: {provider}")

    custom_model = getattr(settings, "LLM_MODEL", None)
    if custom_model:
        models = [custom_model]

    # FIX 1: truncate context BEFORE building the message (was never called)
    trimmed_chunks = _truncate_context(context_chunks, max_chars=4000)

    # FIX 2: clean history before building the message
    clean_history = _clean_history(history or [])

    user_message = _build_user_message(
        query=query,
        context_chunks=trimmed_chunks,
        history=clean_history,
        arabic=arabic,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]
    print("generated user messageL", user_message)
    last_error = None

    for model_name in models:
        try:
            logger.info("Trying provider=%s model=%s query='%s'",
                        provider, model_name, query[:60])

            if provider == "openrouter":
                client = _get_openrouter_client()
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    messages=messages,
                )
                raw = response.choices[0].message.content or ""
                raw = raw.strip()

            elif provider == "nvidia":
                raw = _call_nvidia(model_name, messages, temperature,
                                   max_new_tokens)

            elif provider == "gemini":
                raw = _call_gemini(model_name, messages, temperature,
                                   max_new_tokens)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # FIX 3: log raw response — critical for diagnosing model issues
            logger.info("Raw response from %s: %r", model_name, raw[:200])

            # Empty response
            if not raw:
                logger.warning("Empty response from model=%s", model_name)
                continue

            # FIX 4: minimal response guard — try next model if answer is
            # suspiciously short given non-empty context
            if _is_minimal_response(raw, trimmed_chunks):
                logger.warning(
                    "Minimal response (%d chars) from model=%s with %d "
                    "context chunks — likely model failure, trying next.",
                    len(raw), model_name, len(trimmed_chunks),
                )
                continue

            # FIX 5: refusal check — only flag short pure-refusal responses
            if _is_refusal(raw, arabic) and trimmed_chunks:
                logger.warning(
                    "Refusal response from model=%s despite %d context chunks.",
                    model_name, len(trimmed_chunks),
                )
                continue

            # Truncate very long responses at sentence boundary
            if len(raw) > 2000:
                cutoff = raw.rfind(".", 0, 2000)
                raw    = raw[:cutoff + 1] if cutoff > 1500 else raw[:2000] + "…"

            logger.info("Success: provider=%s model=%s len=%d",
                        provider, model_name, len(raw))
            return raw

        except Exception as e:
            last_error = e
            logger.warning("Model failed | provider=%s | model=%s | error=%s",
                           provider, model_name, e)
            continue

    logger.error("All models failed | provider=%s | last_error=%s",
                 provider, last_error)
    return fallback