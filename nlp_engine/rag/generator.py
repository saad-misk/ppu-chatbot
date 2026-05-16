"""
nlp_engine/rag/generator.py
===========================
LLM generation layer for the PPU chatbot.

Added Groq support (May 2026)
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
        "openrouter/free"
    ],
    "nvidia": [
        "qwen/qwen3.5-122b-a10b",
        "meta/llama-3.1-70b-instruct",
    ],
    "gemini": [
        "gemini-2.0-flash",
    ],
    # ✅ NEW: Groq provider
    "groq": [
        "llama-3.3-70b-versatile",     # Best quality on Groq
        "llama-3.1-8b-instant",        # Fast and cheap
    ],
}

# ── System prompts ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_AR = """\
أنت مساعد أكاديمي ذكي لجامعة بوليتكنك فلسطين.
ستتلقى:
- سياق مستخرج من ملفات ووثائق الجامعة
- سؤال الطالب

التعليمات:
1. اقرأ السياق بعناية واستخرج منه كل المعلومات المتعلقة بالسؤال.
2. أجب بشكل كامل ومفيد بناءً على ما هو موجود في السياق.
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
            timeout=30.0,
        )
        logger.info("Initialized OpenRouter client")
    return _openrouter_client


# ── NVIDIA helper ─────────────────────────────────────────────────────────────
def _call_nvidia(model: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = getattr(settings, "NVIDIA_API_KEY", None)
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        json={"model": model, "messages": messages, "temperature": temperature,
              "max_tokens": max_tokens, "top_p": 0.95, "stream": False},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ── Gemini helper ─────────────────────────────────────────────────────────────
def _call_gemini(model: str, messages: list, temperature: float, max_tokens: int) -> str:
    api_key = getattr(settings, "GEMINI_API_KEY", None)
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured.")

    system_text = ""
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            system_text = content
        else:
            user_parts.append(content)

    if not user_parts:
        raise ValueError("No user content provided for Gemini.")

    payload = {
        "contents": [{"role": "user", "parts": [{"text": "\n\n".join(user_parts)}]}],
        "generationConfig": {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": 0.95,
        },
    }
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini response missing candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


# ── Groq helper (NEW) ─────────────────────────────────────────────────────────
_groq_client: OpenAI | None = None

def _get_groq_client() -> OpenAI:
    global _groq_client
    if _groq_client is None:
        api_key = getattr(settings, "GROQ_API_KEY", None)
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")
        _groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
            max_retries=2,
            timeout=60.0,
        )
        logger.info("Initialized Groq client")
    return _groq_client


def _call_groq(model: str, messages: list, temperature: float, max_tokens: int) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Context truncation, History cleaning, etc. (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _truncate_context(context_chunks: List[Dict], max_chars: int = 4000) -> List[Dict]:
    sorted_chunks = sorted(
        context_chunks,
        key=lambda x: x.get("score", x.get("hybrid_score", 0.0)),
        reverse=True,
    )
    total, truncated = 0, []
    for chunk in sorted_chunks:
        doc_len = len(chunk.get("document", ""))
        if total + doc_len > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                truncated.append({**chunk, "document": chunk["document"][:remaining] + "…"})
            break
        total += doc_len
        truncated.append(chunk)
        if len(truncated) >= 8:
            break
    return truncated


_truncate_context_for_prompt = _truncate_context


_ERROR_MARKERS = ["حدث خطأ", "عذراً، حدث", "error occurred", "sorry, an error"]


def _clean_history(history: List[Dict]) -> List[Dict]:
    if not history:
        return []
    cleaned, skip_next = [], False
    for i, turn in enumerate(history):
        content = (turn.get("content") or "").lower()
        if skip_next:
            skip_next = False
            continue
        is_error = any(m in content for m in _ERROR_MARKERS)
        if is_error:
            if cleaned and cleaned[-1].get("role") == "user":
                cleaned.pop()
            continue
        if i + 1 < len(history):
            next_content = (history[i + 1].get("content") or "").lower()
            if any(m in next_content for m in _ERROR_MARKERS):
                skip_next = True
                continue
        cleaned.append(turn)
    return cleaned[-6:]


def _build_user_message(query: str, context_chunks: List[Dict],
                        history: List[Dict], arabic: bool) -> str:
    if arabic:
        ctx_header, hist_header, q_header = "السياق", "سجل المحادثة السابق", "السؤال"
        user_label, bot_label, no_ctx_msg = "الطالب", "المساعد", "لا يوجد سياق متاح."
    else:
        ctx_header, hist_header, q_header = "Context", "Previous conversation", "Question"
        user_label, bot_label, no_ctx_msg = "Student", "Assistant", "No context available."

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        doc_id = chunk.get("id", meta.get("doc_name", "?"))[:8]
        page = meta.get("page", "?")
        context_parts.append(f"[{i} — {doc_id}, p.{page}]\n{chunk['document']}")

    context_text = "\n\n".join(context_parts) if context_parts else no_ctx_msg

    history_lines = []
    for turn in history:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            prefix = user_label if role == "user" else bot_label
            history_lines.append(f"{prefix}: {content}")

    history_text = "\n".join(history_lines)
    msg = f"### {ctx_header}\n{context_text}\n"
    if history_text:
        msg += f"\n### {hist_header}\n{history_text}\n"
    msg += f"\n### {q_header}\n{query}"
    return msg


_REFUSAL_PATTERNS_AR = ["لا تتوفر", "لا يوجد", "لا أملك", "لا أعرف", "غير متوفر"]
_REFUSAL_PATTERNS_EN = ["i cannot", "i can't", "not available", "i don't have"]


def _is_refusal(response: str, arabic: bool) -> bool:
    if len(response) > 150:
        return False
    patterns = _REFUSAL_PATTERNS_AR if arabic else _REFUSAL_PATTERNS_EN
    return any(p in response.lower() for p in patterns)


def _is_minimal_response(response: str, context_chunks: List[Dict]) -> bool:
    return bool(context_chunks) and len(response.strip()) < 20


# ══════════════════════════════════════════════════════════════════════════════
# Main generation function
# ══════════════════════════════════════════════════════════════════════════════

def generate(query: str, context_chunks: List[Dict],
             history: List[Dict] | None = None,
             max_new_tokens: int = 512, temperature: float = 0.3) -> str:

    arabic = _is_arabic(query)
    fallback = _FALLBACK_AR if arabic else _FALLBACK_EN
    system_prompt = _SYSTEM_PROMPT_AR if arabic else _SYSTEM_PROMPT_EN

    provider = getattr(settings, "LLM_PROVIDER", "openrouter").lower().strip()
    models = _PROVIDER_MODELS.get(provider)
    if not models:
        raise ValueError(f"Unsupported provider: {provider}")

    custom_model = getattr(settings, "LLM_MODEL", None)
    if custom_model:
        models = [custom_model]

    trimmed_chunks = _truncate_context(context_chunks, max_chars=4000)
    clean_history = _clean_history(history or [])

    user_message = _build_user_message(query, trimmed_chunks, clean_history, arabic)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    print("generated user messageL", user_message)
    last_error = None

    for model_name in models:
        try:
            logger.info("Trying provider=%s model=%s", provider, model_name)

            if provider == "openrouter":
                client = _get_openrouter_client()
                response = client.chat.completions.create(
                    model=model_name, temperature=temperature,
                    max_tokens=max_new_tokens, messages=messages
                )
                raw = response.choices[0].message.content.strip()

            elif provider == "nvidia":
                raw = _call_nvidia(model_name, messages, temperature, max_new_tokens)

            elif provider == "gemini":
                raw = _call_gemini(model_name, messages, temperature, max_new_tokens)

            elif provider == "groq":                    # ✅ NEW
                raw = _call_groq(model_name, messages, temperature, max_new_tokens)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            logger.info("Raw response from %s: %r", model_name, raw[:200])

            if not raw:
                continue
            if _is_minimal_response(raw, trimmed_chunks):
                continue
            if _is_refusal(raw, arabic) and trimmed_chunks:
                continue

            if len(raw) > 2000:
                cutoff = raw.rfind(".", 0, 2000)
                raw = raw[:cutoff + 1] if cutoff > 1500 else raw[:2000] + "…"

            logger.info("Success: provider=%s model=%s", provider, model_name)
            return raw

        except Exception as e:
            last_error = e
            logger.warning("Model failed | provider=%s | model=%s | error=%s",
                           provider, model_name, e)
            continue

    logger.error("All models failed | provider=%s | last_error=%s", provider, last_error)
    return fallback