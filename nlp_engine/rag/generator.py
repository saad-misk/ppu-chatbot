from __future__ import annotations

import logging
from typing import List, Dict

import requests
from openai import OpenAI

from shared.config.settings import settings
from shared.utils.lang import is_arabic as _is_arabic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider model registry
# ---------------------------------------------------------------------------

_PROVIDER_MODELS = {

    # -------------------------------------------------------
    # OpenRouter models
    # -------------------------------------------------------

    "openrouter": [
        "meta-llama/llama-3.3-70b-instruct:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
    ],

    # -------------------------------------------------------
    # NVIDIA models
    # -------------------------------------------------------

    "nvidia": [
        "qwen/qwen3.5-122b-a10b",
        "meta/llama-3.1-70b-instruct",
    ],
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_AR = """\
أنت مساعد أكاديمي لجامعة بوليتكنك فلسطين.

ستتلقى:
- سياق مستخرج من ملفات الجامعة
- سؤال الطالب

التعليمات:
1. ابحث داخل السياق عن الإجابة بشكل مباشر.
2. إذا وُجدت المعلومة بشكل جزئي، استخرج أفضل إجابة ممكنة.
3. لا تقل "لا تتوفر لديّ هذه المعلومات" إلا إذا كانت المعلومة غير موجودة فعلاً.
4. أجب بنفس لغة السؤال.
5. لا تخترع معلومات غير موجودة في السياق.
6. كن مختصراً وواضحاً.
"""

_SYSTEM_PROMPT_EN = """\
You are an academic assistant for Palestine Polytechnic University (PPU).

You will receive:
- Context extracted from university documents
- A student question

Instructions:
1. Search the context carefully for the answer.
2. If the information exists partially, provide the best possible answer from the context.
3. Only say the information is unavailable if it truly does not exist in the context.
4. Reply in the same language as the question.
5. Never invent information.
6. Be concise and clear.
"""

# ---------------------------------------------------------------------------
# Fallback responses
# ---------------------------------------------------------------------------

_FALLBACK_AR = (
    "عذراً، حدث خطأ أثناء توليد الإجابة. "
    "يرجى التواصل مع القسم المعني مباشرةً."
)

_FALLBACK_EN = (
    "Sorry, an error occurred while generating the answer. "
    "Please contact the relevant department directly."
)

# ---------------------------------------------------------------------------
# OpenRouter client cache
# ---------------------------------------------------------------------------

_openrouter_client: OpenAI | None = None


def _get_openrouter_client() -> OpenAI:
    global _openrouter_client

    if _openrouter_client is None:

        api_key = getattr(settings, "OPENROUTER_API_KEY", None)

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=0,
            timeout=20.0,
        )

        logger.info("Initialized OpenRouter client")

    return _openrouter_client


# ---------------------------------------------------------------------------
# NVIDIA request helper
# ---------------------------------------------------------------------------

def _call_nvidia(
    model: str,
    messages: list,
    temperature: float,
    max_tokens: int,
) -> str:

    api_key = getattr(settings, "NVIDIA_API_KEY", None)

    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "stream": False,
    }

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=25,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _build_user_message(
    query: str,
    context_chunks: List[Dict],
    history: List[Dict],
    arabic: bool,
) -> str:

    if arabic:
        ctx_header = "السياق"
        hist_header = "سجل المحادثة السابق"
        q_header = "السؤال"
        user_label = "الطالب"
        bot_label = "المساعد"
        no_ctx_msg = "لا يوجد سياق متاح."
    else:
        ctx_header = "Context"
        hist_header = "Previous conversation"
        q_header = "Question"
        user_label = "Student"
        bot_label = "Assistant"
        no_ctx_msg = "No context available."

    # -------------------------------------------------------
    # Context
    # -------------------------------------------------------

    context_parts = []

    for i, chunk in enumerate(context_chunks, 1):

        meta = chunk.get("metadata", {})

        doc = meta.get("doc_name", "unknown")
        page = meta.get("page", "?")

        context_parts.append(
            f"[{i} — {doc}, p.{page}]\n{chunk['document']}"
        )

    context_text = (
        "\n\n".join(context_parts)
        if context_parts
        else no_ctx_msg
    )

    # -------------------------------------------------------
    # History
    # -------------------------------------------------------

    history_lines = []

    for turn in history[-4:]:

        role = turn.get("role", "user")
        content = turn.get("content", "").strip()

        prefix = (
            user_label
            if role == "user"
            else bot_label
        )

        history_lines.append(
            f"{prefix}: {content}"
        )

    history_text = "\n".join(history_lines)

    # -------------------------------------------------------
    # Final message
    # -------------------------------------------------------

    msg = f"### {ctx_header}\n{context_text}\n"

    if history_text:
        msg += (
            f"\n### {hist_header}\n"
            f"{history_text}\n"
        )

    msg += f"\n### {q_header}\n{query}"
    print("Generated user message:\n", msg)
    return msg


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate(
    query: str,
    context_chunks: List[Dict],
    history: List[Dict] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:

    arabic = _is_arabic(query)

    fallback = (
        _FALLBACK_AR
        if arabic
        else _FALLBACK_EN
    )

    system_prompt = (
        _SYSTEM_PROMPT_AR
        if arabic
        else _SYSTEM_PROMPT_EN
    )

    provider = (
        getattr(settings, "LLM_PROVIDER", "openrouter")
        .lower()
        .strip()
    )

    models = _PROVIDER_MODELS.get(provider)

    if not models:
        raise ValueError(
            f"Unsupported provider: {provider}"
        )

    # Optional single-model override
    custom_model = getattr(settings, "LLM_MODEL", None)

    if custom_model:
        models = [custom_model]

    user_message = _build_user_message(
        query=query,
        context_chunks=context_chunks,
        history=history or [],
        arabic=arabic,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]

    last_error = None

    # -----------------------------------------------------------------------
    # Try models in fallback order
    # -----------------------------------------------------------------------

    for model_name in models:

        try:

            logger.info(
                "Trying provider=%s model=%s query='%s'",
                provider,
                model_name,
                query[:60],
            )

            # ---------------------------------------------------------------
            # OpenRouter
            # ---------------------------------------------------------------

            if provider == "openrouter":

                client = _get_openrouter_client()

                response = client.chat.completions.create(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    messages=messages,
                )

                raw = (
                    response
                    .choices[0]
                    .message.content
                    .strip()
                )

            # ---------------------------------------------------------------
            # NVIDIA
            # ---------------------------------------------------------------

            elif provider == "nvidia":

                raw = _call_nvidia(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )

            else:
                raise ValueError(
                    f"Unsupported provider: {provider}"
                )

            # ---------------------------------------------------------------
            # Empty response protection
            # ---------------------------------------------------------------

            if not raw:

                logger.warning(
                    "Empty response from model=%s",
                    model_name,
                )

                continue

            # ---------------------------------------------------------------
            # Limit response length
            # ---------------------------------------------------------------

            if len(raw) > 2000:

                cutoff = raw.rfind(".", 0, 2000)

                raw = (
                    raw[: cutoff + 1]
                    if cutoff > 1500
                    else raw[:2000] + "…"
                )

            logger.info(
                "Successful response from provider=%s model=%s",
                provider,
                model_name,
            )

            return raw

        except Exception as e:

            last_error = e

            logger.warning(
                "Model failed | provider=%s | model=%s | error=%s",
                provider,
                model_name,
                e,
            )

            continue

    # -----------------------------------------------------------------------
    # All models failed
    # -----------------------------------------------------------------------

    logger.error(
        "All models failed | provider=%s | last_error=%s",
        provider,
        last_error,
    )

    return fallback

def _truncate_context_for_prompt(
    context_chunks: List[Dict],
    max_chars: int = 4000,
) -> List[Dict]:
    """Ensure context fits in model window."""
    total = 0
    truncated = []
    
    for chunk in sorted(
        context_chunks,
        key=lambda x: x.get("score", x.get("hybrid_score", 0)),
        reverse=True,
    ):
        doc_len = len(chunk["document"])
        if total + doc_len > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                truncated.append({
                    **chunk,
                    "document": chunk["document"][:remaining] + "..."
                })
            break
        total += doc_len
        truncated.append(chunk)
        if len(truncated) >= 8:  # Max chunks
            break
    
    return truncated


# Add validation function:
def _validate_response(response: str, arabic: bool, context: List[Dict]) -> bool:
    """Check if response is valid."""
    if not response or len(response) < 10:
        return False
    
    # Check for common LLM refusal patterns
    arabic_refusals = [
        "لا يمكنني", "لا استطيع", "ليس لدي", "غير متوفر",
        "غير متاحة", "لا أملك", "لا تتوفر",
    ]
    english_refusals = [
        "i cannot", "i can't", "i am not able", "not available",
        "i don't have", "no information", "cannot provide",
    ]
    
    refusals = arabic_refusals if arabic else english_refusals
    for pattern in refusals:
        if pattern in response.lower():
            # Only flag if context was actually provided
            if context and len(context) > 0:
                logger.warning(f"Refusal pattern '{pattern}' in response despite context")
                return False
    
    return True