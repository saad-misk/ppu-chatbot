"""
RAG Generator — bilingual Arabic/English reply generation via HuggingFace Inference API.

The system prompt is in Arabic (the primary language).
The model detects the user's language and replies in the same language.

Model: mistralai/Mistral-7B-Instruct-v0.2
  • Strong Arabic support with Mistral instruction format
  • Alternatively use: google/gemma-2-9b-it or CohereForAI/aya-23-8B
    (Aya is specifically optimized for Arabic and multilingual tasks)
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Dict

import httpx

from shared.config.settings import settings
from shared.utils.lang import is_arabic as _is_arabic

logger = logging.getLogger(__name__)

_HF_API_URL       = "https://api-inference.huggingface.co/models/{model}"
_GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# ---------------------------------------------------------------------------
# System prompts — one per language so scaffolding stays consistent
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_AR = (
    "أنت مساعد ذكي لجامعة بوليتكنك فلسطين (PPU).\n"
    "مهمتك هي الإجابة على أسئلة الطلاب بشكل دقيق ومفيد.\n"
    "أجب دائماً بنفس لغة السؤال: إذا كان السؤال بالعربية أجب بالعربية، "
    "وإذا كان بالإنجليزية أجب بالإنجليزية.\n"
    "استخدم فقط المعلومات الواردة في السياق المقدم أدناه. لا تخترع أو تخمن أي معلومات.\n"
    "إذا لم تجد الإجابة في السياق قل: "
    "\"لا تتوفر لديّ هذه المعلومات في قاعدة البيانات، يرجى التواصل مع القسم المعني.\"\n"
    "كن موجزاً ودقيقاً في إجاباتك."
)

_SYSTEM_PROMPT_EN = (
    "You are an AI assistant for Palestine Polytechnic University (PPU).\n"
    "Answer student questions accurately and helpfully.\n"
    "Always respond in the same language as the question (Arabic or English).\n"
    "Use ONLY the provided context. Do not hallucinate or invent information.\n"
    "If the answer is not in the context, say so politely and suggest "
    "contacting the relevant department."
)

_FALLBACK_AR = (
    "عذراً، لم أتمكن من إيجاد إجابة دقيقة لسؤالك في قاعدة المعرفة. "
    "يُرجى التواصل مع القسم المعني أو زيارة الموقع الرسمي لجامعة بوليتكنك فلسطين."
)

_FALLBACK_EN = (
    "I'm sorry, I couldn't find a confident answer to your question in the knowledge base. "
    "Please contact the relevant department or visit the PPU official website."
)

# ---------------------------------------------------------------------------
# Module-level httpx client — reused across requests (connection pooling)
# ---------------------------------------------------------------------------

_HTTP_CLIENT: httpx.AsyncClient | None = None


async def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=60.0)
    return _HTTP_CLIENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(query: str, context_chunks: List[Dict], history: List[Dict]) -> str:
    """
    Build a Mistral-format [INST] prompt with bilingual context.

    Mistral-7B-Instruct-v0.2 uses:
        <s>[INST] {system + user message} [/INST]

    It does NOT use <<SYS>> / <</SYS>> tags — those belong to Llama-2.
    Language-aware section headers keep the model's in-context reasoning
    aligned with the query language.
    """
    arabic = _is_arabic(query)

    if arabic:
        system_prompt = _SYSTEM_PROMPT_AR
        ctx_header    = "### السياق"
        hist_header   = "### سجل المحادثة"
        q_header      = "### السؤال"
        user_label    = "الطالب"
        bot_label     = "المساعد"
        no_ctx_msg    = "لا يوجد سياق متاح."
    else:
        system_prompt = _SYSTEM_PROMPT_EN
        ctx_header    = "### Context"
        hist_header   = "### Conversation History"
        q_header      = "### Question"
        user_label    = "Student"
        bot_label     = "Assistant"
        no_ctx_msg    = "No context available."

    # Build context block
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta      = chunk.get("metadata", {})
        doc       = meta.get("doc_name", "unknown")
        page      = meta.get("page", "?")
        score     = chunk.get("score", "")
        score_str = f" ({score:.0%})" if score else ""
        context_parts.append(
            f"[{i} — {doc}, p.{page}{score_str}]\n{chunk['document']}"
        )

    context_text = "\n\n".join(context_parts) if context_parts else no_ctx_msg

    # Build history block (last 4 turns only)
    history_text = ""
    for turn in history[-4:]:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        prefix  = user_label if role == "user" else bot_label
        history_text += f"{prefix}: {content}\n"

    # Mistral [INST] format — no <<SYS>> wrapper needed
    prompt = f"<s>[INST] {system_prompt}\n\n{ctx_header}:\n{context_text}\n\n"
    if history_text:
        prompt += f"{hist_header}:\n{history_text}\n"
    prompt += f"{q_header}:\n{query} [/INST]"
    return prompt


# ---------------------------------------------------------------------------
# Response post-processing
# ---------------------------------------------------------------------------

_PROMPT_ARTIFACTS = (
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "<s>", "</s>",
    "### السياق", "### Context",
    "### السؤال", "### Question",
    "### سجل المحادثة", "### Conversation History",
)


def _clean_response(text: str, fallback: str) -> str:
    """Strip prompt artifacts, trim whitespace, enforce a 2 000-char max."""
    for artifact in _PROMPT_ARTIFACTS:
        text = text.replace(artifact, "")
    text = text.strip()
    if not text:
        return fallback
    if len(text) > 2000:
        cutoff = text.rfind(".", 0, 2000)
        text = text[: cutoff + 1] if cutoff > 1500 else text[:2000] + "…"
    return text


# ---------------------------------------------------------------------------
# Main generation coroutine
# ---------------------------------------------------------------------------

async def generate(
    query: str,
    context_chunks: List[Dict],
    history: List[Dict] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> str:
    """
    Generate a bilingual grounded reply via the HuggingFace Inference API.

    Returns Arabic reply by default; English if the query is in English.
    Falls back gracefully on any API error.
    Retries once on HTTP 503 (model still loading on HF side).
    """
    fallback = _FALLBACK_AR if _is_arabic(query) else _FALLBACK_EN

    if not settings.HF_INFERENCE_API_KEY:
        logger.error("HF_INFERENCE_API_KEY is not set — cannot call Inference API.")
        return fallback

    # Use generation model, not the BERT classifier model name from settings
    model_name = (
        settings.HF_MODEL_NAME
        if settings.HF_MODEL_NAME not in ("bert-base-uncased",)
        else _GENERATION_MODEL
    )

    url     = _HF_API_URL.format(model=model_name)
    prompt  = _build_prompt(query, context_chunks, history or [])
    headers = {
        "Authorization": f"Bearer {settings.HF_INFERENCE_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":   max_new_tokens,
            "temperature":      temperature,
            "return_full_text": False,
            "do_sample":        temperature > 0,
        },
    }

    client = await _get_http_client()

    for attempt in range(2):   # retry once on transient 503
        try:
            response = await client.post(url, headers=headers, json=payload)

            if response.status_code == 503 and attempt == 0:
                logger.warning("HF API 503 (model loading) — retrying in 3 s…")
                await asyncio.sleep(3)
                continue

            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and data:
                raw = data[0].get("generated_text", "")
                return _clean_response(raw, fallback)

            logger.warning("Unexpected HF API response format: %s", str(data)[:200])
            return fallback

        except httpx.HTTPStatusError as e:
            logger.error(
                "HF API HTTP %d: %s",
                e.response.status_code,
                e.response.text[:200],
            )
            return fallback
        except Exception as e:
            logger.error("Generation error: %s", e)
            return fallback

    return fallback
