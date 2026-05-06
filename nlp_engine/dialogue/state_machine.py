"""
Dialogue State Machine — deterministic response routing.

For well-defined intents (greetings, thanks, goodbyes) the bot returns
canned replies without calling the LLM.  For all other intents it signals
that the RAG pipeline should handle the request.

Usage
-----
    from nlp_engine.dialogue.state_machine import StateMachine, RouteDecision

    sm = StateMachine()
    decision = sm.route(intent="greeting", confidence=0.95, entities={})
    if decision.handled:
        return decision.reply         # use canned reply
    else:
        pass                          # fall through to RAG
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from shared.utils.lang import is_arabic as _is_arabic_text


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    handled: bool                    # True  → use reply directly; False → call RAG
    reply: Optional[str]  = None     # populated when handled=True
    reason: str           = ""       # debug label


# ---------------------------------------------------------------------------
# Canned response pools
# ---------------------------------------------------------------------------

_GREETINGS_AR: List[str] = [
    "مرحباً! أنا المساعد الذكي لجامعة بوليتكنك فلسطين. كيف يمكنني مساعدتك اليوم؟",
    "أهلاً وسهلاً! كيف يمكنني خدمتك؟ يمكنني الإجابة على أسئلتك حول الجامعة.",
    "السلام عليكم! أنا هنا لمساعدتك في أي استفسار يتعلق بجامعة بوليتكنك فلسطين.",
]

_GREETINGS_EN: List[str] = [
    "Hello! I'm the PPU assistant. How can I help you today?",
    "Hi there! Welcome to Palestine Polytechnic University's chatbot. What would you like to know?",
    "Greetings! I'm here to help with any PPU-related questions.",
]

_THANKS_AR: List[str] = [
    "عفواً! هل هناك شيء آخر يمكنني مساعدتك به؟",
    "بكل سرور! لا تتردد في طرح أي سؤال آخر.",
    "يسعدني خدمتك! هل تحتاج إلى مزيد من المعلومات؟",
]

_THANKS_EN: List[str] = [
    "You're welcome! Is there anything else I can help you with?",
    "Happy to help! Feel free to ask if you have more questions.",
    "Glad I could assist! Anything else?",
]

_GOODBYES_AR: List[str] = [
    "وداعاً! أتمنى لك يوماً موفقاً في الجامعة.",
    "مع السلامة! لا تتردد في العودة إذا كان لديك أي سؤال.",
    "إلى اللقاء! بالتوفيق في دراستك.",
]

_GOODBYES_EN: List[str] = [
    "Goodbye! Have a great day at PPU!",
    "Farewell! Don't hesitate to come back if you have more questions.",
    "Take care! Best of luck with your studies!",
]

_LOW_CONFIDENCE_AR: List[str] = [
    "لم أفهم سؤالك بشكل كامل. هل يمكنك إعادة صياغته؟",
    "أحتاج إلى مزيد من التفاصيل لأتمكن من مساعدتك. هل يمكنك توضيح سؤالك؟",
    "سؤالك غير واضح بالنسبة لي. هل يمكنك أن تكون أكثر تحديداً؟",
]

_LOW_CONFIDENCE_EN: List[str] = [
    "I'm not sure I understood that correctly. Could you rephrase your question?",
    "I didn't quite catch that. Can you give me more details?",
    "I'm having trouble understanding your question. Could you be more specific?",
]


# ---------------------------------------------------------------------------
# Deterministic intent routing table  (AR / EN pairs)
# ---------------------------------------------------------------------------

_CANNED_INTENTS: Dict[str, tuple] = {
    #  intent       arabic pool          english pool
    "greeting": (_GREETINGS_AR, _GREETINGS_EN),
    "goodbye":  (_GOODBYES_AR,  _GOODBYES_EN),
    "thanks":   (_THANKS_AR,    _THANKS_EN),
}

_UNKNOWN_AR = (
    "لست متأكداً كيف يمكنني مساعدتك في هذا. "
    "يمكنني الإجابة على أسئلة حول الرسوم، التسجيل، الجداول الدراسية، الكادر الأكاديمي، أو الأقسام."
)
_UNKNOWN_EN = (
    "I'm not sure how to help with that. "
    "You can ask me about fees, registration, schedules, staff, or departments."
)


class StateMachine:
    """
    Routes an incoming classified intent to either a canned reply or the RAG pipeline.
    Language-aware: replies in Arabic when the query is Arabic, English otherwise.
    """

    # Canned intents are allowed through at a lower threshold because short
    # social phrases ("مرحبا", "شكرا") reliably classify correctly but with
    # lower softmax confidence than longer factual questions.
    _CANNED_CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold

    def route(
        self,
        intent: str,
        confidence: float,
        entities: Dict,
        query: str = "",
    ) -> RouteDecision:
        """
        Determine how to handle this turn.

        Parameters
        ----------
        intent     : classified intent label
        confidence : classifier confidence score
        entities   : extracted entity dict
        query      : original user message (used for language detection)

        Returns
        -------
        RouteDecision
            .handled=True  → reply is ready, skip RAG
            .handled=False → call RAG pipeline
        """
        arabic = _is_arabic_text(query)

        # 1. Canned intents (greeting / thanks / goodbye) — checked FIRST.
        #    These are handled even at low confidence because the classifier
        #    is almost always correct for such short social phrases; the low
        #    score is an artefact of the softmax spread, not a real mistake.
        if intent in _CANNED_INTENTS and confidence >= self._CANNED_CONFIDENCE_THRESHOLD:
            ar_pool, en_pool = _CANNED_INTENTS[intent]
            pool = ar_pool if arabic else en_pool
            return RouteDecision(
                handled=True,
                reply=random.choice(pool),
                reason=f"canned:{intent}",
            )

        # 2. Low confidence → ask for clarification
        if confidence < self.confidence_threshold:
            pool = _LOW_CONFIDENCE_AR if arabic else _LOW_CONFIDENCE_EN
            return RouteDecision(
                handled=True,
                reply=random.choice(pool),
                reason="low_confidence",
            )

        # 3. Unknown intent → polite bilingual fallback
        if intent == "unknown":
            return RouteDecision(
                handled=True,
                reply=_UNKNOWN_AR if arabic else _UNKNOWN_EN,
                reason="unknown_intent",
            )

        # 4. Everything else → RAG pipeline
        return RouteDecision(handled=False, reason=f"rag:{intent}")

