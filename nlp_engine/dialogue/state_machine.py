"""
Dialogue State Machine — deterministic response routing with conversation tracking.

For well-defined intents (greetings, thanks, goodbyes) the bot returns
canned replies without calling the LLM. For all other intents it signals
that the RAG pipeline should handle the request.

Features:
  • Dual-threshold routing (lower for social phrases)
  • Escalation on repeated low-confidence queries
  • Conversation state tracking (contextual responses)
  • Fallback intent handling
  • Entity-aware responses when possible

Usage
-----
    from nlp_engine.dialogue.state_machine import StateMachine, RouteDecision

    sm = StateMachine()
    decision = sm.route(
        intent="greeting", 
        confidence=0.95, 
        entities={},
        query="مرحبا",
        session_id="user_123"
    )
    if decision.handled:
        return decision.reply
    else:
        pass  # fall through to RAG
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from shared.utils.lang import is_arabic as _is_arabic_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Route decision
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    """Result of state machine routing decision."""
    handled: bool                    # True → use reply directly; False → call RAG
    reply: Optional[str] = None      # populated when handled=True
    reason: str = ""                 # debug label
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional context


# ---------------------------------------------------------------------------
# Conversation state tracking
# ---------------------------------------------------------------------------

class ConversationState:
    """Tracks the state of a single conversation session."""
    
    def __init__(self):
        self.last_intent: Optional[str] = None
        self.last_confidence: float = 0.0
        self.consecutive_low_confidence: int = 0
        self.consecutive_fallbacks: int = 0
        self.greeted: bool = False
        self.helped_with: List[str] = []  # Track what we've helped with
        self.turn_count: int = 0
        self.last_activity: datetime = datetime.now()
    
    def update(self, intent: str, confidence: float, handled: bool):
        """Update state after processing a turn."""
        self.last_intent = intent
        self.last_confidence = confidence
        self.last_activity = datetime.now()
        self.turn_count += 1
        
        if intent in ("fallback", "unknown") or confidence < 0.4:
            self.consecutive_low_confidence += 1
        else:
            self.consecutive_low_confidence = 0
        
        if intent == "fallback":
            self.consecutive_fallbacks += 1
        else:
            self.consecutive_fallbacks = 0
        
        if intent == "greeting":
            self.greeted = True
        
        if handled:
            self.helped_with.append(intent)


# ---------------------------------------------------------------------------
# Canned response pools
# ---------------------------------------------------------------------------

# Greetings
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

# Return greetings (user comes back)
_RETURN_GREETINGS_AR: List[str] = [
    "مرحباً مجدداً! كيف يمكنني مساعدتك هذه المرة؟",
    "أهلاً بعودتك! هل لديك سؤال جديد؟",
]

_RETURN_GREETINGS_EN: List[str] = [
    "Welcome back! How can I help you this time?",
    "Good to see you again! What would you like to know?",
]

# Thanks
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

# Goodbyes
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

# After-help goodbyes (contextual)
_AFTER_HELP_GOODBYES_AR: List[str] = [
    "بالتوفيق! وإذا احتجت أي مساعدة أخرى، أنا هنا دائماً.",
    "وداعاً! أتمنى أن تكون المعلومات مفيدة لك.",
]

_AFTER_HELP_GOODBYES_EN: List[str] = [
    "Good luck! If you need any more help, I'm always here.",
    "Goodbye! I hope the information was helpful.",
]

# Low confidence / clarification
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

# Escalation after repeated failures
_ESCALATION_AR: List[str] = [
    "يبدو أنني أواجه صعوبة في فهم سؤالك. يمكنك التواصل مع مكتب القبول والتسجيل مباشرة للحصول على مساعدة أفضل.",
    "للحصول على مساعدة أكثر دقة، يمكنك زيارة مكتب شؤون الطلاب أو إرسال بريد إلكتروني إلى القسم المختص.",
    "أعتذر عن عدم تمكني من مساعدتك. يمكنك الاتصال على الرقم 02-2233050 للتحدث مع موظف مختص.",
]

_ESCALATION_EN: List[str] = [
    "I seem to be having trouble understanding your question. You can contact the Admissions Office directly for better assistance.",
    "For more accurate help, you can visit the Student Affairs office or email the relevant department.",
    "I apologize for not being able to help. You can call 02-2233050 to speak with a staff member.",
]

# Fallback intent prompts
_FALLBACK_AR: List[str] = [
    "يمكنني مساعدتك في مواضيع مثل: الرسوم الدراسية، التسجيل، الجداول، الكادر الأكاديمي، والأقسام. ماذا تريد أن تعرف؟",
    "لم أتمكن من تحديد سؤالك. يمكنك السؤال عن: القبول، الرسوم، المساقات، أو التخصصات.",
]

_FALLBACK_EN: List[str] = [
    "I can help with topics like: tuition fees, registration, schedules, faculty, and departments. What would you like to know?",
    "I couldn't identify your question. You can ask about: admissions, fees, courses, or majors.",
]

# Unknown intent
_UNKNOWN_AR = (
    "لست متأكداً كيف يمكنني مساعدتك في هذا. "
    "يمكنني الإجابة على أسئلة حول الرسوم، التسجيل، الجداول الدراسية، الكادر الأكاديمي، أو الأقسام."
)
_UNKNOWN_EN = (
    "I'm not sure how to help with that. "
    "You can ask me about fees, registration, schedules, staff, or departments."
)

# Capability introduction
_CAPABILITIES_AR = (
    "يمكنني مساعدتك في:\n"
    "• الرسوم الدراسية والتكاليف\n"
    "• التسجيل والمواعيد\n"
    "• الجداول والمساقات\n"
    "• الكادر الأكاديمي والأساتذة\n"
    "• الأقسام والتخصصات\n"
    "ما الذي تريد معرفته؟"
)
_CAPABILITIES_EN = (
    "I can help you with:\n"
    "• Tuition fees and costs\n"
    "• Registration and deadlines\n"
    "• Schedules and courses\n"
    "• Faculty and professors\n"
    "• Departments and majors\n"
    "What would you like to know?"
)


# ---------------------------------------------------------------------------
# Deterministic intent routing table
# ---------------------------------------------------------------------------

_CANNED_INTENTS: Dict[str, Tuple[List[str], List[str]]] = {
    "greeting": (_GREETINGS_AR, _GREETINGS_EN),
    "goodbye":  (_GOODBYES_AR,  _GOODBYES_EN),
    "thanks":   (_THANKS_AR,    _THANKS_EN),
    "capabilities": (_CAPABILITIES_AR.split("\n"), _CAPABILITIES_EN.split("\n")),
}

# Intents that can trigger a capabilities response
_CAPABILITY_TRIGGERS = {"what_can_you_do", "help", "capabilities", "features"}


class StateMachine:
    """
    Routes an incoming classified intent to either a canned reply or the RAG pipeline.
    
    Features:
    - Language-aware responses
    - Dual-threshold for social vs factual intents
    - Escalation on repeated failures
    - Contextual responses (return greetings, after-help goodbyes)
    - Session tracking per user
    
    Args:
        confidence_threshold: Minimum confidence for factual intents
        max_low_confidence_turns: Escalate after this many unclear queries
        session_timeout_minutes: Clear state after this many minutes
    """

    _CANNED_CONFIDENCE_THRESHOLD = 0.35
    
    def __init__(
        self,
        confidence_threshold: float = 0.55,
        max_low_confidence_turns: int = 3,
        session_timeout_minutes: int = 30,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_low_confidence_turns = max_low_confidence_turns
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        
        # Per-session state tracking
        self._sessions: Dict[str, ConversationState] = defaultdict(ConversationState)

    def _get_session(self, session_id: str = "default") -> ConversationState:
        """Get or create a conversation session, cleaning up expired ones."""
        session = self._sessions[session_id]
        
        # Check if session has expired
        if datetime.now() - session.last_activity > self.session_timeout:
            logger.debug(f"Session {session_id} expired, creating new state")
            self._sessions[session_id] = ConversationState()
            session = self._sessions[session_id]
        
        return session

    def route(
        self,
        intent: str,
        confidence: float,
        entities: Dict,
        query: str = "",
        session_id: str = "default",
    ) -> RouteDecision:
        """
        Determine how to handle this turn.

        Args:
            intent: Classified intent label
            confidence: Classifier confidence score
            entities: Extracted entity dict
            query: Original user message (for language detection)
            session_id: Unique session identifier

        Returns:
            RouteDecision with handling instructions
        """
        arabic = _is_arabic_text(query)
        session = self._get_session(session_id)
        
        # --- 1. Escalation check (repeated failures) ---
        if session.consecutive_low_confidence >= self.max_low_confidence_turns:
            pool = _ESCALATION_AR if arabic else _ESCALATION_EN
            session.consecutive_low_confidence = 0  # Reset after escalation
            session.consecutive_fallbacks = 0
            return RouteDecision(
                handled=True,
                reply=random.choice(pool),
                reason="escalation",
                metadata={"session_ended": True},
            )
        
        # --- 2. Canned intents (greeting / thanks / goodbye) ---
        if intent in _CANNED_INTENTS and confidence >= self._CANNED_CONFIDENCE_THRESHOLD:
            ar_pool, en_pool = _CANNED_INTENTS[intent]
            
            # Contextual response selection
            if intent == "greeting" and session.greeted and session.turn_count > 2:
                # Returning user - use different greeting
                pool = _RETURN_GREETINGS_AR if arabic else _RETURN_GREETINGS_EN
            elif intent == "goodbye" and len(session.helped_with) > 0:
                # User was actually helped - use contextual goodbye
                pool = _AFTER_HELP_GOODBYES_AR if arabic else _AFTER_HELP_GOODBYES_EN
            else:
                pool = ar_pool if arabic else en_pool
            
            session.update(intent, confidence, handled=True)
            
            # If it's a goodbye, mark session for cleanup
            metadata = {"session_ending": True} if intent == "goodbye" else {}
            
            return RouteDecision(
                handled=True,
                reply=random.choice(pool),
                reason=f"canned:{intent}",
                metadata=metadata,
            )
        
        # --- 3. Capabilities request ---
        if intent in _CAPABILITY_TRIGGERS:
            reply = _CAPABILITIES_AR if arabic else _CAPABILITIES_EN
            session.update(intent, confidence, handled=True)
            return RouteDecision(
                handled=True,
                reply=reply,
                reason="capabilities",
            )
        
        # --- 4. Low confidence → ask for clarification ---
        if confidence < self.confidence_threshold:
            session.update(intent, confidence, handled=True)
            
            # Escalate language after multiple failures
            if session.consecutive_low_confidence >= 2:
                pool = _FALLBACK_AR if arabic else _FALLBACK_EN
            else:
                pool = _LOW_CONFIDENCE_AR if arabic else _LOW_CONFIDENCE_EN
            
            return RouteDecision(
                handled=True,
                reply=random.choice(pool),
                reason=f"low_confidence:{session.consecutive_low_confidence}",
            )
        
        # --- 5. Fallback intent (classifier uncertain) ---
        if intent == "fallback":
            session.update(intent, confidence, handled=True)
            
            if session.consecutive_fallbacks >= 2:
                pool = _ESCALATION_AR if arabic else _ESCALATION_EN
                session.consecutive_fallbacks = 0
                return RouteDecision(
                    handled=True,
                    reply=random.choice(pool),
                    reason="fallback_escalation",
                    metadata={"session_ended": True},
                )
            
            pool = _FALLBACK_AR if arabic else _FALLBACK_EN
            return RouteDecision(
                handled=True,
                reply=random.choice(pool),
                reason="fallback",
            )
        
        # --- 6. Unknown intent → polite fallback with capability hints ---
        if intent == "unknown":
            session.update(intent, confidence, handled=True)
            return RouteDecision(
                handled=True,
                reply=_UNKNOWN_AR if arabic else _UNKNOWN_EN,
                reason="unknown_intent",
            )
        
        # --- 7. Everything else → RAG pipeline ---
        session.update(intent, confidence, handled=False)
        return RouteDecision(
            handled=False, 
            reason=f"rag:{intent}",
            metadata={
                "session_turns": session.turn_count,
                "previously_helped_with": session.helped_with[-3:],
            }
        )

    def get_session_state(self, session_id: str = "default") -> Dict:
        """Get current conversation state for debugging/monitoring."""
        session = self._get_session(session_id)
        return {
            "last_intent": session.last_intent,
            "last_confidence": session.last_confidence,
            "consecutive_low_confidence": session.consecutive_low_confidence,
            "consecutive_fallbacks": session.consecutive_fallbacks,
            "greeted": session.greeted,
            "helped_with": session.helped_with,
            "turn_count": session.turn_count,
            "last_activity": session.last_activity.isoformat(),
        }

    def reset_session(self, session_id: str = "default"):
        """Reset a conversation session."""
        self._sessions[session_id] = ConversationState()
        logger.info(f"Session {session_id} reset")

    def clear_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now()
        expired = [
            sid for sid, session in self._sessions.items()
            if now - session.last_activity > self.session_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Cleared {len(expired)} expired sessions")