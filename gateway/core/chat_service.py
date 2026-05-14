import logging
from shared.config.settings import settings

logger = logging.getLogger(__name__)


async def process_chat_message(session_id: str, message: str, channel: str = "telegram"):
    """
    Clean service used by both Web and Telegram Bot
    """
    try:
        # Import inside function to avoid circular imports
        from gateway.api.routes.chat import call_nlp_engine
        from shared.schemas.message import ProcessRequest
        from gateway.storage.db import get_db
        from gateway.storage import chat_repo

        db = next(get_db())

        # Get or create session
        session = chat_repo.get_session(db, session_id)
        if not session:
            session = chat_repo.create_session(db, channel=channel)

        # Get history
        history_turns = chat_repo.get_history(db, session_id)
        history = [{"role": t.role, "content": t.content} for t in history_turns]

        # Save user message
        chat_repo.save_turn(db, session_id=session_id, role="user", content=message)

        # Call NLP Engine
        nlp_request = ProcessRequest(
            session_id=session_id,
            message=message,
            history=history,
            channel=channel,
        )

        nlp_response = await call_nlp_engine(nlp_request)

        # Save assistant response
        chat_repo.save_turn(
            db,
            session_id=session_id,
            role="assistant",
            content=nlp_response.reply,
            intent=nlp_response.intent,
            confidence=nlp_response.confidence,
        )

        return {
            "reply": nlp_response.reply,
            "low_confidence": nlp_response.confidence < settings.CONFIDENCE_THRESHOLD
        }

    except Exception as e:
        logger.error(f"Chat service error: {e}")
        return {
            "reply": "⚠️ Sorry, I'm having trouble processing your request right now. Please try again."
        }
    finally:
        if 'db' in locals():
            db.close()