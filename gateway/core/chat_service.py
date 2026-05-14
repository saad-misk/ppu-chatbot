"""
Chat service — used by the Telegram channel and other non-HTTP callers.
Uses get_db_ctx() (context-manager version) instead of get_db() generator.
"""
import logging

from shared.config.settings import settings

logger = logging.getLogger(__name__)


async def process_chat_message(
    session_id: str,
    message: str,
    channel: str = "telegram",
    user_id: str | None = None,
):
    """
    Shared service used by both Web and Telegram Bot.
    `user_id` is optional — pass None for anonymous/guest sessions.
    """
    from gateway.api.routes.chat import call_nlp_engine
    from shared.schemas.message import ProcessRequest
    from gateway.storage.db import get_db_ctx
    from gateway.storage import chat_repo

    try:
        with get_db_ctx() as db:
            session = chat_repo.get_session(db, session_id)
            if not session:
                session = chat_repo.create_session(db, channel=channel, user_id=user_id)

            history_turns = chat_repo.get_history(db, session_id)
            history = [{"role": t.role, "content": t.content} for t in history_turns]

            chat_repo.save_turn(db, session_id=session_id, role="user", content=message)

            nlp_request = ProcessRequest(
                session_id=session_id,
                message=message,
                history=history,
                channel=channel,
            )
            nlp_response = await call_nlp_engine(nlp_request)

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
                "low_confidence": nlp_response.confidence < settings.CONFIDENCE_THRESHOLD,
            }

    except Exception as exc:
        logger.error("Chat service error: %s", exc)
        return {
            "reply": "⚠️ عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
        }