from fastapi import HTTPException

MAX_MESSAGE_LENGTH = 1000


def validate_message(message: str) -> str:
    """
    Validates a user message.
    Returns the cleaned message or raises HTTPException.
    """
    if not message or not message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Message too long. Max {MAX_MESSAGE_LENGTH} characters."
        )

    return message.strip()