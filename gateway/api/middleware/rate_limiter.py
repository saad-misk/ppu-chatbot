import time
from collections import defaultdict
from fastapi import Request, HTTPException

# Store: {ip: [timestamp, timestamp, ...]}
_request_log: dict[str, list[float]] = defaultdict(list)

MAX_REQUESTS = 30   # per window
WINDOW_SECONDS = 60


def check_rate_limit(request: Request):
    ip = request.client.host
    now = time.time()

    # Keep only timestamps within the current window
    _request_log[ip] = [t for t in _request_log[ip] if now - t < WINDOW_SECONDS]

    if len(_request_log[ip]) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {MAX_REQUESTS} requests per {WINDOW_SECONDS}s."
        )

    _request_log[ip].append(now)