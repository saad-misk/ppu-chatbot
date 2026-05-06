"""End-to-end smoke test for the NLP engine.

Run after starting the NLP engine:
  uvicorn nlp_engine.nlp_server:app --port 8001 --reload
"""
from __future__ import annotations

import argparse
import sys
import uuid

import httpx


def _require_keys(payload: dict, keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise RuntimeError(f"Missing keys in {context}: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="NLP engine end-to-end smoke test")
    parser.add_argument("--nlp-url", default="http://localhost:8001")
    parser.add_argument("--message", default="What are the CS tuition fees?")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())

    payload = {
        "session_id": session_id,
        "message": args.message,
        "history": [],
        "channel": "web",
    }

    try:
        with httpx.Client(timeout=args.timeout) as client:
            health = client.get(f"{args.nlp_url}/health")
            health.raise_for_status()
            print("health:", health.json())

            resp = client.post(f"{args.nlp_url}/process", json=payload)
            resp.raise_for_status()
            data = resp.json()

            _require_keys(data, ["reply", "intent", "confidence", "sources"], "process response")

            print("intent:", data["intent"])
            print("confidence:", data["confidence"])
            print("reply:", data["reply"][:400])
        return 0
    except Exception as exc:
        print(f"E2E smoke test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
