import base64
import json
from typing import Generator

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

import db
import pipeline

app = FastAPI()
log = structlog.get_logger()


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a DB session per request.
    """
    with Session(db.get_engine()) as session:
        yield session


@app.post("/", status_code=204)
async def handle_pubsub(
    request: Request,
    session: Session = Depends(get_session),
) -> None:
    """
    Receive a Pub/Sub push message and process the embedded expedition IDs.

    Returns 204 to acknowledge the message. Returning non-2xx causes Pub/Sub
    to retry, so only raise on unrecoverable errors (bad message format).
    DB write failures propagate as 500s — Pub/Sub will retry the batch.
    """
    body = await request.json()

    try:
        raw_data = body["message"]["data"]
        payload = json.loads(base64.b64decode(raw_data).decode("utf-8"))
        expedition_ids: list[int] = payload["expedition_ids"]
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log.error("invalid_pubsub_message", error=str(exc))
        raise HTTPException(status_code=400, detail="Invalid Pub/Sub message format")

    if not expedition_ids:
        return  # valid no-op, already returns 204

    log.info("processing_batch", expedition_count=len(expedition_ids))
    pipeline.process(session, expedition_ids)
    log.info("batch_complete", expedition_count=len(expedition_ids))
