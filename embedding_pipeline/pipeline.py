import structlog
from sqlalchemy.orm import Session

import db
import embedder

log = structlog.get_logger()

def process(session: Session, expedition_ids: list[int]) -> None:
    """
    For each expedition ID: load → embed → upsert. Commits once after the batch.

    Skips IDs that no longer exist in the DB (logs a warning).
    Raises on DB write failure so Pub/Sub retries the batch.
    """
    for expedition_id in expedition_ids:
        expedition = db.load_expedition(session, expedition_id)

        if expedition is None:
            log.warning("expedition_not_found", expedition_id=expedition_id)
            continue

        vector = embedder.compute(expedition)
        db.upsert_embedding(session, expedition_id, vector, embedder.MODEL_VERSION)
        log.info("embedding_computed", expedition_id=expedition_id, model=embedder.MODEL_VERSION)

    session.commit()
