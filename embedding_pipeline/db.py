import os
from functools import lru_cache

import numpy as np
import structlog
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine as _create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

log = structlog.get_logger()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create and cache a SQLAlchemy engine. Reads DATABASE_URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if url is None:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    engine = _create_engine(url, pool_pre_ping=True)

    # Register the pgvector psycopg2 adapter on every new connection so psycopg2
    # knows how to serialise/deserialise the `vector` PostgreSQL type.
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_connection, connection_record):
        register_vector(dbapi_connection)

    return engine


def load_expedition(session: Session, expedition_id: int) -> dict | None:
    """Load expedition with its members from PostgreSQL.

    Returns a dict ready for the embedder, or None if the expedition doesn't exist.
    """
    row = session.execute(
        text("""
            SELECT id, peak_id, year, season, team_size, commercial, style
            FROM expeditions
            WHERE id = :id
        """),
        {"id": expedition_id},
    ).mappings().first()

    if row is None:
        return None

    expedition = dict(row)

    members = session.execute(
        text("""
            SELECT age, gender, nationality, oxygen_used, summit_reached,
                   death_on_mountain, prior_8000m_summits, highest_prev_altitude_m
            FROM expedition_members
            WHERE expedition_id = :id
        """),
        {"id": expedition_id},
    ).mappings().all()

    expedition["members"] = [dict(m) for m in members]
    return expedition


def upsert_embedding(
    session: Session,
    expedition_id: int,
    vector: np.ndarray,
    model_version: str,
) -> None:
    """UPSERT a 512-dim embedding into expedition_embeddings.

    Does NOT commit — the caller (pipeline.process) is responsible for committing.
    This keeps the batch atomic: all embeddings commit together or not at all.
    """
    # pgvector accepts the Python list representation as a vector literal —
    # str([0.1, 0.2, ...]) produces "[0.1, 0.2, ...]" which is exactly the
    # format PostgreSQL expects. The register_vector() call in get_engine()
    # teaches psycopg2 to handle the vector type on return, so reads work too.
    embedding_str = str(vector.tolist())

    session.execute(
        text("""
            INSERT INTO expedition_embeddings (expedition_id, embedding, model_version)
            VALUES (:expedition_id, CAST(:embedding AS vector), :model_version)
            ON CONFLICT (expedition_id) DO UPDATE
            SET embedding      = EXCLUDED.embedding,
                model_version  = EXCLUDED.model_version,
                created_at     = now()
        """),
        {
            "expedition_id": expedition_id,
            "embedding": embedding_str,
            "model_version": model_version,
        },
    )
