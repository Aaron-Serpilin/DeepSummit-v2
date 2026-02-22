import numpy as np
import pytest
from sqlalchemy import text

import db


def test_load_expedition_returns_dict(db_session, test_expedition_id):
    result = db.load_expedition(db_session, test_expedition_id)

    assert result is not None
    assert result["id"] == test_expedition_id
    assert result["year"] == 9999
    assert result["season"] == "Spring"
    assert len(result["members"]) == 1
    assert result["members"][0]["age"] == 30


def test_load_expedition_returns_none_for_missing_id(db_session):
    result = db.load_expedition(db_session, -999)

    assert result is None


def test_upsert_embedding_inserts_row(db_session, test_expedition_id):
    vector = np.random.default_rng(42).standard_normal(512).astype(np.float32)

    db.upsert_embedding(db_session, test_expedition_id, vector, "test-v0")
    db_session.flush()

    row = db_session.execute(
        text("SELECT model_version FROM expedition_embeddings WHERE expedition_id = :id"),
        {"id": test_expedition_id},
    ).first()
    assert row.model_version == "test-v0"


def test_upsert_embedding_updates_on_conflict(db_session, test_expedition_id):
    vector = np.random.default_rng(1).standard_normal(512).astype(np.float32)

    db.upsert_embedding(db_session, test_expedition_id, vector, "test-v0")
    db_session.flush()
    db.upsert_embedding(db_session, test_expedition_id, vector, "test-v1")
    db_session.flush()

    row = db_session.execute(
        text("SELECT model_version FROM expedition_embeddings WHERE expedition_id = :id"),
        {"id": test_expedition_id},
    ).first()
    assert row.model_version == "test-v1"
