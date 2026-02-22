import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get("DATABASE_URL")

@pytest.fixture(scope="session")
def engine():
    return create_engine(DATABASE_URL)


@pytest.fixture
def db_session(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def test_expedition_id(db_session):
    """Insert one synthetic expedition + one member. Cleans up after the test.

    Uses year=9999 as a sentinel so stale rows are identifiable if cleanup fails.
    """
    result = db_session.execute(
        text("""
            INSERT INTO expeditions (peak_id, year, season, team_size, commercial, style)
            VALUES (1, 9999, 'Spring', 2, false, 'alpine')
            RETURNING id
        """)
    )
    expedition_id = result.scalar()

    db_session.execute(
        text("""
            INSERT INTO expedition_members
                (expedition_id, age, gender, nationality, oxygen_used,
                 summit_reached, prior_8000m_summits, highest_prev_altitude_m)
            VALUES (:eid, 30, 'M', 'USA', false, true, 1, 7000)
        """),
        {"eid": expedition_id},
    )
    # Both rows committed atomically — if either INSERT fails, nothing lands in the DB.
    db_session.commit()

    yield expedition_id

    # Teardown — rollback any partial test state before cleanup so the session
    # is in a clean state even if the test left it mid-transaction or errored.
    try:
        db_session.rollback()
        db_session.execute(
            text("DELETE FROM expedition_members WHERE expedition_id = :id"),
            {"id": expedition_id},
        )
        db_session.execute(
            text("DELETE FROM expeditions WHERE id = :id"),
            {"id": expedition_id},
        )
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise
