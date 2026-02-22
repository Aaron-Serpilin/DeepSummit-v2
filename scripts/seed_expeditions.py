#!/usr/bin/env python3
"""Seed 5 synthetic Everest expeditions for local development.

Run once after `docker compose up -d`:
    python scripts/seed_expeditions.py

Safe to re-run — skips if any expeditions already exist.
"""

import os
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://deepsummit:deepsummit_password@localhost:5432/deepsummit_psql_db_dev",
)

# 5 synthetic Everest (peak_id=1) expeditions
EXPEDITIONS = [
    {"peak_id": 1, "year": 2019, "season": "Spring",  "team_size": 8,  "commercial": True,  "style": "expedition"},
    {"peak_id": 1, "year": 2020, "season": "Spring",  "team_size": 3,  "commercial": False, "style": "alpine"},
    {"peak_id": 1, "year": 2021, "season": "Autumn",  "team_size": 6,  "commercial": True,  "style": "expedition"},
    {"peak_id": 1, "year": 2022, "season": "Spring",  "team_size": 2,  "commercial": False, "style": "alpine"},
    {"peak_id": 1, "year": 2023, "season": "Spring",  "team_size": 10, "commercial": True,  "style": "expedition"},
]

# (age, gender, nationality, oxygen_used, summit_reached, prior_8000m_summits, highest_prev_altitude_m)
MEMBERS = [
    [(32, "M", "USA", True,  True,  1, 7500), (29, "F", "GBR", True,  True,  0, 6500)],
    [(45, "M", "NZL", False, False, 3, 8200), (38, "M", "CHE", False, False, 2, 7900), (41, "F", "FRA", False, True, 1, 7500)],
    [(35, "M", "DEU", True,  True,  0, 0)],
    [(50, "M", "AUS", False, False, 5, 8600), (44, "F", "JPN", True,  True,  2, 8000)],
    [(28, "M", "NPL", False, True,  7, 8849)],
]


def seed() -> None:
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        existing = session.execute(text("SELECT COUNT(*) FROM expeditions")).scalar()
        if existing and existing > 0:
            print(f"Database already has {existing} expedition(s) — skipping seed.")
            return

        for i, exp_data in enumerate(EXPEDITIONS):
            result = session.execute(
                text("""
                    INSERT INTO expeditions (peak_id, year, season, team_size, commercial, style)
                    VALUES (:peak_id, :year, :season, :team_size, :commercial, :style)
                    RETURNING id
                """),
                exp_data,
            )
            expedition_id = result.scalar()

            for member in MEMBERS[i]:
                session.execute(
                    text("""
                        INSERT INTO expedition_members
                            (expedition_id, age, gender, nationality, oxygen_used,
                             summit_reached, prior_8000m_summits, highest_prev_altitude_m)
                        VALUES (:eid, :age, :gender, :nat, :oxy, :summit, :prior, :highest)
                    """),
                    {
                        "eid": expedition_id,
                        "age": member[0], "gender": member[1], "nat": member[2],
                        "oxy": member[3], "summit": member[4],
                        "prior": member[5], "highest": member[6],
                    },
                )

        session.commit()
        print(f"Seeded {len(EXPEDITIONS)} expeditions with members.")


if __name__ == "__main__":
    seed()
