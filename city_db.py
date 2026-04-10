import sqlite3
from pathlib import Path
from typing import List

import geonamescache

DB_DIR = Path("data")
DB_FILE = DB_DIR / "indian_cities.db"


def _connect() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_FILE)


def init_city_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                population INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cities_normalized_name ON cities(normalized_name)"
        )

        count = conn.execute("SELECT COUNT(*) FROM cities").fetchone()[0]
        if count > 0:
            return

        gc = geonamescache.GeonamesCache()
        cities = gc.get_cities()
        rows = []
        seen = set()

        for city in cities.values():
            if city.get("countrycode") != "IN":
                continue
            name = (city.get("name") or "").strip()
            if not name:
                continue

            state_code = city.get("admin1code") or ""
            state = f"State-{state_code}" if state_code else "Unknown"
            normalized_name = name.lower()
            population = int(city.get("population") or 0)

            key = (normalized_name, state.lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append((name, state, normalized_name, population))

        rows.sort(key=lambda row: (-row[3], row[0]))
        conn.executemany(
            """
            INSERT INTO cities(name, state, normalized_name, population)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )


def search_indian_cities(query: str, limit: int = 8) -> List[str]:
    q = (query or "").strip().lower()
    if len(q) < 1:
        return []

    with _connect() as conn:
        cursor = conn.execute(
            """
            SELECT name
            FROM cities
            WHERE normalized_name LIKE ?
            GROUP BY normalized_name
            ORDER BY MAX(population) DESC, name ASC
            LIMIT ?
            """,
            (f"{q}%", limit),
        )
        return [name for (name,) in cursor.fetchall()]
