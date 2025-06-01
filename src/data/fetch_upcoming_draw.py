#!/usr/bin/env python
"""Fetch upcoming WTA tournament draw and output matchups.

This script downloads the draw for the next scheduled WTA event from the
official WTA website API. It maps player names to IDs stored in the local
``tennis.db`` database and writes a CSV compatible with ``HeadToHeadPredictor``
with columns ``player1_id,player2_id,surface``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# Setup logging similar to other scripts
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("fetch_upcoming_draw")

# Path constants
DATA_DIR = Path("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DATA_DIR / "tennis.db"
DEFAULT_OUT = PROCESSED_DATA_DIR / "upcoming_draw.csv"


def _http_json(url: str) -> dict:
    """Return JSON from a URL using ``urllib``."""
    logger.info("Fetching %s", url)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status} for {url}")
            return json.load(resp)
    except HTTPError as exc:
        logger.error("HTTP error %s for %s", exc.code, url)
        raise


def _get_next_event() -> tuple[str, str]:
    """Return the id and surface for the next WTA event.

    The WTA website exposes a calendar API which we query for the first
    upcoming tournament. Only minimal fields are used so the function remains
    resilient to minor schema changes.
    """
    url = (
        "https://www.wtatennis.com/proxy/api/v1/events?page=0&pageSize=1"
        "&state=upcoming&sort=startDate&order=asc"
    )
    data = _http_json(url)
    events = data.get("events") or data.get("tournaments")
    if not events:
        raise ValueError("No upcoming events found")
    event = events[0]
    event_id = str(event.get("uuid") or event.get("id"))
    surface = event.get("surface", "Hard")
    logger.info("Next event %s on %s", event_id, surface)
    return event_id, surface


def _fetch_draw(event_id: str) -> list[tuple[str, str]]:
    """Return list of player name pairs for the tournament draw."""
    draw_url = f"https://www.wtatennis.com/proxy/api/v1/events/{event_id}/matches"
    data = _http_json(draw_url)
    pairs: list[tuple[str, str]] = []
    for match in data.get("matches", []):
        p1 = match.get("player1", {}).get("name")
        p2 = match.get("player2", {}).get("name")
        if p1 and p2:
            pairs.append((p1, p2))
    logger.info("Fetched %d matchups", len(pairs))
    return pairs


def _parse_name(full_name: str) -> tuple[str, str]:
    parts = full_name.strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _map_players(names: Iterable[str], conn: sqlite3.Connection) -> dict[str, str | None]:
    """Map names to ``player_id`` using the ``players`` table."""
    mapped: dict[str, str | None] = {}
    cur = conn.cursor()
    for name in names:
        first, last = _parse_name(name)
        row = cur.execute(
            "SELECT player_id FROM players WHERE tour='WTA' AND lower(first_name)=lower(?) AND lower(last_name)=lower(?)",
            (first, last),
        ).fetchone()
        mapped[name] = row[0] if row else None
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch upcoming WTA draw")
    parser.add_argument("--db-path", type=Path, default=DB_PATH, help="Path to tennis.db")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    event_id, surface = _get_next_event()
    pairs = _fetch_draw(event_id)

    conn = sqlite3.connect(args.db_path)
    name_map = _map_players({n for pair in pairs for n in pair}, conn)
    conn.close()

    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player1_id", "player2_id", "surface"])
        for p1, p2 in pairs:
            id1 = name_map.get(p1)
            id2 = name_map.get(p2)
            if id1 and id2:
                writer.writerow([id1, id2, surface])
            else:
                logger.warning("Skipping matchup %s vs %s due to missing IDs", p1, p2)

    logger.info("Saved draw to %s", args.output)


if __name__ == "__main__":
    main()
