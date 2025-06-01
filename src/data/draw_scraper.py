"""Simple utilities for scraping and processing tournament draws."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
from typing import Iterable, Tuple, List

from .player_utils import lookup_player_id, DB_PATH

logger = logging.getLogger(__name__)


def parse_draw_file(path: Path) -> List[Tuple[str, str]]:
    """Placeholder parser that reads a simple CSV with ``player1,player2`` rows."""
    matchups = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                matchups.append((parts[0], parts[1]))
    return matchups


def convert_matchups_to_ids(
    matchups: Iterable[Tuple[str, str]],
    conn: sqlite3.Connection | None = None,
    manual_map: dict | None = None,
) -> List[Tuple[str | None, str | None]]:
    """Convert a list of matchup player name tuples to player id tuples."""
    manual_map = manual_map or {}
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    results = []
    for p1_name, p2_name in matchups:
        p1_first, p1_last = p1_name.split(" ", 1)
        p2_first, p2_last = p2_name.split(" ", 1)
        p1_id = lookup_player_id(p1_first, p1_last, conn=conn, manual_map=manual_map)
        p2_id = lookup_player_id(p2_first, p2_last, conn=conn, manual_map=manual_map)
        results.append((p1_id, p2_id))

    if close_conn:
        conn.close()

    return results
