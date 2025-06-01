import sqlite3
from pathlib import Path
import difflib
import logging

# Default path to the SQLite database
DB_PATH = Path("data") / "processed" / "tennis.db"

logger = logging.getLogger(__name__)


def lookup_player_id(
    first_name: str,
    last_name: str,
    conn: sqlite3.Connection | None = None,
    manual_map: dict | None = None,
    fuzzy_cutoff: float = 0.8,
) -> str | None:
    """Lookup a player_id by first and last name.

    The search is case-insensitive and will attempt fuzzy matching if an exact
    match is not found. A manual mapping dictionary can be supplied to handle
    known name variations.

    Parameters
    ----------
    first_name : str
        Player's first name.
    last_name : str
        Player's last name.
    conn : sqlite3.Connection, optional
        Database connection. If not provided, a new connection is created using
        ``DB_PATH`` and closed on exit.
    manual_map : dict, optional
        Mapping of ``(first_name.lower(), last_name.lower())`` tuples to
        ``player_id`` values for manual overrides.
    fuzzy_cutoff : float, optional
        Similarity threshold for fuzzy matching (0-1 range).

    Returns
    -------
    str | None
        The corresponding ``player_id`` or ``None`` if no reasonable match is
        found.
    """
    manual_map = manual_map or {}
    norm_first = first_name.strip().lower()
    norm_last = last_name.strip().lower()

    # Manual override
    if (norm_first, norm_last) in manual_map:
        return manual_map[(norm_first, norm_last)]

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        close_conn = True

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT player_id FROM players
            WHERE lower(first_name) = ? AND lower(last_name) = ?
            """,
            (norm_first, norm_last),
        )
        rows = cursor.fetchall()
        if rows:
            # If multiple matches, return the first one deterministically
            return rows[0][0]

        # Fuzzy match across all players
        cursor.execute("SELECT player_id, first_name, last_name FROM players")
        players = cursor.fetchall()
        name_to_id = {f"{fn.lower()} {ln.lower()}": pid for pid, fn, ln in players}
        query = f"{norm_first} {norm_last}"
        matches = difflib.get_close_matches(
            query, name_to_id.keys(), n=1, cutoff=fuzzy_cutoff
        )
        if matches:
            return name_to_id[matches[0]]
    except Exception as exc:
        logger.error("Error looking up player %s %s: %s", first_name, last_name, exc)
    finally:
        if close_conn:
            conn.close()

    return None
