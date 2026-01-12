import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict

DB_NAME = "interactions.db"


def _db_path() -> str:
    return os.path.join(os.path.dirname(__file__), DB_NAME)


def init_db() -> None:
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY,
            username TEXT,
            question TEXT,
            answer TEXT,
            source TEXT,
            corrected_answer TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def log_interaction(username: str, question: str, answer: str, source: Optional[List[Dict]] = None) -> int:
    """Log a user interaction and return the inserted row id."""
    init_db()
    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()
    src_json = json.dumps(source or [])
    cur.execute(
        "INSERT INTO interactions (username, question, answer, source, created_at) VALUES (?, ?, ?, ?, ?)",
        (username, question, answer, src_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    rowid = cur.lastrowid
    conn.close()
    return rowid


def add_correction(interaction_id: int, corrected_answer: str) -> bool:
    """Attach a corrected answer to an existing interaction."""
    init_db()
    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()
    cur.execute(
        "UPDATE interactions SET corrected_answer = ? WHERE id = ?",
        (corrected_answer, interaction_id),
    )
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok


def get_recent(limit: int = 50) -> List[Dict]:
    init_db()
    conn = sqlite3.connect(_db_path())
    cur = conn.cursor()
    cur.execute("SELECT id, username, question, answer, source, corrected_answer, created_at FROM interactions ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    out = []
    for r in rows:
        src = []
        try:
            src = json.loads(r[4]) if r[4] else []
        except Exception:
            src = []
        out.append({
            "id": r[0],
            "username": r[1],
            "question": r[2],
            "answer": r[3],
            "source": src,
            "corrected_answer": r[5],
            "created_at": r[6],
        })
    return out


if __name__ == "__main__":
    init_db()
    print("Interaction DB initialized at", _db_path())
