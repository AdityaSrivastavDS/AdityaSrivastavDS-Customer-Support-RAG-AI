import sqlite3
from pathlib import Path
from typing import Optional

def init_sqlite(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_question TEXT,
        model_answer TEXT,
        resolved INTEGER,          -- 1 thumbs up, 0 thumbs down
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

def store_feedback(db_path: Path, user_q: str, answer: str, resolved: bool):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO feedback (user_question, model_answer, resolved) VALUES (?, ?, ?)", (user_q, answer, int(resolved)))
    conn.commit()
    conn.close()
