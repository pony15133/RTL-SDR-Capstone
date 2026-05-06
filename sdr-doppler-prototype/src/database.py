import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS capture_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input_file TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    detection_result INTEGER NOT NULL,
    confidence_score REAL NOT NULL,
    valid_signal_ratio REAL NOT NULL,
    frequency_drift_hz REAL NOT NULL,
    smoothness_score REAL NOT NULL,
    spectrogram_image_path TEXT,
    raw_iq_file_path TEXT,
    notes TEXT
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def insert_result(db_path: Path, row: dict) -> int:
    init_db(db_path)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    values = list(row.values())
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO capture_results ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return int(cur.lastrowid)
