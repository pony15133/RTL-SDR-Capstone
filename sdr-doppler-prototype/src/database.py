import sqlite3
from pathlib import Path

# Original schema (unchanged - existing databases/readers keep working).
# `detection_result`/`confidence_score` ARE the rule-based detector's
# result columns and have always been populated by it; they are kept
# exactly as-is for backward compatibility.
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

# Columns added for the ML component. `rule_detection_result` /
# `rule_confidence_score` duplicate `detection_result` / `confidence_score`
# under names that read naturally side-by-side with `ml_detection_result` /
# `ml_confidence_score` in a comparison query - a deliberate, small
# redundancy in exchange for self-documenting rows. `ml_*` columns and
# `model_version` are NULL whenever no ML model was available for a given
# capture (see detection/ml_detector.py MODEL_NOT_AVAILABLE).
ML_COLUMNS = {
    "rule_detection_result": "INTEGER",
    "rule_confidence_score": "REAL",
    "ml_detection_result": "INTEGER",
    "ml_confidence_score": "REAL",
    "model_version": "TEXT",
}


def _existing_columns(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _migrate_ml_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add the ML columns to the capture_results table.

    Runs after CREATE TABLE IF NOT EXISTS on every init_db() call, so it's
    a no-op on a database that already has these columns, and a safe,
    non-destructive migration (existing rows just get NULL for the new
    columns) on a database created before the ML component existed.
    """
    existing = _existing_columns(conn, "capture_results")
    for column, sql_type in ML_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE capture_results ADD COLUMN {column} {sql_type}")


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA)
        _migrate_ml_columns(conn)
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


def get_result(db_path: Path, result_id: int) -> dict:
    """Fetch one capture_results row as a dict (column name -> value). Used by tests/tooling."""
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM capture_results WHERE id = ?", (result_id,))
        row = cur.fetchone()
        return dict(row) if row else None
