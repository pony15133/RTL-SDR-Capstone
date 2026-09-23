import sqlite3
from pathlib import Path
from contextlib import closing

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

CAPTURE_COLUMNS = {
    "satellite_name": "TEXT",
    "norad_id": "INTEGER",
    "frequency_hz": "INTEGER",
    "sample_rate": "INTEGER",
    "gain": "REAL",
    "recording_status": "TEXT",
    "scheduled_aos": "TEXT",
    "scheduled_los": "TEXT",
    "actual_recording_start": "TEXT",
    "actual_recording_stop": "TEXT",
    "recording_duration_seconds": "REAL",
    "output_file_size": "INTEGER",
    "expected_file_size": "INTEGER",
    "simulated": "INTEGER",
    "device_index": "INTEGER",
    "metadata_file_path": "TEXT",
}

#: Added with the automatic capture loop: what happened to the raw IQ file
#: after detection, and why (see src/retention.py). `processing_status` says
#: whether detection ran at all - failed/busy recordings are logged too, so
#: every recorder run leaves a row.
RETENTION_COLUMNS = {
    "processing_status": "TEXT",
    "iq_retention": "TEXT",
    "retention_reason": "TEXT",
    "decision_source": "TEXT",
    "decision_score": "REAL",
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


def _migrate_capture_columns(conn: sqlite3.Connection) -> None:
    """Idempotently add capture metadata columns to the capture_results table."""
    existing = _existing_columns(conn, "capture_results")

    for column, sql_type in CAPTURE_COLUMNS.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE capture_results ADD COLUMN {column} {sql_type}"
            )


#: Satellite position history: where the satellite was (az/el/range and
#: predicted Doppler) during each recorded pass, linked to its capture row.
#: Status log: timestamped events from the scheduler/recorder/pipeline, so
#: the station's state can be monitored and failures diagnosed afterwards.
EXTRA_TABLES = """
CREATE TABLE IF NOT EXISTS pass_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_id INTEGER,
    norad_id INTEGER,
    satellite_name TEXT,
    timestamp_utc TEXT NOT NULL,
    azimuth_deg REAL,
    elevation_deg REAL,
    range_km REAL,
    doppler_hz REAL
);
CREATE INDEX IF NOT EXISTS idx_pass_positions_capture ON pass_positions(capture_id);
CREATE TABLE IF NOT EXISTS status_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    component TEXT NOT NULL,
    state TEXT NOT NULL,
    message TEXT
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(SCHEMA)
        conn.executescript(EXTRA_TABLES)
        _migrate_ml_columns(conn)
        _migrate_capture_columns(conn)
        existing = _existing_columns(conn, "capture_results")
        for column, sql_type in RETENTION_COLUMNS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE capture_results ADD COLUMN {column} {sql_type}")
        conn.commit()


def insert_result(db_path: Path, row: dict) -> int:
    init_db(db_path)
    columns = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    values = list(row.values())
    with closing(sqlite3.connect(db_path)) as conn:
        cur = conn.execute(
            f"INSERT INTO capture_results ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        return int(cur.lastrowid)


def get_result(db_path: Path, result_id: int) -> dict:
    """Fetch one capture_results row as a dict (column name -> value). Used by tests/tooling."""
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM capture_results WHERE id = ?", (result_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def update_result(db_path: Path, result_id: int, changes: dict) -> None:
    """Update columns of one existing row (e.g. after the IQ file was archived)."""
    if not changes:
        return
    assignments = ", ".join(f"{column} = ?" for column in changes)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(f"UPDATE capture_results SET {assignments} WHERE id = ?", [*changes.values(), result_id])
        conn.commit()


def list_results(db_path: Path, limit: int = 50) -> list:
    """Most recent rows first - used by the capture-history view."""
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM capture_results ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [dict(r) for r in rows]


def insert_pass_positions(db_path: Path, capture_id, positions: list) -> int:
    """Store a satellite track: positions = [{timestamp_utc, azimuth_deg,
    elevation_deg, range_km, doppler_hz, norad_id, satellite_name}, ...]."""
    if not positions:
        return 0
    init_db(db_path)
    cols = ("capture_id", "norad_id", "satellite_name", "timestamp_utc", "azimuth_deg", "elevation_deg",
            "range_km", "doppler_hz")
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executemany(
            f"INSERT INTO pass_positions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [tuple(capture_id if c == "capture_id" else p.get(c) for c in cols) for p in positions],
        )
        conn.commit()
    return len(positions)


def get_pass_positions(db_path: Path, capture_id: int) -> list:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(
            "SELECT * FROM pass_positions WHERE capture_id = ? ORDER BY timestamp_utc", (capture_id,))]


def log_status(db_path: Path, component: str, state: str, message: str = "") -> None:
    """Append a status event (e.g. scheduler WAITING, recorder RECORDING, pipeline ERROR)."""
    from datetime import datetime, timezone

    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("INSERT INTO status_log (timestamp_utc, component, state, message) VALUES (?, ?, ?, ?)",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"), component, state, message))
        conn.commit()


def list_status(db_path: Path, limit: int = 50) -> list:
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM status_log ORDER BY id DESC LIMIT ?", (int(limit),))]
