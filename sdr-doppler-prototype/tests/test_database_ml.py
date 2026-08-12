"""Tests for the ML-related SQLite schema/migration (src/database.py)."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import ML_COLUMNS, get_result, init_db, insert_result  # noqa: E402

BASE_ROW = {
    "input_file": "data/raw/sample.npy",
    "timestamp_utc": "2026-08-09T12:00:00+00:00",
    "detection_result": 1,
    "confidence_score": 0.9,
    "valid_signal_ratio": 0.8,
    "frequency_drift_hz": 10000.0,
    "smoothness_score": 100.0,
    "spectrogram_image_path": None,
    "raw_iq_file_path": None,
    "notes": "test row",
}


class TestFreshDatabase:
    def test_init_db_creates_all_expected_columns(self, tmp_path):
        db_path = tmp_path / "fresh.sqlite3"
        init_db(db_path)

        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(capture_results)")}

        expected = {
            "id", "input_file", "timestamp_utc", "detection_result", "confidence_score",
            "valid_signal_ratio", "frequency_drift_hz", "smoothness_score",
            "spectrogram_image_path", "raw_iq_file_path", "notes",
            *ML_COLUMNS.keys(),
        }
        assert expected.issubset(columns)

    def test_insert_result_with_ml_fields(self, tmp_path):
        db_path = tmp_path / "fresh.sqlite3"
        row = {
            **BASE_ROW,
            "rule_detection_result": 1,
            "rule_confidence_score": 0.9,
            "ml_detection_result": 1,
            "ml_confidence_score": 0.87,
            "model_version": "random_forest_20260809T000000Z",
        }

        result_id = insert_result(db_path, row)
        stored = get_result(db_path, result_id)

        assert stored["rule_detection_result"] == 1
        assert stored["rule_confidence_score"] == 0.9
        assert stored["ml_detection_result"] == 1
        assert stored["ml_confidence_score"] == 0.87
        assert stored["model_version"] == "random_forest_20260809T000000Z"

    def test_insert_result_with_no_ml_model_available(self, tmp_path):
        """MODEL_NOT_AVAILABLE: ml_* columns and model_version should store as NULL, not crash."""
        db_path = tmp_path / "fresh.sqlite3"
        row = {
            **BASE_ROW,
            "rule_detection_result": 1,
            "rule_confidence_score": 0.9,
            "ml_detection_result": None,
            "ml_confidence_score": None,
            "model_version": None,
        }

        result_id = insert_result(db_path, row)
        stored = get_result(db_path, result_id)

        assert stored["ml_detection_result"] is None
        assert stored["ml_confidence_score"] is None
        assert stored["model_version"] is None
        # the rule-based baseline result must still be present and usable
        assert stored["rule_detection_result"] == 1

    def test_insert_result_without_ml_keys_at_all_still_works(self, tmp_path):
        """Backward compatibility: a row dict with only the original columns must still insert."""
        db_path = tmp_path / "fresh.sqlite3"

        result_id = insert_result(db_path, dict(BASE_ROW))
        stored = get_result(db_path, result_id)

        assert stored["detection_result"] == 1
        assert stored["ml_detection_result"] is None
        assert stored["model_version"] is None


class TestMigrationOfExistingDatabase:
    def _make_legacy_db(self, db_path: Path) -> None:
        """A database created with the pre-ML schema, populated the way the
        original (pre-ML) database.py would have."""
        legacy_schema = """
        CREATE TABLE capture_results (
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
        with sqlite3.connect(db_path) as conn:
            conn.execute(legacy_schema)
            conn.execute(
                "INSERT INTO capture_results "
                "(input_file, timestamp_utc, detection_result, confidence_score, valid_signal_ratio, "
                " frequency_drift_hz, smoothness_score, spectrogram_image_path, raw_iq_file_path, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("legacy.npy", "2025-01-01T00:00:00+00:00", 1, 0.75, 0.6, 5000.0, 200.0, None, None, "legacy row"),
            )
            conn.commit()

    def test_migration_adds_ml_columns_without_losing_data(self, tmp_path):
        db_path = tmp_path / "legacy.sqlite3"
        self._make_legacy_db(db_path)

        init_db(db_path)  # this is the migration step

        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(capture_results)")}
            row_count = conn.execute("SELECT COUNT(*) FROM capture_results").fetchone()[0]
            legacy_row = conn.execute("SELECT input_file, detection_result FROM capture_results WHERE id = 1").fetchone()

        assert ML_COLUMNS.keys() <= columns
        assert row_count == 1
        assert legacy_row == ("legacy.npy", 1)

    def test_migration_is_idempotent(self, tmp_path):
        db_path = tmp_path / "legacy.sqlite3"
        self._make_legacy_db(db_path)

        init_db(db_path)
        init_db(db_path)  # must not raise "duplicate column" on a second call

        with sqlite3.connect(db_path) as conn:
            row_count = conn.execute("SELECT COUNT(*) FROM capture_results").fetchone()[0]
        assert row_count == 1

    def test_legacy_rows_have_null_ml_fields_after_migration(self, tmp_path):
        db_path = tmp_path / "legacy.sqlite3"
        self._make_legacy_db(db_path)

        init_db(db_path)
        stored = get_result(db_path, 1)

        assert stored["ml_detection_result"] is None
        assert stored["ml_confidence_score"] is None
        assert stored["model_version"] is None
        assert stored["rule_detection_result"] is None  # never backfilled from detection_result - just added as NULL
