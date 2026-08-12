"""Tests for the shared training-CSV I/O helpers (src/features/dataset_io.py)."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.dataset_io import (  # noqa: E402
    CSV_COLUMNS,
    DatasetSchemaError,
    append_row,
    existing_source_files,
    prompt_for_label,
)


def _row(capture_id="cap1", source_file="a.bin", label=1):
    row = {name: 0.0 for name in CSV_COLUMNS}
    row["capture_id"] = capture_id
    row["source_file"] = source_file
    row["label"] = label
    row["is_synthetic"] = 0
    row["notes"] = ""
    return row


class TestAppendRow:
    def test_writes_header_once(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        append_row(csv_path, _row("cap1"))
        append_row(csv_path, _row("cap2"))

        with csv_path.open() as handle:
            lines = handle.readlines()
        header_lines = [line for line in lines if line.startswith("capture_id,")]
        assert len(header_lines) == 1
        assert len(lines) == 3  # header + 2 rows

    def test_creates_parent_directory(self, tmp_path):
        csv_path = tmp_path / "nested" / "dir" / "dataset.csv"
        append_row(csv_path, _row())
        assert csv_path.exists()

    def test_row_values_round_trip(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        append_row(csv_path, _row(capture_id="cap42", source_file="x.bin", label=1))

        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["capture_id"] == "cap42"
        assert rows[0]["source_file"] == "x.bin"
        assert rows[0]["label"] == "1"

    def test_incompatible_existing_header_raises(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        csv_path.write_text("capture_id,some_old_column,label\ncap0,1.0,1\n")

        with pytest.raises(DatasetSchemaError):
            append_row(csv_path, _row("cap1"))

        # the incompatible file must be left untouched, not partially written
        assert csv_path.read_text() == "capture_id,some_old_column,label\ncap0,1.0,1\n"

    def test_matching_header_on_existing_file_appends_normally(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        append_row(csv_path, _row("cap1"))
        append_row(csv_path, _row("cap2"))  # same schema - must not raise

        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2


class TestExistingSourceFiles:
    def test_missing_file_returns_empty_set(self, tmp_path):
        assert existing_source_files(tmp_path / "does_not_exist.csv") == set()

    def test_empty_file_returns_empty_set(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.touch()
        assert existing_source_files(csv_path) == set()

    def test_returns_all_source_files(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        append_row(csv_path, _row("cap1", source_file="a.bin"))
        append_row(csv_path, _row("cap2", source_file="b.bin"))

        assert existing_source_files(csv_path) == {"a.bin", "b.bin"}

    def test_csv_without_source_file_column_returns_empty_set(self, tmp_path):
        csv_path = tmp_path / "dataset.csv"
        csv_path.write_text("capture_id,label\ncap1,1\n")
        assert existing_source_files(csv_path) == set()


class TestPromptForLabel:
    def test_accepts_one(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "1")
        assert prompt_for_label() == 1

    def test_accepts_zero(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "0")
        assert prompt_for_label() == 0

    def test_skip_returns_none(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "s")
        assert prompt_for_label() is None

    def test_empty_input_returns_none(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert prompt_for_label() is None

    def test_reprompts_on_invalid_then_accepts(self, monkeypatch):
        responses = iter(["banana", "1"])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        assert prompt_for_label() == 1

    def test_context_is_shown_in_prompt(self, monkeypatch):
        seen_prompts = []

        def fake_input(prompt_text):
            seen_prompts.append(prompt_text)
            return "0"

        monkeypatch.setattr("builtins.input", fake_input)
        prompt_for_label(context="/tmp/capture123.bin")

        assert "/tmp/capture123.bin" in seen_prompts[0]
