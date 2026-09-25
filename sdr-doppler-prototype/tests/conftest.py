"""Shared test setup: reviews in tests must never write the real station training files."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(autouse=True)
def _station_training_files_in_tmp(tmp_path, monkeypatch):
    import review

    monkeypatch.setattr(review, "STATION_DATASET", tmp_path / "station_waterfall_features.csv")
    monkeypatch.setattr(review, "STATION_IQ_DATASET", tmp_path / "station_iq_features.csv")
