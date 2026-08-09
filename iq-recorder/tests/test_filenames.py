"""Tests for filename sanitisation, generation, and collision handling."""

from datetime import datetime

from rtl_recorder.filenames import build_base_filename, sanitize_satellite_name, unique_output_paths


class TestSanitizeSatelliteName:
    def test_leaves_simple_name_untouched(self):
        assert sanitize_satellite_name("METEOR-M2-4") == "METEOR-M2-4"

    def test_replaces_spaces(self):
        assert sanitize_satellite_name("NOAA 19") == "NOAA_19"

    def test_replaces_slashes_and_special_chars(self):
        assert sanitize_satellite_name("ISS (ZARYA)/Test") == "ISS_ZARYA_Test"

    def test_collapses_repeated_separators(self):
        assert sanitize_satellite_name("A!!!B") == "A_B"

    def test_strips_leading_trailing_underscores(self):
        assert sanitize_satellite_name("  ***KISS92***  ") == "KISS92"

    def test_empty_name_falls_back_to_unknown(self):
        assert sanitize_satellite_name("") == "UNKNOWN"
        assert sanitize_satellite_name("   ") == "UNKNOWN"
        assert sanitize_satellite_name("!!!") == "UNKNOWN"


class TestBuildBaseFilename:
    def test_matches_expected_format(self):
        ts = datetime(2026, 8, 9, 19, 34, 30)
        name = build_base_filename("METEOR-M2-4", ts, 137_900_000)
        assert name == "METEOR-M2-4_20260809_193430_137900000Hz"

    def test_sanitises_satellite_name(self):
        ts = datetime(2026, 8, 9, 19, 34, 30)
        name = build_base_filename("KISS 92 Singapore", ts, 92_000_000)
        assert name.startswith("KISS_92_Singapore_")
        assert name.endswith("92000000Hz")


class TestUniqueOutputPaths:
    def test_returns_iq_and_json_paths(self, tmp_path):
        iq_path, meta_path = unique_output_paths(tmp_path, "SAT_20260809_193430_137900000Hz")
        assert iq_path == tmp_path / "SAT_20260809_193430_137900000Hz.iq"
        assert meta_path == tmp_path / "SAT_20260809_193430_137900000Hz.json"

    def test_never_overwrites_existing_iq_file(self, tmp_path):
        base = "SAT_20260809_193430_137900000Hz"
        (tmp_path / f"{base}.iq").write_bytes(b"existing data")

        iq_path, meta_path = unique_output_paths(tmp_path, base)

        assert iq_path.name == f"{base}_2.iq"
        assert meta_path.name == f"{base}_2.json"
        assert not iq_path.exists()

    def test_never_overwrites_existing_metadata_file(self, tmp_path):
        base = "SAT_20260809_193430_137900000Hz"
        (tmp_path / f"{base}.json").write_text("{}")

        iq_path, meta_path = unique_output_paths(tmp_path, base)

        assert iq_path.name == f"{base}_2.iq"

    def test_handles_multiple_collisions(self, tmp_path):
        base = "SAT_20260809_193430_137900000Hz"
        (tmp_path / f"{base}.iq").write_bytes(b"1")
        (tmp_path / f"{base}_2.iq").write_bytes(b"2")
        (tmp_path / f"{base}_3.iq").write_bytes(b"3")

        iq_path, _ = unique_output_paths(tmp_path, base)

        assert iq_path.name == f"{base}_4.iq"
