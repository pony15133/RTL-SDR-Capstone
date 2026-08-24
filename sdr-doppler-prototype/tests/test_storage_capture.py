from pathlib import Path
import sys
import tempfile

# Allow this test file to import modules from ../src
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from database import insert_result, get_result


def main():
    # Use a temporary SQLite database so the real project database is untouched.
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_storage.sqlite3"

        test_row = {
            # Existing capture/detection fields
            "input_file": "recordings/test_capture.iq",
            "timestamp_utc": "2026-08-12T10:00:00+00:00",
            "detection_result": 1,
            "confidence_score": 0.82,
            "valid_signal_ratio": 0.91,
            "frequency_drift_hz": 1450.0,
            "smoothness_score": 0.88,
            "spectrogram_image_path": "results/test_spectrogram.png",
            "raw_iq_file_path": "recordings/test_capture.iq",
            "notes": "Storage component integration test",

            # Rule-based + ML results
            "rule_detection_result": 1,
            "rule_confidence_score": 0.82,
            "ml_detection_result": 1,
            "ml_confidence_score": 0.93,
            "model_version": "random-forest-test-v1",

            # Recorder/capture metadata
            "satellite_name": "TEST-SAT",
            "norad_id": 99999,
            "frequency_hz": 137100000,
            "sample_rate": 2400000,
            "gain": 28.0,
            "recording_status": "SUCCESS",
            "scheduled_aos": "2026-08-12T09:59:55+00:00",
            "scheduled_los": "2026-08-12T10:00:35+00:00",
            "actual_recording_start": "2026-08-12T10:00:00+00:00",
            "actual_recording_stop": "2026-08-12T10:00:30+00:00",
            "recording_duration_seconds": 30.0,
            "output_file_size": 144000000,
            "expected_file_size": 144000000,
            "simulated": 1,
            "device_index": 0,
            "metadata_file_path": "recordings/test_capture.json",
        }

        # Insert into SQLite
        result_id = insert_result(db_path, test_row)

        # Retrieve the same record
        stored = get_result(db_path, result_id)

        assert stored is not None

        # Verify every value we inserted was retrieved correctly.
        for key, expected_value in test_row.items():
            actual_value = stored[key]
            assert actual_value == expected_value, (
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )

        print("Storage test PASSED")
        print(f"Result ID: {result_id}")
        print(f"Satellite: {stored['satellite_name']}")
        print(f"Frequency: {stored['frequency_hz']} Hz")
        print(f"Duration: {stored['recording_duration_seconds']} seconds")
        print(f"Rule detection: {stored['rule_detection_result']}")
        print(f"Rule confidence: {stored['rule_confidence_score']}")
        print(f"ML detection: {stored['ml_detection_result']}")
        print(f"ML confidence: {stored['ml_confidence_score']}")
        print(f"Model version: {stored['model_version']}")
        print("Insert and retrieval values match.")


if __name__ == "__main__":
    main()