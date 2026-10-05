# Simulated end-to-end pipeline verification

Generated 2026-10-05T12:11:52+00:00 on commit `040c140f28ce292f3b93299ae403604067b9327d` by `tools/verify_simulated_pipeline.py` (Python 3.13.16, Linux-6.18.44-fc-v70-x86_64-with-glibc2.39).

**Result: ALL CHECKS PASSED**

Simulated recorder output (random bytes): detection outcomes are meaningless; this verifies that every stage runs and stores its results. Not live-hardware evidence.

## A: auto_capture --demo

| Check | Result | Detail |
|---|---|---|
| auto_capture --demo exit code 0 | PASS | exit 0 |
| one capture_results row | PASS | 1 row(s) |
| recording SUCCESS and processed | PASS | SUCCESS/DETECTED |
| scheduled AOS/LOS and actual start/stop stored | PASS |  |
| offset tuning: tuned 150 kHz below the downlink | PASS | target 137900000 tuned 137750000 |
| rule detector result stored | PASS |  |
| ML detector ran (model present) | PASS | model_version rsp03_iq_rf |
| retention decision stored | PASS | archived: ML confidence 0.29 < 0.30: no satellite |
| IQ file where the row says it is (or deleted) | PASS | /tmp/sdr_verify_lioyjm24/demo/recordings/rejected/METEOR-M2-3_20261005_121132_137750000Hz.iq |
| recorder JSON sidecar exists | PASS |  |
| spectrogram PNG exists | PASS |  |
| summary JSON exists | PASS |  |
| no Doppler correction on the TLE-less demo pass (expected) | PASS |  |
| status_log has scheduler WAITING, recorder SUCCESS, pipeline DONE | PASS | [('pipeline', 'DONE'), ('recorder', 'SUCCESS'), ('scheduler', 'WAITING')] |
| schedule.json written | PASS |  |
| heartbeat live_status.json written, final phase 'stopped' | PASS | stopped |
| dashboard snapshot shows the capture and totals | PASS | totals {'n': 1, 'ok': 1, 'failed': 0, 'kept': 0, 'archived': 1, 'deleted': 0, 'detected': 0, 'verdicts': {'detected': 0, 'uncertain': 0, 'not_detected': 1}, 'pe |

Stored `capture_results` row (selected fields):

```json
{
 "satellite_name": "METEOR-M2-3",
 "norad_id": 57166,
 "frequency_hz": 137750000,
 "target_frequency_hz": 137900000,
 "sample_rate": 1024000,
 "gain": 30.0,
 "recording_status": "SUCCESS",
 "scheduled_aos": "2026-10-05T12:11:33.075352+00:00",
 "scheduled_los": "2026-10-05T12:11:43.075352+00:00",
 "actual_recording_start": "2026-10-05T12:11:32.076143+00:00",
 "actual_recording_stop": "2026-10-05T12:11:44.076456+00:00",
 "recording_duration_seconds": 12.000313,
 "output_file_size": 65536,
 "simulated": 1,
 "processing_status": "DETECTED",
 "rule_detection_result": 0,
 "rule_confidence_score": 0.6666666666666666,
 "ml_detection_result": 0,
 "ml_confidence_score": 0.1407142857142857,
 "model_version": "rsp03_iq_rf",
 "wf_ml_detection_result": 0,
 "doppler_corrected": 0,
 "doppler_max_hz": null,
 "decision_source": "waterfall-ml",
 "decision_score": 0.28928892211530277,
 "iq_retention": "archived",
 "retention_reason": "ML confidence 0.29 < 0.30: no satellite",
 "raw_iq_file_path": "/tmp/sdr_verify_lioyjm24/demo/recordings/rejected/METEOR-M2-3_20261005_121132_137750000Hz.iq",
 "metadata_file_path": "/tmp/sdr_verify_lioyjm24/demo/recordings/METEOR-M2-3_20261005_121132_137750000Hz.json",
 "spectrogram_image_path": "/tmp/sdr_verify_lioyjm24/demo/results/METEOR-M2-3_20261005_121132_137750000Hz_2026-10-05T121144+0000_spectrogram.png",
 "waterfall_image_path": "/tmp/sdr_verify_lioyjm24/demo/results/METEOR-M2-3_20261005_121132_137750000Hz_2026-10-05T121144+0000_waterfall.png"
}
```

## B: TLE pass via run_plan

| Check | Result | Detail |
|---|---|---|
| recording SUCCESS | PASS | SUCCESS |
| Doppler curve computed from the TLE and applied | PASS | max /Doppler/ 4470 Hz |
| Doppler-corrected waterfall PNG exists | PASS |  |
| pass_positions rows stored (az/el/range/Doppler) | PASS | 1 point(s) |

Stored `capture_results` row (selected fields):

```json
{
 "satellite_name": "ISS",
 "norad_id": 25544,
 "frequency_hz": 437650000,
 "target_frequency_hz": 437800000,
 "sample_rate": 1024000,
 "gain": 35.0,
 "recording_status": "SUCCESS",
 "scheduled_aos": "2026-10-05T12:11:47.200142+00:00",
 "scheduled_los": "2026-10-05T12:11:51.200142+00:00",
 "actual_recording_start": "2026-10-05T12:11:46.700785+00:00",
 "actual_recording_stop": "2026-10-05T12:11:51.701023+00:00",
 "recording_duration_seconds": 5.000238,
 "output_file_size": 65536,
 "simulated": 1,
 "processing_status": "DETECTED",
 "rule_detection_result": 0,
 "rule_confidence_score": 0.6666666666666666,
 "ml_detection_result": 0,
 "ml_confidence_score": 0.1407142857142857,
 "model_version": "rsp03_iq_rf",
 "wf_ml_detection_result": 0,
 "doppler_corrected": 1,
 "doppler_max_hz": 4470.283783318457,
 "decision_source": "waterfall-ml",
 "decision_score": 0.33145751610056484,
 "iq_retention": "kept for review",
 "retention_reason": "ML confidence 0.33 between 0.30 and 0.70: uncertain - kept for review",
 "raw_iq_file_path": "/tmp/sdr_verify_lioyjm24/tle/recordings/uncertain/ISS_20261005_121146_437650000Hz.iq",
 "metadata_file_path": "/tmp/sdr_verify_lioyjm24/tle/recordings/ISS_20261005_121146_437650000Hz.json",
 "spectrogram_image_path": "/tmp/sdr_verify_lioyjm24/tle/results/ISS_20261005_121146_437650000Hz_2026-10-05T121151+0000_spectrogram.png",
 "waterfall_image_path": "/tmp/sdr_verify_lioyjm24/tle/results/ISS_20261005_121146_437650000Hz_2026-10-05T121151+0000_waterfall.png"
}
```
