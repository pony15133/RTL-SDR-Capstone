# Simulated end-to-end pipeline verification

Generated 2026-09-24T09:07:01+00:00 on commit `ecbe2781682ad0fa4466d63b6216b0d86ca22f12` by `tools/verify_simulated_pipeline.py` (Python 3.11.15, Linux-6.18.44-fc-v37-x86_64-with-glibc2.39).

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
| retention decision stored | PASS | archived: ML confidence 0.14 < threshold 0.50 |
| IQ file where the row says it is (or deleted) | PASS | /tmp/sdr_verify_ezzprczk/demo/recordings/rejected/METEOR-M2-3_20260924_090640_137750000Hz.iq |
| recorder JSON sidecar exists | PASS |  |
| spectrogram PNG exists | PASS |  |
| summary JSON exists | PASS |  |
| no Doppler correction on the TLE-less demo pass (expected) | PASS |  |
| status_log has scheduler WAITING, recorder SUCCESS, pipeline DONE | PASS | [('pipeline', 'DONE'), ('recorder', 'SUCCESS'), ('scheduler', 'WAITING')] |
| schedule.json written | PASS |  |
| heartbeat live_status.json written, final phase 'stopped' | PASS | stopped |
| dashboard snapshot shows the capture and totals | PASS | totals {'n': 1, 'ok': 1, 'detected': 0, 'kept': 0, 'archived': 1, 'deleted': 0} |

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
 "scheduled_aos": "2026-09-24T09:06:41.398658+00:00",
 "scheduled_los": "2026-09-24T09:06:51.398658+00:00",
 "actual_recording_start": "2026-09-24T09:06:40.399276+00:00",
 "actual_recording_stop": "2026-09-24T09:06:52.399639+00:00",
 "recording_duration_seconds": 12.000363,
 "output_file_size": 65536,
 "simulated": 1,
 "processing_status": "DETECTED",
 "rule_detection_result": 0,
 "rule_confidence_score": 0.6666666666666666,
 "ml_detection_result": 0,
 "ml_confidence_score": 0.1407142857142857,
 "model_version": "rsp03_iq_rf",
 "wf_ml_detection_result": null,
 "doppler_corrected": 0,
 "doppler_max_hz": null,
 "decision_source": "ml",
 "decision_score": 0.1407142857142857,
 "iq_retention": "archived",
 "retention_reason": "ML confidence 0.14 < threshold 0.50",
 "raw_iq_file_path": "/tmp/sdr_verify_ezzprczk/demo/recordings/rejected/METEOR-M2-3_20260924_090640_137750000Hz.iq",
 "metadata_file_path": "/tmp/sdr_verify_ezzprczk/demo/recordings/METEOR-M2-3_20260924_090640_137750000Hz.json",
 "spectrogram_image_path": "/tmp/sdr_verify_ezzprczk/demo/results/METEOR-M2-3_20260924_090640_137750000Hz_2026-09-24T090652+0000_spectrogram.png",
 "waterfall_image_path": "/tmp/sdr_verify_ezzprczk/demo/results/METEOR-M2-3_20260924_090640_137750000Hz_2026-09-24T090652+0000_waterfall.png"
}
```

## B: TLE pass via run_plan

| Check | Result | Detail |
|---|---|---|
| recording SUCCESS | PASS | SUCCESS |
| Doppler curve computed from the TLE and applied | PASS | max /Doppler/ 9688 Hz |
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
 "scheduled_aos": "2026-09-24T09:06:55.611805+00:00",
 "scheduled_los": "2026-09-24T09:06:59.611805+00:00",
 "actual_recording_start": "2026-09-24T09:06:55.112662+00:00",
 "actual_recording_stop": "2026-09-24T09:07:00.112878+00:00",
 "recording_duration_seconds": 5.000216,
 "output_file_size": 65536,
 "simulated": 1,
 "processing_status": "DETECTED",
 "rule_detection_result": 0,
 "rule_confidence_score": 0.6666666666666666,
 "ml_detection_result": 0,
 "ml_confidence_score": 0.1427142857142857,
 "model_version": "rsp03_iq_rf",
 "wf_ml_detection_result": null,
 "doppler_corrected": 1,
 "doppler_max_hz": 9688.237758278008,
 "decision_source": "ml",
 "decision_score": 0.1427142857142857,
 "iq_retention": "archived",
 "retention_reason": "ML confidence 0.14 < threshold 0.50",
 "raw_iq_file_path": "/tmp/sdr_verify_ezzprczk/tle/recordings/rejected/ISS_20260924_090655_437650000Hz.iq",
 "metadata_file_path": "/tmp/sdr_verify_ezzprczk/tle/recordings/ISS_20260924_090655_437650000Hz.json",
 "spectrogram_image_path": "/tmp/sdr_verify_ezzprczk/tle/results/ISS_20260924_090655_437650000Hz_2026-09-24T090700+0000_spectrogram.png",
 "waterfall_image_path": "/tmp/sdr_verify_ezzprczk/tle/results/ISS_20260924_090655_437650000Hz_2026-09-24T090700+0000_waterfall.png"
}
```
