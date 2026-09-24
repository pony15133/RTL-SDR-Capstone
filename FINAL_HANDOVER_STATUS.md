# Final handover status

**Code commit:** `ecbe2781682ad0fa4466d63b6216b0d86ca22f12` (branch `claude/laughing-darwin-6rmhv6`, based on
`main` at `e8fd55f`). The evidence files in `evidence/` were generated from this commit, except the performance
benchmark (from `7ed7ec4`; the processing code it measures has not changed since). Date: 24 Sep 2026.

**Relationship to earlier documents.** The code has moved on since Milestone 4 / Sprint Report 4 (19 Sep 2026) and
the internal Handover Materials draft V0.9. Those documents are historical records and are left as written. Several of
their statements no longer describe this code: `record_pass()` raising `NotImplementedError`, "no TLE/SatNOGS pass
prediction", "127/127 tests", "ML trained on synthetic data only", `label_lora_chunks.py`. The section
*For the handover pack* at the end lists what the V1.0 pack needs to say instead.

**How to read this.** "Verified" means verified by the automated or simulated evidence listed. Nothing in this file
has been verified on a real RTL-SDR or on Windows. A requirement is only listed as verified where there is evidence
for it. Final ratings in the requirement matrix are for the client and supervisor to give.

## 1. Implemented and verified (automated / simulated evidence)

| Capability | Evidence |
|---|---|
| Automatic pass prediction from TLEs (AOS/LOS/max elevation/Doppler; sgp4, or a built-in Kepler+J2 fallback) | `iq-recorder/tests/test_passes.py`; `auto_capture.py --list-only` |
| `record_pass()`: wait for the pass, record AOS - buffer to LOS + buffer, cancellable (simulated recorder) | `iq-recorder/tests/test_passes.py`, `tests/test_auto_capture.py`; `evidence/pipeline/` |
| Full automatic chain in simulation: schedule, then `record_pass()`, spectrogram and features, rule + ML detection, Doppler-corrected waterfall, SQLite row, retention, status log, dashboard heartbeat | `evidence/pipeline/SIMULATED_PIPELINE.md` (21/21 checks) |
| Doppler correction from the TLE prediction + offset tuning (DC spike kept off the signal) | run B in `evidence/pipeline/`; `tests/test_auto_capture.py::test_offset_tuning_doppler_and_waterfall_are_recorded` |
| Position histories (`pass_positions`) and status monitoring (`status_log`) | `evidence/pipeline/`; `sdr-doppler-prototype/tests/test_history_gui.py` |
| SQLite storage of capture, recorder, detection, model-version and retention fields; non-destructive migration of older databases | `tests/test_pipeline.py`, `sdr-doppler-prototype/tests/test_database_ml.py`, `test_storage_capture.py` |
| Selective retention: keep-all / archive-negatives / delete-negatives, with the reason stored per row | `tests/test_pipeline.py` (archive, delete, positive kept, unknown policy rejected) |
| Rule-based detector (peaks, drift, smoothness) | `sdr-doppler-prototype/tests/test_detect.py`, `test_features.py` |
| Random Forest training with a grouped train/validation/test split (no recording in two splits), and inference that falls back to `MODEL_NOT_AVAILABLE` | `test_ml_training.py`, `test_ml_inference.py`, `test_ml_evaluation_report.py`; `evidence/ml/` |
| REQ-1 text/CSV spectrogram input through to detection, JSON, PNG and database | `sdr-doppler-prototype/tests/test_text_input_pipeline.py` (6 tests) |
| Waterfall visualisation without ML (`visualize.py`, GUI "Visualise IQ"); cu8 / ci16 / complex64 / WAV / SigMF readers | `sdr-doppler-prototype/tests/test_iq_io_visualize.py`, `test_convert_sigmf_to_iq.py` |
| Live dashboard: snapshot, HTTP endpoints, image path allow-list, and no crash on an empty, pre-ML, missing or corrupt database or a bad or missing heartbeat | `tests/test_dashboard.py` (10 tests) |
| Desktop GUI starts, builds all 7 tabs, and runs "Run Demo Capture" and "Show History" (automated, virtual display) | `evidence/gui/GUI_SMOKE.md` (6/6 checks, 9 screenshots) |
| Cross-platform tool discovery (`find_executable`, `RTL_SDR_HOME`) and environment doctor, **as unit tests on Linux** | `iq-recorder/tests/test_cross_platform.py` |
| Processing time and peak memory per recording | `evidence/performance/` |

## 2. Implemented, awaiting live-hardware verification

Procedure: [LIVE_HARDWARE_TEST.md](LIVE_HARDWARE_TEST.md). **No real-hardware run has been done in this handover
round. There is no dongle in the verification environment.**

- Real `rtl_sdr` capture of a satellite pass through `auto_capture.py` / `run_station`, and a `capture_results` row with `simulated = 0`
- Hardware test suite `iq-recorder/tests/hardware/test_hardware_kiss92.py` (3 tests, deselected by default, not run)
- `check_dongle` on a real dongle
- Pass-time accuracy against a real pass (the code is tested against the SGP4 reference case, not against the sky)
- Rule and ML detection on real RTL-SDR captures (see section 6: not validated)
- **Windows**: `setup.bat`, `check_dongle.bat`, `run_station.bat`, `launch_gui.bat` (added this round) and the
  CTRL_BREAK stop path are written for Windows but were **not executed on Windows** here
- The GUI on a real desktop, used by a person (the automated smoke test is not a usability check)
- The SatNOGS-trained waterfall model: `setup --fetch-satnogs` downloads from network.satnogs.org, which this
  environment could not reach (HTTP 403 from the proxy, and CelesTrak was also blocked). That model was **not trained
  or evaluated here**, and its external test on the RSP-03 waterfall set was **not run**

## 3. Partially implemented

| Item | What exists | What is missing |
|---|---|---|
| REQ-3 "validate input data" | Recorder parameter validation (`test_validation.py`, 26 tests); undersized-file check after recording; training-CSV validation; `load_input` rejects non-2D text; `probe()` format detection | No single input-validation layer. A text matrix with `nan` cells or a header row is not caught cleanly (header gives a raw `ValueError`) |
| NFR 5.3-a "basic file access control" | Dashboard listens on 127.0.0.1 by default, opens the database read-only, and serves only `.png` files inside allowed folders (tested) | No authentication (`--host 0.0.0.0` exposes it to the LAN); no OS permission handling |
| NFR 5.1 performance | Measured (section 7); the spectrogram image memory defect was fixed | About 6.7 GB peak per pass at the default settings (60 s window: about 3.5 GB) |
| GUI layout | All functions work | At the default 1150x820 window the bottom Results pane is squeezed to nothing, so the window must be maximised. Not changed, because it needs a layout rework |
| ML confidence scoring (post-SRS scope) | Pipeline, grouped evaluation and integration done | Real-world validation (section 6) |

## 4. Not implemented / deferred

- **REQ-12 "detect abnormal signal patterns" and REQ-13 "flag anomalies": not implemented.** Searching the code
  finds no signal-anomaly logic. The rule detector and the models only answer "satellite candidate: yes/no" (with a
  confidence). `status_log` records application events (scheduler WAITING, recorder FAILED, pipeline ERROR). Those
  are system/operational status, not abnormal *signal* patterns, and are not claimed as REQ-12/13. Nothing was added,
  because the SRS gives no definition of "abnormal" to implement against. Candidates for the client to choose from:
  a detected trace that does not follow the predicted Doppler curve, unexpected strong carriers, or rule/ML
  disagreement. Choosing one is a requirement decision, not a coding one.
- Per-satellite / multi-class classification (binary only)
- Live/interactive spectrogram view (static PNGs, the dashboard image and GUI preview only)
- Running as an OS service across reboots (`run_station` runs in a terminal window)
- Formal 99%-power occupied bandwidth (`occupied_bandwidth_hz` is a threshold-crossing estimate)

## 5. Test results

Full detail: `evidence/tests/TEST_RESULTS.md`.

| | Baseline `e8fd55f` | Final `ecbe278` |
|---|---|---|
| Passed | 334 | **354** |
| Failed / errors / skipped | 0 / 0 / 0 | **0 / 0 / 0** |
| Deselected (hardware, not run) | 3 | 3 |

From the repository root: `python -m pytest`. The environment was Python 3.11.15 on Linux (container, 4 CPUs,
16 GB); by project: iq-recorder 124, sdr-doppler-prototype 193, root `tests/` 37. There were no regressions to fix
on the baseline. The GUI smoke test ran separately under Python 3.12.3 / Tk 8.6 with a virtual display (Xvfb).

## 6. ML evaluation results (real data)

Full detail: `evidence/ml/ML_EVALUATION.md`, `ml_evaluation.json`, `split_assignment.csv`. Reproduce with
`cd sdr-doppler-prototype && python scripts/ml_evaluation_report.py`.

Dataset: `rsp03_camras_features.csv`, with 111 one-second windows (62 noise, 49 signal, 0 synthetic) from **4
independent recordings** of **one** satellite pass at **one** station (CAMRAS Dwingeloo 25 m dish, 436.95 MHz), **not
an RTL-SDR**. Split by recording (the repository's methodology: tuned on validation, refit on train + validation, test
scored once). Recordings per split: train 2 (t4, t5), validation 1 (t3), test 1 (t1). The script checks that no
recording appears in two splits.

| Set | Recordings | Windows | Accuracy | Precision | Recall | F1 | FPR | FNR | TN FP FN TP |
|---|---|---|---|---|---|---|---|---|---|
| Training (train+val, resubstitution) | 3 | 69 | 0.986 | 0.974 | 1.000 | 0.987 | 0.031 | 0.000 | 31 1 0 37 |
| Validation (chose hyperparameters) | 1 | 17 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 11 0 0 6 |
| **Held-out test** | **1** | **42** | **1.000** | **1.000** | **1.000** | **1.000** | **0.000** | **0.000** | 30 0 0 12 |
| External test | - | - | not run | | | | | | |

- Grouped cross-validation on train + validation: F1 **0.78 +/- 0.30**; the fold scores were 0.98, **0.36** and 1.00.
- Supplementary leave-one-recording-out (default hyperparameters): F1 between 0.978 and 1.000.
- **What this supports:** the training and evaluation method runs on real data without leakage, and on this one
  pass the features separate the windows.
- **What it does not support:** any claim that the model works on RTL-SDR captures, on other satellites or
  stations, or in general. The test set is one recording. The 0.36 fold shows the result swings with which
  recordings the model is trained on. The labels come from a known-carrier SNR rule, not independent ground truth.
  About 75% of the samples are clipped. The status stays **IMPLEMENTED BUT NOT VALIDATED**.
- There is no external test for the IQ model: no independent labelled real IQ-feature dataset exists in the repo.
  The waterfall model's external test needs the SatNOGS download (section 2).

## 7. Performance (NFR 5.1)

`evidence/performance/benchmark_linux_20260924.md` covers synthetic rtl_sdr cu8 captures at 1.024 Msps through
`pipeline.process_recording()`, with the default 120 s detection window and the IQ model loaded. Machine: Linux,
4 CPUs, 16 GB RAM.

| Capture | File | Time | Peak RAM |
|---|---|---|---|
| 10 s | 20 MB | 6.0 s | 0.7 GB |
| 60 s | 123 MB | 35 s | 3.5 GB |
| 120 s | 246 MB | 71 s | 6.7 GB |
| 600 s (whole-pass length) | 1.2 GB | 70 s | 6.7 GB |

The first measurement at `e8fd55f` found a defect: drawing the spectrogram PNG took about 7x the matrix size in
memory. A 60 s capture peaked at 8.5 GB, and the default 120 s window would have needed about 16 GB on every real
pass. This is fixed in `6bb4906`; detection is unchanged and only the image input is averaged. The new
`max_detection_seconds` setting (default unchanged at 120) lets an 8 GB computer use 60 s.

## 8. Known limitations

1. No live-hardware or Windows evidence yet (section 2).
2. ML not validated for real-world use (section 6). The waterfall model was not trained or tested here.
3. Detection with the rule and IQ-feature ML uses only the first `max_detection_seconds` of a recording (default 120 s,
   starting 30 s before AOS), which is the low-elevation start of the pass. The Doppler-corrected waterfall (and the
   waterfall model) covers the whole recording.
4. About 6.7 GB peak RAM per pass at the default settings; use `max_detection_seconds: 60` on 8 GB computers.
5. The simulated recorder writes 64 KB of random bytes, so demo runs prove the wiring, not detection.
6. Text/CSV input: headerless numeric matrices only; `--sample-rate` and `--center-freq` must be given; no real time axis.
7. GUI: maximise the window to see the Results pane. The GUI updates Tk variables from worker threads. This works in
   the smoke test, but Tk does not guarantee it.
8. REQ-12/13 not implemented (section 4). Binary classification only. Static visualisation only.
9. `main.py --no-ml` writes `ml_status: MODEL_NOT_AVAILABLE` in the summary JSON, where "skipped" would be accurate
   (cosmetic).
10. TLEs older than 7 days are only warned about. Without internet, cached TLEs are used.

## 9. Exact run commands

Windows (Command Prompt, repository root). On macOS/Linux use the `.sh` files and `.venv/bin/python`.

```bat
setup.bat                                   :: venv, packages, capture_config.json, IQ model, doctor
.venv\Scripts\python pipeline.py --doctor   :: environment check
check_dongle.bat                            :: 1-minute hardware test (FM station waterfall)
.venv\Scripts\python auto_capture.py --config capture_config.json --list-only
.venv\Scripts\python auto_capture.py --config capture_config.json --demo      :: 20 s simulated end-to-end run
run_station.bat                             :: dashboard (http://localhost:8050) + automatic capture, Ctrl+C stops
.venv\Scripts\python dashboard.py --config capture_config.json
sdr-doppler-prototype\launch_gui.bat        :: desktop GUI (maximise it)
.venv\Scripts\python sdr-doppler-prototype\src\history.py [--capture N]
.venv\Scripts\python -m pytest              :: automated tests
.venv\Scripts\python -m pytest -m hardware  :: hardware tests, dongle attached
.venv\Scripts\python tools\verify_simulated_pipeline.py
.venv\Scripts\python tools\benchmark_processing.py
.venv\Scripts\python tools\gui_smoke_test.py
cd sdr-doppler-prototype && ..\.venv\Scripts\python scripts\ml_evaluation_report.py
```

REQ-1 text input: `cd sdr-doppler-prototype` then
`python src/main.py --input <matrix.txt|.csv> --output data/results --sample-rate <Hz> --center-freq <Hz> --save-image`.

## 10. Evidence artefacts

| Path | Contents |
|---|---|
| `evidence/tests/TEST_RESULTS.md`, `baseline_e8fd55f_pytest.txt`, `final_ecbe278_pytest.txt` | Test counts, environment, raw logs |
| `evidence/pipeline/SIMULATED_PIPELINE.md`, `simulated_pipeline.json` | 21 stage checks, stored rows |
| `evidence/ml/ML_EVALUATION.md`, `ml_evaluation.json`, `split_assignment.csv` | ML metrics, per-row split |
| `evidence/performance/benchmark_linux_20260924.md/.json` | Time and memory |
| `evidence/gui/GUI_SMOKE.md`, `gui_smoke.json`, `01-09_*.png` | GUI smoke checks and screenshots |
| `evidence/live/<date>/` | **Does not exist yet: created by the live-hardware run** (LIVE_HARDWARE_TEST.md) |

## 11. Changes made in this handover round

On top of `main` `e8fd55f`, in separate commits:

- **Defects fixed:** spectrogram PNG memory blow-up (`sdr-doppler-prototype/src/spectrogram.py`); dashboard crashes on
  a corrupt database or a bad heartbeat, and zero totals on a pre-ML database (`dashboard.py`); missing
  `launch_gui.bat`; a storage test that pytest never collected.
- **Added:** `max_detection_seconds` setting (`auto_capture.py`, default unchanged); REQ-1, dashboard, image and
  report tests; `sdr-doppler-prototype/scripts/ml_evaluation_report.py`; `tools/` (simulated-pipeline verifier,
  benchmark, GUI smoke test); `LIVE_HARDWARE_TEST.md`; this file; `evidence/`.
- **Docs corrected:** root README (configuration keys, outputs, tools), QUICKSTART, iq-recorder README
  (`record_pass()` is implemented), sdr-doppler-prototype README and training README (no longer "no GUI" or
  "synthetic only"; input formats; REQ-1 limits).
- No architecture changes. No working feature removed, no test weakened or deleted.

## For the handover pack (V0.9 to V1.0)

Statements in the V0.9 pack and matrix that this code contradicts:

- "record_pass() remains a stub raising NotImplementedError", "no ... TLE/orbit propagation": pass prediction
  (`rtl_recorder/passes.py`) and `record_pass()` are implemented and tested in simulation. Real-pass evidence is pending.
- "127/127 tests": 354 passed / 0 failed at `ecbe278`, 3 hardware tests not run.
- "ML ... operates with synthetic data" / "train/validation/test separation not carried out": done on one real
  dataset with grouped splits (section 6). It is still not validated.
- REQ-8 "Not confirmed": selective retention is implemented and tested (section 1). Real-hardware use is pending.
- REQ-12/13 "Not confirmed": confirmed **not implemented** (section 4).
- REQ-1 "not yet verified": verified for headerless `.txt`/`.csv` matrices, with limits (sections 1 and 8).
- NFR 5.1-a/b "Not evidenced": measured (section 7).
- "LoRa chunk labelling ... label_lora_chunks.py": that file was removed. The workflow is
  `sdr-doppler-prototype/scripts/chunk_bin_to_dataset.py`.
- Internal checks now answerable from the repo: installation commands, config keys, CLI/GUI launch commands and output
  locations (README "Configuration" and "Outputs"). Still open: live-hardware evidence, Windows evidence, GUI manual
  verification, ML-scope provenance, owner confirmations, client acknowledgements.
