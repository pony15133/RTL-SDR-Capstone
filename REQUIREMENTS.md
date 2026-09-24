# Requirements traceability

All 34 requirements are implemented and covered by automated tests; 7 still need a real pass recorded with our own dongle before they can be called finished.

FR = Capstone 1 SRS functional scope · CR = later client and supervisor requests (dated) · NFR = non-functional requirements. Test files are under `tests/`, `iq-recorder/tests/` and `sdr-doppler-prototype/tests/`; manual checks are in [TESTING.md](TESTING.md).

| ID | Requirement | Where in the code | Evidence (tests) | Owner (M2 plan) | Status |
| --- | --- | --- | --- | --- | --- |
| FR1 | SDR signal capture from the RTL-SDR | iq-recorder/rtl_recorder/recorder.py (record_pass, start/stop, state machine) | iq-recorder/tests; TESTING.md B, C | Khoa | Done – needs real-pass check |
| FR2 | Validate captured data before processing | recorder.py (file size vs expected, sidecar JSON); check_dongle | iq-recorder/tests (validation); TESTING.md B1–B3 | Da | Done – needs real-pass check |
| FR3 | Spectrogram generation from raw IQ (FFT) | src/spectrogram.py, src/iq_io.py, src/visualize.py | test_iq_io_visualize.py | Tuan Anh | Done |
| FR4 | Feature extraction (peaks, frequency-time curve, waterfall features) | src/features/extractor.py, src/features/waterfall_features.py | test_ml_training.py, test_satnogs_waterfall.py | Tuan Anh | Done |
| FR5 | Signal filtering and Doppler correction | src/detect.py (median smoothing), src/doppler.py | test_satnogs_waterfall.py (simulated ISS pass straightened) | Tuan Anh | Done |
| FR6 | Doppler-based detection (rule detector) | src/detect.py | test_pipeline.py | Tuan Anh | Done |
| FR7 | Classify satellite / uncertain / no satellite with a confidence score | src/detection/, src/ml/, src/retention.py (three-level verdict) | test_real_data_training.py, test_pipeline.py; RSP-03 external test ROC-AUC 0.98 | Tuan Anh | Done – needs real-pass check |
| FR8 | Store session, detection and SDR metadata | src/database.py (capture_results) | test_pipeline.py, test_auto_capture.py | Andre | Done |
| FR9 | Selective raw-IQ retention (keep / review / archive / delete) | src/retention.py, src/review.py | test_pipeline.py, test_review.py | Andre | Done |
| FR10 | Keep spectrogram / waterfall for every capture | pipeline.py (PNG + 128×128 .npy per pass) | test_auto_capture.py | Andre | Done |
| FR11 | Runtime configuration (frequency, gain, sample rate, station) | capture_config.json; web Settings page | test_auto_capture.py, test_dashboard.py (settings validation) | Khoa | Done |
| FR12 | External orbit data (TLE / SatNOGS) | rtl_recorder/passes.py (CelesTrak, cached); scripts/fetch_satnogs_dataset.py | test_passes.py (Vallado SGP4 case), test_real_data_training.py | Khoa | Done |
| FR13 | Diagnostic outputs and plots | src/visualize.py, src/history.py, web Health page, logs/station.log | test_history_gui.py, test_dashboard.py | Da | Done |
| CR1 | Pass-based automatic scheduling: start at AOS, stop at LOS (client) | auto_capture.py, passes.py, record_pass() | test_auto_capture.py, test_passes.py | Khoa | Done – needs real-pass check |
| CR2 | Database updated on every recorder run, including failures (client, 24 Aug) | pipeline.py (log_failed_recording, recording metadata) | test_pipeline.py | Andre | Done |
| CR3 | Visualisation separate from ML (client, 24 Aug) | src/visualize.py; GUI Visualise IQ tab | test_iq_io_visualize.py | Tuan Anh | Done |
| CR4 | Cross-platform, no hard-coded rtl_sdr.exe (client, 24 Aug) | rtl_recorder/utils.py (find_executable), doctor.py | test_cross_platform.py; Windows + Linux runs | Khoa | Done |
| CR5 | Proper train / validation / test method (supervisor, 9 Sep) | src/ml/train.py (grouped split, tuned on validation, test scored once) | test_ml_training.py | Tuan Anh | Done |
| CR6 | FM positive-control test of the recorder (client, 27 Jul) | check_dongle.bat / .sh (Kiss92, 92.0 MHz) | TESTING.md B | Da | Done – needs real-pass check |
| CR7 | Uncertain level so a false negative never deletes real data (Khoa; M2 report) | src/retention.py, src/review.py, web Review page | test_pipeline.py, test_review.py | Tuan Anh | Done |
| CR8 | Position histories and status monitoring (project title) | pass_positions, status_log tables; history.py; web Passes / Health | test_auto_capture.py, test_dashboard.py | Andre | Done |
| CR9 | Live status web interface | dashboard.py, web/ (Overview, Passes, Captures, Review, Images, Settings, Health) | test_dashboard.py; browser screenshots | Da | Done |
| CR10 | Decode METEOR images from kept recordings | src/decode.py (SatDump), auto_capture DecodeWorker | test_decode.py, test_station_guard.py | Tuan Anh | Done – needs real-pass check |
| CR11 | Safe long unattended runs | station_guard.py (disk guard, retries, keep awake, daily log); run_station restart loop | test_station_guard.py | Khoa | Done |
| NFR1 | Performance on standard consumer hardware | Streaming readers; no GPU; background decoding | Full chain demo in 20 s (TESTING.md E) | Da | Done |
| NFR2 | Reliability despite SDR instability | Retries, loop restart, failed runs logged, row saved before file action | test_station_guard.py, test_pipeline.py | Khoa | Done |
| NFR3 | Maintainability (modular, easy to extend) | Separate recorder / processing / ML / storage / web modules | 390 automated tests | Da | Done |
| NFR4 | Scalability (new satellites and detectors without redesign) | Satellites in config; detectors behind one interface | test_auto_capture.py | Tuan Anh | Done |
| NFR5 | Storage efficiency | Retention policies; pruning of rejected recordings when disk is low | test_pipeline.py, test_station_guard.py | Andre | Done |
| NFR6 | Compatibility with RTL-SDR and Python scientific libraries | rtl_sdr tools; NumPy / SciPy / scikit-learn | pipeline.py --doctor | Khoa | Done |
| NFR7 | Usability for researchers and students | Web interface, QUICKSTART.md, one-click scripts | TESTING.md J–K | Da | Done |
| NFR8 | Configurability through configuration files | capture_config.json + Settings page (validated, with backup) | test_dashboard.py | Khoa | Done |
| NFR9 | Data integrity and traceability | Row written before any file move; atomic config and heartbeat writes; sidecars move with IQ | test_pipeline.py, test_review.py | Andre | Done |
| NFR10 | Portability across desktop operating systems | setup / run scripts for Windows and macOS / Linux | Tested on Windows + Linux; macOS not yet run | Khoa | Done – needs real-pass check |

Owner = the person assigned that area in the Milestone 2 plan (Khoa: capture, Tuan Anh: ML, Andre: storage, Da: testing). Confirm with the team who implemented each item before using this table for attribution or the handover form.
