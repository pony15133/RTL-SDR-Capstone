# GUI start-up smoke test

Generated 2026-09-24T09:07:28+00:00 on commit `ecbe2781682ad0fa4466d63b6216b0d86ca22f12` by `tools/gui_smoke_test.py` - Python 3.12.3, Tk 8.6, Linux-6.18.44-fc-v37-x86_64-with-glibc2.39.

**Result: ALL CHECKS PASSED**

Automated: the real Tk window, every tab, and the Run Demo Capture / Show History code paths against a temporary database. Not a manual usability check.

| Check | Result | Detail |
|---|---|---|
| main window opens | PASS | SDR Doppler Prototype |
| all function tabs built | PASS | Visualise IQ, Run Detection, Satellite Passes, Capture History, Train Model, Convert SigMF (.sigmf-data/.sigmf-meta), Open Results Folder |
| Run Demo Capture completes | PASS | Running demo capture complete |
| demo capture stored in the GUI's database | PASS | [('METEOR-M2-3', 'SUCCESS', 'DETECTED')] |
| results pop-up window opened | PASS |  |
| Show History lists the demo capture | PASS | Loading capture history complete |

Screenshots:

- 01_tab_visualise_iq.png
- 02_tab_run_detection.png
- 03_tab_satellite_passes.png
- 04_tab_capture_history.png
- 05_tab_train_model.png
- 06_tab_convert_sigmf.png
- 07_tab_open_results_folder.png
- 08_after_demo_capture.png
- 09_capture_history.png
