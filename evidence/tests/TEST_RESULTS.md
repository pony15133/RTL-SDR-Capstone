# Automated test results

## Final reconciled code (5 Oct 2026)

Code commit `040c140` on `release/final-reconciled`: `775a92f` (final `main`) with the handover-verification
line `7e21078` merged back in (`2b6ce39`), the pre-flight notes (`5df7298`) and the SatNOGS-dataset fix (`040c140`).
Clean checkout, no trained models or outputs present when the suite ran.

| | Final `main` before reconciliation | Reconciled final code |
|---|---|---|
| Commit | `775a92fd3795eecbe8d9eed728f2a28ad8e4c1c9` | `040c140` (see `final_040c140_pytest.txt`) |
| Passed | 411 | **433** |
| Failed / errors / skipped | 0 / 0 / 0 | **0 / 0 / 0** |
| Deselected (`hardware` marker, need a dongle) | 3 | 3 |

Command: `python -m pytest -p no:cacheprovider -q -rs` from the repository root. The 22 extra tests are the
handover round's REQ-1 text input (6), ML evaluation report (3), dashboard robustness (4), spectrogram-image
memory (4), detection-window setting (2) and storage check (1), plus 2 new tests for the SatNOGS-dataset fix.
The 3 hardware tests were **not run** (no RTL-SDR attached). Windows was not re-tested on this commit; the
latest Windows result (353 passed, 1 skipped) is from the 25 Sep pre-reconciliation line.

Environment: Python 3.13.16, Linux-6.18.44 x86_64 (container, 2 CPUs, 8 GB RAM); numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, matplotlib 3.11.2, pytest 9.1.1.

Also run on this commit: `tools/verify_simulated_pipeline.py` (all checks passed, `evidence/pipeline/`),
`tools/gui_smoke_test.py` under Xvfb with Python 3.12 / Tk 8.6 (6/6, `evidence/gui/`),
`tools/benchmark_processing.py` (`evidence/performance/benchmark_linux_20261005.md`) and the ML reproduction in
`evidence/ml/ML_REPRODUCTION_20261005.md`.
`evidence/pipeline/` and `evidence/gui/` were regenerated on this commit; the 24 Sep (`ecbe278`) versions are in git history.

---

## Handover round (24 Sep 2026, historical)

| | Baseline (teammates' `main`) | Final (this handover) |
|---|---|---|
| Commit | `e8fd55ffd796ec683df5cc554dbd98909bf035e5` | `ecbe2781682ad0fa4466d63b6216b0d86ca22f12` |
| Passed | 334 | 354 |
| Failed | 0 | 0 |
| Errors | 0 | 0 |
| Skipped | 0 | 0 |
| Deselected (`hardware` marker, need a dongle) | 3 | 3 |
| Duration | 127.5 s | 132.1 s |

Command, from the repository root in a clean checkout (no trained models, no outputs):
`python -m pytest -p no:cacheprovider -q -rs`. The `pytest.ini` default `-m "not hardware"` deselects
`iq-recorder/tests/hardware/test_hardware_kiss92.py` (3 tests). They were **not run**, because no RTL-SDR is attached.

Final run by project: iq-recorder 124, root tests/ 37, sdr-doppler-prototype 193.

The 20 extra tests come from this handover round: REQ-1 text input (6), ML evaluation report (3),
dashboard robustness (4), spectrogram image memory (4), detection-window setting (2), and the previously
uncollected storage check (1).

Environment: Python 3.11.15, Linux-6.18.44-fc-v37-x86_64-with-glibc2.39; numpy 2.4.6, scipy 1.17.1,
scikit-learn 1.9.1, pandas 3.0.6, matplotlib 3.11.2, pytest 9.1.1.
Linux container, 4 CPUs, 16 GB RAM. **Windows was not tested.**

Raw logs: `baseline_e8fd55f_pytest.txt`, `final_ecbe278_pytest.txt`.
