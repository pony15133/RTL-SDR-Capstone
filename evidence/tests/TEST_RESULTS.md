# Automated test results

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
