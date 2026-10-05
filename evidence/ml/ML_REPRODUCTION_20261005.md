# ML results reproduced from committed data (5 Oct 2026)

Commit `040c140`, clean checkout. Commands: `python setup_station.py --skip-doctor`, then
`python train_client_data.py --skip-build` (no promotion). Every figure below came out of these runs on committed
CSVs; nothing was downloaded. Environment: Python 3.13.16, Linux-6.18.44 x86_64 (container, 2 CPUs, 8 GB RAM); numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, matplotlib 3.11.2, pytest 9.1.1.

| Claim (final presentation) | Reproduced result | Source data |
|---|---|---|
| Waterfall model trained on 3,000 SatNOGS passes, 474 satellites | 3,000 rows (1,500 signal / 1,500 no signal), 474 NORAD ids, 340 station names | `data/training/satnogs_waterfall_features.csv` |
| Held-out 567 passes: ROC-AUC 0.929 | Test set 567 passes: ROC-AUC 0.929, F1 0.861 (threshold 0.45); grouped CV F1 0.830 +/- 0.037 | same |
| Client RSP-03 (never seen): ROC-AUC 0.998, 0 false alarms | 111 windows: ROC-AUC 0.998, F1 0.957, TN 62 FP 0 FN 4 TP 45 | `rsp03_waterfall_features.csv` |
| SARAL pass: 12 / 12 signal windows | SARAL pass 2: 12 of 12 signal windows detected, 2 of 2 noise windows correct | `client_passes_waterfall_external.csv` (SARAL rows) |
| IQ model on held-out client passes 0.927 -> 0.939 | ROC-AUC 0.927 (RSP-03 model) -> 0.939 (RSP-03 + client snapshots). F1 at the chosen thresholds fell, 0.882 -> 0.842 | `client_passes_iq_external.csv` (621 rows) |

Limits: all of these are offline tests on recordings from other stations (SatNOGS, CAMRAS Dwingeloo dish, client
recordings). None is a capture from the team's own RTL-SDR. On the full held-out client waterfall set (16 rows,
including the NORAD-16 marginal pass) the SatNOGS-only model raised 2 false alarms out of 4 noise windows.
`setup` trains the IQ model on RSP-03 only; the client-retrained IQ model is created by
`train_client_data --skip-build` and used only after `--promote iq`.
