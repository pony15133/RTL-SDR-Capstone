# Training the models on real data

The station's main detector is the **waterfall model**. It looks at the Doppler-corrected 128 × 128 waterfall of a pass and answers one question: is a satellite signal visible? It learns from three sources of real data. Each one is worth more than the one before.

| Source | What it is | Size | How |
|---|---|---|---|
| SatNOGS | Vetted observations from ~100+ volunteer stations worldwide | hundreds to thousands | `setup --fetch-satnogs [--big-data]` |
| Label review | The same SatNOGS data, with wrong labels fixed by you | a few dozen fixes | `scripts/review_labels.py` |
| Our own passes | Captures from our antenna, dongle and location | grows with every pass | `scripts/label_station_captures.py` |

The CAMRAS RSP-03 pass (`data/training/rsp03_waterfall_features.csv`) is never used for training. It stays as an **external test**: a different dish, satellite and data source. Setup prints its score after every training run.

All commands are run from the repo root, with `.venv\Scripts\python` (Windows) or `.venv/bin/python` (macOS/Linux) as `python`.

## 1. More SatNOGS data

```
setup.bat --fetch-satnogs --big-data          (./setup.sh on macOS/Linux)
```

This downloads 1,500 "signal" and 1,500 "no signal" observations. It samples time windows at random over the last 120 days, so the data isn't dominated by one busy day. It caps how many come from any one station, and it also downloads extra METEOR-M2-3, METEOR-M2-4 and ISS observations (the satellites we record). It takes 1–2 hours. **It is resumable:** press Ctrl+C at any time, and run the same command again to continue. When it finishes, it retrains and prints the test and RSP-03 scores.

Only the 32 kB arrays are kept, not the PNGs: about 100 MB for 3,000 observations.

Why more data: on the first 400 observations, a learning curve (train on 25/50/75/100 % of the stations, test on unseen stations) gave ROC-AUC 0.891 → 0.895 → 0.910 → 0.925. It was still climbing.

## 2. Better labels (waterfall vetting)

SatNOGS vets every observation twice. `status` (good/bad) says whether data was received. `waterfall_status` (with-signal/without-signal) says whether the signal can be seen in the waterfall, which is exactly the model's question. The downloader now labels by `waterfall_status` and only falls back to `status` when the waterfall wasn't vetted. The `label_source` column records which one was used.

To relabel data downloaded before this change (metadata only, no images, about 10 minutes for 400):

```
setup.bat --refresh-labels
```

## 3. Fix the labels that are still wrong

```
python sdr-doppler-prototype/scripts/review_labels.py
```

This scores every observation with a model that never saw it, then draws the 72 biggest disagreements as numbered contact sheets in `sdr-doppler-prototype/data/satnogs/review/`. Open each `sheet_XX.png` and, in `label_review.csv`, fill in `your_label`:

- `1`: a satellite trace is visible
- `0`: no signal
- `x`: unusable (dropped)
- empty: keep the current label

Then apply the changes and retrain:

```
python sdr-doppler-prototype/scripts/review_labels.py --apply sdr-doppler-prototype/data/satnogs/review/label_review.csv
setup.bat
```

Reviewed rows keep their original label in `label_original`, and they aren't shown again. Run the review again to get the next batch. On the first 400 observations, about half of the top suspects were clearly mislabelled for our question. For example, some were vetted "bad" even though another satellite's trace is plainly visible, and some were vetted "good" because packets decoded while nothing shows in the waterfall.

## 4. Add our own passes (the most valuable data)

Every processed pass now saves its waterfall matrix (`*_waterfall.npy`) next to the PNG, even when the IQ file is archived or deleted. Once the station has recorded some passes:

```
python sdr-doppler-prototype/scripts/label_station_captures.py
```

It opens each unlabelled pass's waterfall and asks for `1` (signal), `0` (none), `s` (skip) or `q` (quit). **Label by looking at the waterfall, not by what the model said.** Otherwise the model just learns its own mistakes.

The labels go into `data/training/station_waterfall_features.csv`. After that, `setup` trains on SatNOGS and our own passes together. Each pass is its own group, so a pass never appears in both train and test.

## What changed in the model

- **Two new features, `wf_vert_coherence` and `wf_horiz_coherence`.** They measure how strongly neighbouring pixels agree along time and along frequency. A real trace spans several pixels, and noise doesn't. Compared on the same data (10 repeats of grouped 5-fold CV):

  | | Grouped CV ROC-AUC | RSP-03 external ROC-AUC |
  |---|---|---|
  | Before | 0.912 | 0.944 |
  | After | 0.920 | 0.977 |

  Old datasets are upgraded automatically from the saved arrays: setup runs `fetch_satnogs_dataset.py --rebuild-features`.
- **Tuned decision threshold.** Once there are at least 100 rows from 10 or more stations, the threshold is chosen from grouped out-of-fold F1 on train + validation, never the test set. It is saved in the model's metadata and used everywhere: the detector, `evaluate_model.py` and retention. You can override it with `--threshold 0.5`.
- **Retention follows the model.** `keep_threshold: null` (the new default) uses the model's tuned threshold. A number overrides it.
- **Several datasets at once:** `train_model.py --dataset a.csv b.csv`.

## Honest limits

- SatNOGS is mostly packet satellites (FSK/GMSK bursts). Our targets are continuous: METEOR LRPT and CW-like carriers. That is why `--preset station` and our own passes matter.
- The held-out test set for 400 observations is only 60 observations, so its F1 moves by ±0.05 between random splits. Quote cross-validation and RSP-03 alongside it, and the test set gets more reliable as the data grows.
- Features were tried and rejected when they didn't help. For example, off-centre trace tracking and pixel-distribution shape gave no CV gain, so they were left out.
