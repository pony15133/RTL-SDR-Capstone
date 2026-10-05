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

This first downloads up to 150 "signal" and 150 "no signal" observations of the satellites we record (METEOR-M2-3, METEOR-M2-4, ISS). Then it tops up with other satellites until there are 1,500 of each class. It samples time windows at random over the last 120 days, so the data isn't dominated by one busy day, and it caps how many observations come from any one station. When it finishes, it retrains and prints the test and RSP-03 scores.

**SatNOGS allows only a limited number of API requests per hour.** When that runs out, the downloader prints something like "asks us to pause 58 min", shows the time it will continue, counts down every 5 minutes and then carries on by itself. Nothing is frozen, and everything downloaded so far is saved. You can press Ctrl+C at any time and run the same command later to continue, or add `--no-wait` to stop at the first pause. A big run takes a few hours, so it's easiest to leave it running overnight.

### Overnight

```
train_overnight.bat          (./train_overnight.sh on macOS/Linux)
```

This runs the same big download and then retrains. On the way it:

- keeps the computer from sleeping until it's done, without changing any settings, and lets it sleep normally again afterwards
- waits through the SatNOGS pauses by itself
- writes everything to `logs\overnight_<date>.txt`, so in the morning you can send that file to see the results

Keep the laptop plugged in with the lid open, because closing the lid can still put it to sleep. If it's stopped for any reason, run it again and it continues from where it left off.

**About the limit.** It's SatNOGS's rule for a free service run by volunteers, so we stay within it rather than work around it. Using VPNs or several IP addresses to dodge it would likely get the station blocked. What the downloader does instead:

- it takes up to 25 observations from every API page, which makes about 3× fewer requests
- image downloads come from a separate file host and don't count against the limit
- optionally, you can use your own free SatNOGS account: log in at network.satnogs.org, copy the API key from your profile, and run `setx SATNOGS_API_TOKEN <key>` once, then open a new terminal. The downloader then identifies itself with your account, and only to the SatNOGS API. SatNOGS doesn't document whether accounts get a bigger allowance, so treat it as a courtesy rather than a guaranteed speed-up.

Only the 32 kB arrays are kept, not the PNGs: about 100 MB for 3,000 observations.

Why more data: on the first 400 observations, a learning curve (train on 25/50/75/100 % of the stations, test on unseen stations) gave ROC-AUC 0.891 → 0.895 → 0.910 → 0.925. It was still climbing.

## 2. Better labels (waterfall vetting)

SatNOGS vets every observation twice. `status` (good/bad) says whether data was received. `waterfall_status` (with-signal/without-signal) says whether the signal can be seen in the waterfall, which is exactly the model's question. The downloader now labels by `waterfall_status` and only falls back to `status` when the waterfall wasn't vetted. The `label_source` column records which one was used.

New downloads get the waterfall vetting automatically, at no extra cost. You can also relabel data downloaded before this change with `setup.bat --refresh-labels`. It's usually not worth it, though: it needs one API request per observation, so it keeps running into the hourly limit, and in the first 50 of our 400 observations, none changed label. The label review in step 3 catches more. If you do run it, it saves progress every 25 observations and skips finished ones when you run it again.

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

Each label (from this script or the web Review page) goes into two files:

- `data/training/station_waterfall_features.csv` for the waterfall model (trains together with SatNOGS)
- `data/training/station_iq_features.csv` for the IQ model (trains together with RSP-03)

The IQ features are saved on the capture row when the pass is processed, so the IQ row is added even if the raw IQ file was archived or deleted. Older captures without saved features are recomputed from the IQ file if it is still there. After that, `setup` trains both models on the public data and our own passes together. Each pass is its own group, so a pass never appears in both train and test.

Note: the RSP-03 rows are 1-second windows, while station rows describe the first 120 s of a whole pass, which is also what the IQ model scores at runtime. Station rows are therefore the closest match to real use, and they matter more as they grow.

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

## Record of every run

Every `train_model.py` run and every `scripts/evaluate_model.py` run adds one row to `sdr-doppler-prototype/models/ml_history.csv` (date, data used, sizes, test and external scores, git commit, notes). Nothing is overwritten, so it is the project's lab notebook. Add `--notes "what changed"` to say why a run happened. The milestones before this log existed are in `docs/ml_milestones.csv`, and the presentation charts are in `presentation/figures/`.

## The client's labelled passes (client_pass_recordings)

The client recorded four real passes as ~1-second spectrogram snapshots (250 x 1024, dB, every ~1.6 s), sorted into folders by pass and elevation.

Folder `E:\CAPSTONE\CAP2\client_pass_recordings` (outside the repo, ~20 GB, not committed). Each pass folder is named `<SATELLITE>_<passN>_<client's verdict>`; the sub-folders keep the client's elevation names (`notinsky`, `justrisen`, `max35`, `17to5`, ...).

| Pass folder | Client's original name | Frequency | Used for |
|---|---|---|---|
| SARAL_pass1_detected | DetectedSatellite(SARAL) | 465.988 MHz | training |
| LILACSAT-2_not_detected | Nondetection(LILACSAT2) | 437.200 MHz | training (all noise: in the sky, nothing received) |
| SARAL_pass2_detected | DetectedSatellite2(SARAL) | 465.988 MHz | held-out test (different day) |
| NORAD-16_marginal | MarginalDetection(NORAD-16) | 465.988 MHz | held-out test (hard case) |

```
train_client_data.bat                 (finds ..\..\client_pass_recordings next to the CODE folder, or add --root <folder>)
```

Without the raw recordings (for example on a fresh clone), add `--skip-build` to use the committed `client_passes_*` CSVs instead.

It builds the datasets (`scripts/build_client_spectrogram_set.py`), scores the current models on the held-out passes, trains `iq_rf_client` and `waterfall_rf_client` with the client data added, and scores them again. Nothing is replaced until you run `train_client_data.bat --promote iq` (or `waterfall`, or `both`).

How the labels are made: fixed local spurs (and the DC bin) are removed using the part of the pass where the satellite is below the horizon; the Doppler track is found from the strongest peaks within +/-15 kHz and fitted as a decreasing curve; a snapshot is **signal** when it is in the sky and >= 6 dB on the track, **noise** when the satellite is below the horizon (or the pass is a non-detection), and **dropped** when it is in the sky but faint. **Check the previews** in `sdr-doppler-prototype/data/client_passes/preview/` - one picture per pass with the track and labels drawn on it.
