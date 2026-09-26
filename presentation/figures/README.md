# Presentation figures

Charts of the work so far, 1920x1080, ready for slides. Numbers come from real runs
(see `docs/ml_milestones.csv`). Every new training run and external test is now also
appended automatically to `sdr-doppler-prototype/models/ml_history.csv`.

| File | Slide message | Speaker note |
|---|---|---|
| 01_model_journey.png | The detector got better at every step | Blue is the honest number: a real recording from different hardware the model never trained on. 0.944 -> 0.998. |
| 02_three_detectors.png | From hand-set rules to a model that generalises | Rules found almost nothing on real data (F1 0.02). The waterfall model scores 0.96 on RSP-03 without ever seeing it. |
| 03_overnight_download.png | One night: 3,000 real passes | The SatNOGS hourly limit meant ~2/3 of the night was waiting; the script paused, resumed and kept the laptop awake by itself. |
| 04_learning_curve.png | More of the same data no longer helps | 10x the data gave +0.014 AUC and other algorithms were no better, so the next gains come from cleaner labels and our own passes. |
| 05_accuracy_by_group.png | Works across satellites, weakest on faint/low passes | ISS 0.95; METEOR-M2-3 lower but based on only 22 negative examples. Low passes near the horizon are hardest. |
| 06_confusion_matrices.png | What the mistakes look like | RSP-03: zero false alarms, 4 misses. The uncertain level keeps borderline misses for review instead of deleting them. |
| 07_label_review.png | Cleaning volunteer labels | 20 wrong labels fixed; 45 invisible positives dropped, not flipped, so the model never learns to delete weak real passes. AI-suggested, human-confirmed. |
| 08_waterfall_examples.png | What the model looks at | Two real signals, interference that fools simple rules, and pure noise. |
| 09_project_timeline.png | Project timeline | From the May prototype to an automatic station; next is our own dongle. |

Caveats to say out loud: axes on 01, 04 and 05 do not start at zero; the label-review
point on 01 is a trial before human confirmation; nothing is validated on our own dongle yet.
