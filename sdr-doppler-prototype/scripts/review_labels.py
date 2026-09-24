#!/usr/bin/env python3
"""Find and fix wrong labels in the SatNOGS training set.

SatNOGS labels come from volunteers, so some are wrong - and for our
question ("is a satellite signal visible in this waterfall?") some are
wrong on purpose: an observation is vetted "bad" when the *expected*
transmitter is missing, even if another satellite's trace is plainly
visible, and "good" when packets decoded even if nothing shows in the
waterfall.

Step 1 - find suspects. Every observation is scored by a model that never
saw it (grouped out-of-fold), and the ones where model and label disagree
most are drawn as numbered contact sheets:

    python scripts/review_labels.py
    -> data/satnogs/review/sheet_01.png, sheet_02.png, ...
    -> data/satnogs/review/label_review.csv   (one line per suspect)

Step 2 - look at each sheet and fill in the `your_label` column of
label_review.csv: 1 = signal visible, 0 = no signal, x = unusable (drop).
Leave it empty to keep the current label.

Step 3 - apply (keeps the original label in label_original):

    python scripts/review_labels.py --apply data/satnogs/review/label_review.csv

Then retrain. Re-running step 1 afterwards finds the next batch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.waterfall_features import WATERFALL_FEATURE_NAMES  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "training" / "satnogs_waterfall_features.csv"
DEFAULT_ARRAYS = ROOT / "data" / "satnogs" / "waterfalls"
PER_SHEET = 24


def out_of_fold_scores(df: pd.DataFrame, seeds=(0, 1, 2)) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold

    X = df[list(WATERFALL_FEATURE_NAMES)].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    groups = df["recording_id"].astype(str).to_numpy()
    folds = max(2, min(5, min(len(np.unique(groups[y == c])) for c in (0, 1))))
    total = np.zeros(len(y))
    for seed in seeds:
        for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups):
            clf = RandomForestClassifier(300, max_depth=8, min_samples_leaf=2, random_state=seed, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            total[te] += clf.predict_proba(X[te])[:, list(clf.classes_).index(1)]
    return total / len(seeds)


def save_sheets(sub: pd.DataFrame, arrays_dir: Path, out_dir: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for page, start in enumerate(range(0, len(sub), PER_SHEET), 1):
        chunk = sub.iloc[start:start + PER_SHEET]
        fig, axes = plt.subplots(3, 8, figsize=(20, 10))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, r) in zip(axes.ravel(), chunk.iterrows()):
            m = np.load(arrays_dir / f"{r['satnogs_id']}.npy").astype(np.float32)
            lo, hi = np.percentile(m, [5, 99.7])
            ax.imshow(m, aspect="auto", origin="lower", cmap="viridis", vmin=lo, vmax=hi)
            ax.set_title(f"#{r['review_no']}  id {r['satnogs_id']}\nlabel {int(r['label'])}  model {r['model_score']:.2f}",
                         fontsize=8, color="crimson" if abs(r["label"] - r["model_score"]) > 0.7 else "black")
        fig.suptitle("Label review - time runs upward, satellite should be near the centre. "
                     "Fill `your_label` in label_review.csv (1 signal / 0 none / x drop)", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"sheet_{page:02d}.png"
        fig.savefig(path, dpi=70)
        plt.close(fig)
        paths.append(path)
    return paths


def find_suspects(dataset: Path, arrays_dir: Path, out_dir: Path, top: int) -> Path:
    df = pd.read_csv(dataset)
    if "label_reviewed" in df.columns:
        reviewed = df["label_reviewed"].fillna(0).astype(int).astype(bool)
    else:
        reviewed = pd.Series(False, index=df.index)
    df["model_score"] = out_of_fold_scores(df)
    df["disagreement"] = (df["label"] - df["model_score"]).abs()
    has_array = df["satnogs_id"].map(lambda s: (arrays_dir / f"{s}.npy").exists())
    sub = df[has_array & ~reviewed].sort_values("disagreement", ascending=False).head(top).copy()
    sub["review_no"] = range(1, len(sub) + 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets = save_sheets(sub, arrays_dir, out_dir)
    review = sub[["review_no", "satnogs_id", "norad_cat_id", "transmitter_mode", "label", "model_score"]].copy()
    review["model_score"] = review["model_score"].round(3)
    review["your_label"] = ""
    path = out_dir / "label_review.csv"
    review.to_csv(path, index=False)
    print(f"{len(sub)} suspects (biggest model/label disagreement first) on {len(sheets)} sheet(s) in {out_dir}")
    print(f"Fill in `your_label` in {path}, then run:  python scripts/review_labels.py --apply {path}")
    return path


def apply_review(dataset: Path, review_csv: Path) -> dict:
    df = pd.read_csv(dataset)
    review = pd.read_csv(review_csv, dtype={"your_label": str})
    for col, default in (("label_original", np.nan), ("label_reviewed", 0)):
        if col not in df.columns:
            df[col] = default
    if "label_source" not in df.columns:
        df["label_source"] = ""
    df["label_source"] = df["label_source"].astype(object)
    index = {str(s): i for i, s in enumerate(df["satnogs_id"].astype(str))}
    changed = kept = dropped = 0
    drop = []
    for _, r in review.iterrows():
        answer = str(r.get("your_label") if pd.notna(r.get("your_label")) else "").strip().lower()
        i = index.get(str(r["satnogs_id"]))
        if i is None or answer == "":
            continue
        if answer == "x":
            drop.append(i)
            dropped += 1
            continue
        if answer not in {"0", "1"}:
            raise SystemExit(f"FAILED: review #{r['review_no']}: your_label must be 1, 0, x or empty, got {answer!r}")
        new = int(answer)
        if pd.isna(df.at[i, "label_original"]):
            df.at[i, "label_original"] = df.at[i, "label"]
        if int(df.at[i, "label"]) != new:
            changed += 1
            df.at[i, "label"] = new
            df.at[i, "label_source"] = "manual"
        else:
            kept += 1
        df.at[i, "label_reviewed"] = 1
    df = df.drop(index=drop).reset_index(drop=True)
    df.to_csv(dataset, index=False)
    return {"changed": changed, "confirmed": kept, "dropped": dropped}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Review suspicious SatNOGS labels.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--arrays-dir", type=Path, default=DEFAULT_ARRAYS)
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <arrays-dir>/../review")
    p.add_argument("--top", type=int, default=72, help="How many suspects to show")
    p.add_argument("--apply", type=Path, default=None, help="Apply a filled-in label_review.csv")
    args = p.parse_args(argv)
    if args.apply:
        r = apply_review(args.dataset, args.apply)
        print(f"Applied: {r['changed']} relabelled, {r['confirmed']} confirmed, {r['dropped']} dropped")
        return 0
    find_suspects(args.dataset, args.arrays_dir, args.out_dir or args.arrays_dir.parent / "review", args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
