#!/usr/bin/env python3
"""Build a real, labelled training set from the SatNOGS network.

SatNOGS (https://network.satnogs.org) is a worldwide network of volunteer
ground stations. Every observation publishes a waterfall image and is
vetted by people: status "good" = the satellite's signal is there,
"bad" = no signal. That is exactly our question, across hundreds of
satellites and stations - far more varied than a single recorded pass.

For each observation this script downloads the waterfall PNG, turns it
into the standard 128 x 128 waterfall (src/waterfall_png.py), computes the
waterfall features (src/features/waterfall_features.py) and appends a row
to the CSV. Rows are grouped by ground station (recording_id = station),
so the train/validation/test split in ml/train.py tests on stations the
model has never seen.

    # ~400 observations, mixed satellites, balanced good/bad (10-20 min)
    python scripts/fetch_satnogs_dataset.py --per-class 200

    # only some satellites (NORAD ids)
    python scripts/fetch_satnogs_dataset.py --norad 25544 57166 59051 --per-class 100

    # then train the waterfall model
    python train_model.py --feature-set waterfall \\
        --dataset data/training/satnogs_waterfall_features.csv --output models/waterfall_rf.joblib

It's resumable (already-downloaded observations are skipped), keeps only
the small 128 x 128 arrays (not the PNGs, unless --keep-png), and waits
between requests to be polite to the SatNOGS servers.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_features  # noqa: E402
from waterfall_png import WaterfallParseError, png_to_waterfall  # noqa: E402

API = "https://network.satnogs.org/api/observations/"
USER_AGENT = "rtl-sdr-capstone-dataset/1.0 (university capstone; polite, rate-limited)"
LABELS = {"good": 1, "bad": 0}
META_COLUMNS = ["satnogs_id", "norad_cat_id", "status", "transmitter_mode", "observation_frequency",
                "max_altitude", "station_name", "ground_station", "start", "waterfall_url"]
CSV_COLUMNS = ["capture_id", *WATERFALL_FEATURE_NAMES, "label", "is_synthetic", "recording_id", "source_file",
               *META_COLUMNS]


def http_get(url: str, *, retries: int = 4, timeout: float = 30.0):
    """(body bytes, headers). Backs off on 429/5xx."""
    delay = 5.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, image/png"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https hosts
                return resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = float(exc.headers.get("Retry-After") or delay)
                print(f"  server said {exc.code}, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                delay *= 2
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def next_link(headers, body_json):
    """DRF pagination: either a Link: <...>; rel="next" header or a 'next' field."""
    if isinstance(body_json, dict) and body_json.get("next"):
        return body_json["next"]
    link = headers.get("Link") if headers else None
    if link:
        m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        if m:
            return m.group(1)
    return None


def fetch_window(status: str, norad, start, end, page_delay: float, max_pages: int = 5):
    """Observations with this status that started inside [start, end).

    Follows the API's Link-header pagination for a few pages. A 400/404 on a
    later page means "no more pages" (the SatNOGS API rejects out-of-range
    pages), so it ends the window instead of crashing the whole download.
    """
    params = {"format": "json", "status": status,
              "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")}
    if norad is not None:
        params["norad_cat_id"] = norad
    url = API + "?" + urllib.parse.urlencode(params)
    for page in range(max_pages):
        try:
            body, headers = http_get(url)
        except urllib.error.HTTPError as exc:
            if exc.code in (400, 404) and page > 0:
                return
            raise
        data = json.loads(body)
        items = data.get("results", []) if isinstance(data, dict) else data
        yield from items
        url = next_link(headers, data)
        if not url or not items:
            return
        time.sleep(page_delay)


def iter_windows(days_back: int, hours: int = 12):
    """Consecutive time windows going back from now (newest first)."""
    from datetime import datetime, timedelta, timezone

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    stop = end - timedelta(days=days_back)
    while end > stop:
        start = end - timedelta(hours=hours)
        yield start, end
        end = start


def count_labels(csv_path: Path) -> dict:
    counts = {1: 0, 0: 0}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[int(row["label"])] += 1
    return counts


def existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["satnogs_id"] for row in csv.DictReader(f)}


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def save_parse_check(png_bytes: bytes, matrix: np.ndarray, out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(8, 6))
    ax[0].imshow(mpimg.imread(io.BytesIO(png_bytes), format="png"))
    ax[0].set_title("SatNOGS PNG")
    ax[1].imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax[1].set_title("parsed 128x128 (row 0 = start)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def process(obs: dict, args, arrays_dir: Path, check_dir: Path, checks_done: list) -> dict | None:
    url = obs.get("waterfall")
    if not url:
        return None
    png, _ = http_get(url)
    try:
        matrix = png_to_waterfall(np.asarray(_decode_png(png)))
    except WaterfallParseError as exc:
        print(f"  skip {obs['id']}: {exc}")
        return None
    np.save(arrays_dir / f"{obs['id']}.npy", matrix.astype(np.float16))
    if args.keep_png:
        (arrays_dir / f"{obs['id']}.png").write_bytes(png)
    if len(checks_done) < args.parse_checks:
        save_parse_check(png, matrix, check_dir / f"{obs['id']}_{obs['status']}.png",
                         f"obs {obs['id']} NORAD {obs.get('norad_cat_id')} {obs.get('transmitter_mode')} - {obs['status']}")
        checks_done.append(obs["id"])
    feats = waterfall_features(matrix)
    return {
        "capture_id": f"satnogs_{obs['id']}", **feats, "label": LABELS[obs["status"]], "is_synthetic": 0,
        "recording_id": f"station_{obs.get('ground_station')}", "source_file": url,
        "satnogs_id": obs["id"], "norad_cat_id": obs.get("norad_cat_id"), "status": obs["status"],
        "transmitter_mode": obs.get("transmitter_mode"), "observation_frequency": obs.get("observation_frequency"),
        "max_altitude": obs.get("max_altitude"), "station_name": obs.get("station_name"),
        "ground_station": obs.get("ground_station"), "start": obs.get("start"), "waterfall_url": url,
    }


def _decode_png(png: bytes) -> np.ndarray:
    import matplotlib.image as mpimg

    return mpimg.imread(io.BytesIO(png), format="png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download vetted SatNOGS observations as a labelled waterfall dataset.")
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--output", type=Path, default=root / "data" / "training" / "satnogs_waterfall_features.csv")
    p.add_argument("--arrays-dir", type=Path, default=root / "data" / "satnogs" / "waterfalls")
    p.add_argument("--per-class", type=int, default=200, help="Target number of good AND of bad observations")
    p.add_argument("--norad", type=int, nargs="*", default=None, help="Only these satellites (default: any)")
    p.add_argument("--max-per-station", type=int, default=15, help="Cap per station per class, for variety")
    p.add_argument("--per-window", type=int, default=8, help="Max observations per class per 12-hour window, for variety")
    p.add_argument("--days-back", type=int, default=45, help="How far back in time to look")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between image downloads")
    p.add_argument("--page-delay", type=float, default=1.5, help="Seconds between API pages")
    p.add_argument("--parse-checks", type=int, default=6, help="Save this many side-by-side parse-check images")
    p.add_argument("--keep-png", action="store_true")
    args = p.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.arrays_dir.mkdir(parents=True, exist_ok=True)
    check_dir = args.arrays_dir.parent / "parse_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    done = existing_ids(args.output)
    print(f"Writing {args.output} ({len(done)} observations already there)")

    checks_done: list = []
    targets = args.norad or [None]
    counts = count_labels(args.output)
    per_station = {}
    print(f"Target: {args.per_class} good + {args.per_class} bad (have {counts[1]} good, {counts[0]} bad)")
    for start, end in iter_windows(args.days_back):
        if all(counts[label] >= args.per_class for label in LABELS.values()):
            break
        # Alternate good/bad inside every window, so the set stays balanced even if interrupted.
        for status, label in LABELS.items():
            if counts[label] >= args.per_class:
                continue
            for norad in targets:
                try:
                    observations = list(fetch_window(status, norad, start, end, args.page_delay))
                except Exception as exc:  # network hiccup: skip this window, keep going
                    print(f"  window {start:%m-%d %H}h {status}: {exc}")
                    continue
                added = 0
                for obs in observations:
                    if counts[label] >= args.per_class or added >= args.per_window:
                        break
                    if str(obs.get("id")) in done or obs.get("status") != status:
                        continue
                    station = obs.get("ground_station")
                    if per_station.get((station, label), 0) >= args.max_per_station:
                        continue
                    try:
                        row = process(obs, args, args.arrays_dir, check_dir, checks_done)
                    except Exception as exc:  # one bad download mustn't stop the run
                        print(f"  skip {obs.get('id')}: {exc}")
                        continue
                    if row is None:
                        continue
                    append_row(args.output, row)
                    done.add(str(obs["id"]))
                    per_station[(station, label)] = per_station.get((station, label), 0) + 1
                    counts[label] += 1
                    added += 1
                    time.sleep(args.delay)
        print(f"  {start:%Y-%m-%d %H}h: {counts[1]} good / {counts[0]} bad so far", flush=True)
    if min(counts.values()) == 0:
        print("\nWARNING: one class has no observations - training needs both good and bad. "
              "Run again (it resumes) or raise --days-back.")
    print(f"\nDataset: {args.output}")
    print(f"Parse checks (eyeball these!): {check_dir}")
    print("Next: python train_model.py --feature-set waterfall "
          f"--dataset {args.output} --output models/waterfall_rf.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
