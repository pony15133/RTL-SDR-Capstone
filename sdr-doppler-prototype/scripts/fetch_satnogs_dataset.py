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


def iter_observations(status: str, norad: int | None, page_delay: float):
    params = {"format": "json", "status": status}
    if norad is not None:
        params["norad_cat_id"] = norad
    url = API + "?" + urllib.parse.urlencode(params)
    page = 1
    seen = set()
    while url:
        body, headers = http_get(url)
        data = json.loads(body)
        items = data.get("results", []) if isinstance(data, dict) else data
        fresh = [o for o in items if o.get("id") not in seen]
        if not fresh:
            return
        for obs in fresh:
            seen.add(obs.get("id"))
            yield obs
        nxt = next_link(headers, data)
        if nxt is None:  # no Link header: fall back to ?page=N
            page += 1
            nxt = API + "?" + urllib.parse.urlencode({**params, "page": page})
        url = nxt
        time.sleep(page_delay)


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
    for status, label in LABELS.items():
        per_target = max(1, args.per_class // len(targets))
        for norad in targets:
            got, per_station = 0, {}
            already = 0
            print(f"\n== status={status} (label {label}) NORAD={norad or 'any'}: target {per_target}")
            for obs in iter_observations(status, norad, args.page_delay):
                if got >= per_target:
                    break
                if str(obs["id"]) in done:
                    already += 1
                    got += 1
                    continue
                station = obs.get("ground_station")
                if per_station.get(station, 0) >= args.max_per_station:
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
                per_station[station] = per_station.get(station, 0) + 1
                got += 1
                if got % 10 == 0:
                    print(f"  {got}/{per_target} (station variety: {len(per_station)})", flush=True)
                time.sleep(args.delay)
            print(f"  done: {got} ({already} were already in the CSV)")
    print(f"\nDataset: {args.output}")
    print(f"Parse checks (eyeball these!): {check_dir}")
    print("Next: python train_model.py --feature-set waterfall "
          f"--dataset {args.output} --output models/waterfall_rf.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
