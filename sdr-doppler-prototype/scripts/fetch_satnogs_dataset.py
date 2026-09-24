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

    # ~400 observations, mixed satellites, balanced signal / no-signal (10-20 min)
    python scripts/fetch_satnogs_dataset.py --per-class 200

    # a big run: 1500 per class spread over 120 days, plus extra observations
    # of the satellites our station records (METEOR-M2-3/-4, ISS) (1-2 h)
    python scripts/fetch_satnogs_dataset.py --per-class 1500 --days-back 120 --preset station

    # only some satellites (NORAD ids)
    python scripts/fetch_satnogs_dataset.py --norad 25544 57166 59051 --per-class 100

Labels. Every SatNOGS observation is vetted twice by people: its overall
"status" (good = data received) and its "waterfall_status" (with-signal /
without-signal = can you SEE the signal in the waterfall). Our model looks
at waterfalls, so the waterfall vetting is the right label; "status" is only
used when the waterfall was not vetted (column label_source says which).
Observations downloaded before this existed are relabelled with
--refresh-labels (metadata only, no images).

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

import os

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features.waterfall_features import WATERFALL_FEATURE_NAMES, waterfall_features  # noqa: E402
from waterfall_png import WaterfallParseError, png_to_waterfall  # noqa: E402

API = "https://network.satnogs.org/api/observations/"
USER_AGENT = "rtl-sdr-capstone-dataset/1.0 (university capstone; polite, rate-limited)"
LABELS = {"good": 1, "bad": 0}
META_COLUMNS = ["satnogs_id", "norad_cat_id", "status", "waterfall_status", "label_source", "transmitter_mode",
                "observation_frequency", "max_altitude", "station_name", "ground_station", "start", "waterfall_url"]
#: The satellites capture_config.example.json records - extra examples of
#: exactly what our station sees (METEOR LRPT at 137.9 MHz, ISS at 437.8 MHz).
PRESETS = {"station": [57166, 59051, 25544]}
WITH_SIGNAL = {"with-signal", "true", "1", "yes"}
WITHOUT_SIGNAL = {"without-signal", "false", "0", "no"}
CSV_COLUMNS = ["capture_id", *WATERFALL_FEATURE_NAMES, "label", "is_synthetic", "recording_id", "source_file",
               *META_COLUMNS]


#: Longest pause http_get sits through by itself. SatNOGS answers a burst
#: of API requests with 429 + Retry-After of up to an hour (an hourly
#: allowance); anything longer than this is raised as RateLimited so the
#: caller can save progress and tell the user what's going on.
MAX_INLINE_WAIT = 120.0


class RateLimited(Exception):
    """The SatNOGS API asked us to pause for a long time (hourly request allowance used up)."""

    def __init__(self, wait_seconds: float):
        super().__init__(f"SatNOGS rate limit - asked to wait {wait_seconds:.0f} s")
        self.wait_seconds = float(wait_seconds)


def api_token():
    """Optional personal API key from network.satnogs.org (profile page),
    via the SATNOGS_API_TOKEN environment variable. Sent only to the
    SatNOGS API, never to the image host."""
    return (os.environ.get("SATNOGS_API_TOKEN") or "").strip() or None


def keep_awake(enable: bool = True) -> bool:
    """Stop Windows from going to sleep while this process runs (like a video
    player does). Changes no settings; ends when the program ends. macOS/Linux:
    use caffeinate / systemd-inhibit (train_overnight.sh does). Returns True if applied."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:  # never fatal
        return False


def http_get(url: str, *, retries: int = 4, timeout: float = 30.0):
    """(body bytes, headers). Backs off on 429/5xx; raises RateLimited for long pauses."""
    delay = 5.0
    for attempt in range(retries + 1):
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, image/png"}
        token = api_token()
        if token and url.startswith(API):
            headers["Authorization"] = f"Token {token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https hosts
                return resp.read(), resp.headers
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                try:
                    asked = float(exc.headers.get("Retry-After") or delay) if exc.headers else delay
                except ValueError:
                    asked = delay
                if asked > MAX_INLINE_WAIT or attempt >= retries:
                    raise RateLimited(asked) from exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = float((exc.headers.get("Retry-After") if exc.headers else None) or delay)
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


def wait_out_rate_limit(exc: RateLimited, no_wait: bool, sleep=time.sleep) -> bool:
    """Tell the user about the pause and wait it out with a visible countdown.
    Returns False (stop now) when --no-wait was given."""
    from datetime import datetime, timedelta

    wait = exc.wait_seconds + 30  # small margin
    resume = datetime.now() + timedelta(seconds=wait)
    print(f"\n  SatNOGS says we've used this hour's API allowance and asks us to pause {wait / 60:.0f} min.")
    print("  Everything downloaded so far is saved.")
    if no_wait:
        print("  Stopping (--no-wait). Run the same command again later to continue where it left off.")
        return False
    print(f"  Waiting until about {resume:%H:%M}, then continuing automatically. "
          "Or press Ctrl+C now and run the same command later - it resumes.", flush=True)
    remaining = wait
    while remaining > 0:
        step = min(300.0, remaining)
        sleep(step)
        remaining -= step
        if remaining > 0:
            print(f"  ...{remaining / 60:.0f} min to go", flush=True)
    print("  Continuing.", flush=True)
    return True


def label_for(obs: dict, label_source: str = "waterfall"):
    """(label, source) for an observation, or (None, reason) to skip it.

    label_source "waterfall": waterfall vetting, falling back to status;
    "waterfall-only": skip observations whose waterfall wasn't vetted;
    "status": the overall status only (the original behaviour).
    """
    wf = str(obs.get("waterfall_status")).strip().lower() if obs.get("waterfall_status") is not None else ""
    if label_source != "status":
        if wf in WITH_SIGNAL:
            return 1, "waterfall"
        if wf in WITHOUT_SIGNAL:
            return 0, "waterfall"
        if label_source == "waterfall-only":
            return None, "waterfall not vetted"
    status = obs.get("status")
    if status in LABELS:
        return LABELS[status], "status"
    return None, f"status {status!r}"


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


def iter_windows(days_back: int, hours: int = 12, shuffle_seed=None):
    """Time windows covering the last ``days_back`` days.

    Newest first by default. With ``shuffle_seed`` the windows come in a
    random (reproducible) order, so a run that stops at its target has
    still sampled the whole period - different days, passes and conditions
    instead of a single busy day.
    """
    import random
    from datetime import datetime, timedelta, timezone

    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    stop = end - timedelta(days=days_back)
    windows = []
    while end > stop:
        start = end - timedelta(hours=hours)
        windows.append((start, end))
        end = start
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(windows)
    yield from windows


def count_labels(csv_path: Path, norads=None) -> dict:
    """{1: n_signal, 0: n_no_signal}, optionally only for some satellites."""
    counts = {1: 0, 0: 0}
    wanted = {str(n) for n in norads} if norads else None
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if wanted is not None and str(row.get("norad_cat_id", "")).split(".")[0] not in wanted:
                    continue
                counts[int(row["label"])] += 1
    return counts


def existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        return {row["satnogs_id"] for row in csv.DictReader(f)}


def _header(csv_path: Path) -> list:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f), [])


def needs_migration(csv_path: Path) -> bool:
    """True if the file lacks some of today's columns (made by an older version)."""
    header = _header(csv_path)
    return bool(header) and any(c not in header for c in CSV_COLUMNS)


def append_row(csv_path: Path, row: dict) -> None:
    header = _header(csv_path)
    new = not header
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header or CSV_COLUMNS, extrasaction="ignore")
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


def rebuild_features(csv_path: Path, arrays_dir: Path) -> int:
    """Recompute every feature column from the saved 128x128 arrays (after
    the feature set changes). Rows whose array is missing are dropped.
    Returns the number of rows written."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    keep, feats = [], []
    for i, sid in enumerate(df["satnogs_id"]):
        f = arrays_dir / f"{sid}.npy"
        if not f.exists():
            continue
        keep.append(i)
        feats.append(waterfall_features(np.load(f).astype(np.float64)))
    out = df.iloc[keep].reset_index(drop=True)
    for name in WATERFALL_FEATURE_NAMES:
        out[name] = [row[name] for row in feats]
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out[CSV_COLUMNS + [c for c in out.columns if c not in CSV_COLUMNS]].to_csv(csv_path, index=False)
    dropped = len(df) - len(out)
    print(f"Recomputed features for {len(out)} observations" + (f" ({dropped} without a saved array dropped)" if dropped else ""))
    return len(out)


def needs_rebuild(csv_path: Path) -> bool:
    header = _header(csv_path)
    return bool(header) and any(name not in header for name in WATERFALL_FEATURE_NAMES)


def refresh_labels(csv_path: Path, label_source: str, page_delay: float, no_wait: bool = False,
                   sleep=time.sleep, redo: bool = False) -> dict:
    """Re-read each observation's vetting from the API (no image download)
    and relabel. Returns counts of what changed.

    One API request per observation, so the SatNOGS hourly allowance runs out
    after a few dozen: progress is saved every 25 rows, rows already
    refreshed (label_source filled in) are skipped on the next run, and a
    long rate-limit pause is waited out with a countdown (or, with no_wait,
    ends the run)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    for col in ("waterfall_status", "label_source"):
        if col not in df.columns:
            df[col] = ""
    df["waterfall_status"] = df["waterfall_status"].astype(object)
    df["label_source"] = df["label_source"].astype(object)
    changed = checked = failed = dropped = skipped = 0
    drop = []

    def save():
        df.drop(index=[i for i in drop if i in df.index]).to_csv(csv_path, index=False)

    done_before = (df["label_source"].fillna("").astype(str).str.len() > 0) & (not redo)
    todo = int((~done_before).sum())
    if todo < len(df):
        print(f"  {len(df) - todo} observations were already refreshed - skipping them")
    for i, sid in enumerate(df["satnogs_id"]):
        if done_before.iloc[i]:
            skipped += 1
            continue
        while True:
            try:
                body, _ = http_get(API + "?" + urllib.parse.urlencode({"format": "json", "id": sid}))
                data = json.loads(body)
                items = data.get("results", []) if isinstance(data, dict) else data
                obs = next((o for o in items if str(o.get("id")) == str(sid)), None)
                break
            except RateLimited as exc:
                save()
                if not wait_out_rate_limit(exc, no_wait, sleep):
                    return {"checked": checked, "relabelled": changed, "dropped": len(drop), "failed": failed,
                            "stopped_early": True}
            except Exception as exc:  # keep going; a later run can retry
                print(f"  {sid}: {exc}")
                obs = "error"
                break
        if obs == "error":
            failed += 1
            continue
        if obs is None:
            failed += 1
            continue
        checked += 1
        label, source = label_for(obs, label_source)
        df.at[i, "waterfall_status"] = obs.get("waterfall_status")
        df.at[i, "status"] = obs.get("status")
        if label is None:
            drop.append(i)
            continue
        if int(df.at[i, "label"]) != label:
            changed += 1
        df.at[i, "label"] = label
        df.at[i, "label_source"] = source
        if checked % 25 == 0:
            print(f"  checked {checked}/{todo}, {changed} relabelled so far", flush=True)
            save()
        sleep(page_delay)
    if drop:
        df = df.drop(index=drop).reset_index(drop=True)
        dropped = len(drop)
    df.to_csv(csv_path, index=False)
    return {"checked": checked, "relabelled": changed, "dropped": dropped, "failed": failed}


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
    label, source = label_for(obs, getattr(args, "label_source", "waterfall"))
    if label is None:
        return None
    return {
        "capture_id": f"satnogs_{obs['id']}", **feats, "label": label, "is_synthetic": 0,
        "recording_id": f"station_{obs.get('ground_station')}", "source_file": url,
        "satnogs_id": obs["id"], "norad_cat_id": obs.get("norad_cat_id"), "status": obs["status"],
        "waterfall_status": obs.get("waterfall_status"), "label_source": source,
        "transmitter_mode": obs.get("transmitter_mode"), "observation_frequency": obs.get("observation_frequency"),
        "max_altitude": obs.get("max_altitude"), "station_name": obs.get("station_name"),
        "ground_station": obs.get("ground_station"), "start": obs.get("start"), "waterfall_url": url,
    }


def _decode_png(png: bytes) -> np.ndarray:
    import matplotlib.image as mpimg

    return mpimg.imread(io.BytesIO(png), format="png")


class StopRun(Exception):
    """End the download cleanly (rate limit with --no-wait)."""


def harvest(args, state, check_dir, *, targets, window_hours, quota, quota_counts, shuffle_seed):
    """Walk time windows, querying each target satellite (None = any) for
    good and bad observations, until both classes reach the quota.

    quota_counts: counts for this phase only (phase 1); None = the overall
    dataset counts (state["counts"]). Every row also counts overall, and the
    overall --per-class limit is never exceeded.
    """
    counts = state["counts"]
    phase = quota_counts if quota_counts is not None else counts
    windows = iter_windows(args.days_back, hours=window_hours, shuffle_seed=shuffle_seed)
    for n_window, (start, end) in enumerate(windows, 1):
        if all(phase[label] >= quota or counts[label] >= args.per_class for label in LABELS.values()):
            return
        # Query good and bad inside every window, so the set stays balanced even if interrupted.
        for status, wanted in LABELS.items():
            if phase[wanted] >= quota or counts[wanted] >= args.per_class:
                continue
            for norad in targets:
                while True:
                    try:
                        observations = list(fetch_window(status, norad, start, end, args.page_delay))
                        break
                    except RateLimited as exc:
                        if not wait_out_rate_limit(exc, args.no_wait):
                            raise StopRun from exc
                    except Exception as exc:  # network hiccup: skip this window, keep going
                        print(f"  window {start:%m-%d %H}h {status}: {exc}")
                        observations = []
                        break
                added = 0
                for obs in observations:
                    if added >= args.per_window:
                        break
                    if str(obs.get("id")) in state["done"] or obs.get("status") != status:
                        continue
                    label, _ = label_for(obs, args.label_source)
                    if label is None or phase[label] >= quota or counts[label] >= args.per_class:
                        continue
                    station = obs.get("ground_station")
                    if state["per_station"].get((station, label), 0) >= args.max_per_station:
                        continue
                    try:
                        row = process(obs, args, args.arrays_dir, check_dir, state["checks_done"])
                    except Exception as exc:  # one bad download mustn't stop the run
                        print(f"  skip {obs.get('id')}: {exc}")
                        continue
                    if row is None:
                        continue
                    append_row(args.output, row)
                    state["done"].add(str(obs["id"]))
                    state["per_station"][(station, label)] = state["per_station"].get((station, label), 0) + 1
                    counts[label] += 1
                    if quota_counts is not None:
                        quota_counts[label] += 1
                    added += 1
                    time.sleep(args.delay)
        print(f"  window {n_window} ({start:%Y-%m-%d %H}h): {counts[1]} signal / {counts[0]} no signal so far",
              flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download vetted SatNOGS observations as a labelled waterfall dataset.")
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--output", type=Path, default=root / "data" / "training" / "satnogs_waterfall_features.csv")
    p.add_argument("--arrays-dir", type=Path, default=root / "data" / "satnogs" / "waterfalls")
    p.add_argument("--per-class", type=int, default=200, help="Target number of signal AND of no-signal observations")
    p.add_argument("--norad", type=int, nargs="*", default=None, help="Only these satellites (default: any)")
    p.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="station: also query the satellites our station records (METEOR-M2-3/-4, ISS) in every window")
    p.add_argument("--label-source", choices=("waterfall", "waterfall-only", "status"), default="waterfall",
                   help="waterfall (default): waterfall vetting, else status; waterfall-only: skip unvetted "
                        "waterfalls; status: overall status only")
    p.add_argument("--max-per-station", type=int, default=15, help="Cap per station per class, for variety")
    p.add_argument("--preset-per-class", type=int, default=150,
                   help="With --preset: how many signal and no-signal observations of those satellites to get first")
    p.add_argument("--per-window", type=int, default=10,
                   help="Max observations per class per satellite query per time window, for variety")
    p.add_argument("--window-hours", type=int, default=12, help="Length of each time window")
    p.add_argument("--keep-awake", action="store_true",
                   help="Windows: keep the computer from sleeping until the download finishes (for overnight runs)")
    p.add_argument("--refresh-all", action="store_true",
                   help="With --refresh-labels: also re-check observations refreshed before")
    p.add_argument("--no-wait", action="store_true",
                   help="When SatNOGS rate-limits us, stop (progress saved) instead of waiting it out")
    p.add_argument("--days-back", type=int, default=45, help="How far back in time to look")
    p.add_argument("--newest-first", action="store_true",
                   help="Walk windows newest first instead of in a random order over the whole period")
    p.add_argument("--seed", type=int, default=0, help="Random order of the time windows (reproducible)")
    p.add_argument("--delay", type=float, default=0.5, help="Seconds between image downloads")
    p.add_argument("--page-delay", type=float, default=1.5, help="Seconds between API pages")
    p.add_argument("--parse-checks", type=int, default=6, help="Save this many side-by-side parse-check images")
    p.add_argument("--keep-png", action="store_true")
    p.add_argument("--rebuild-features", action="store_true",
                   help="Only recompute the feature columns from the saved arrays, then stop")
    p.add_argument("--refresh-labels", action="store_true",
                   help="Only re-read each saved observation's vetting from SatNOGS and relabel, then stop")
    args = p.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.arrays_dir.mkdir(parents=True, exist_ok=True)
    check_dir = args.arrays_dir.parent / "parse_check"
    check_dir.mkdir(parents=True, exist_ok=True)

    if args.keep_awake:
        print("Keeping the computer awake until this finishes" if keep_awake()
              else "(--keep-awake: not on Windows - the overnight script uses caffeinate/systemd-inhibit instead)")
    if api_token():
        print("Using your SatNOGS API token (SATNOGS_API_TOKEN)")

    if args.rebuild_features:
        if not args.output.exists():
            print(f"Nothing to rebuild: {args.output} does not exist")
            return 0
        rebuild_features(args.output, args.arrays_dir)
        return 0
    if needs_rebuild(args.output) or needs_migration(args.output):
        print("Dataset was made by an older version - updating its columns first")
        rebuild_features(args.output, args.arrays_dir)
    if args.refresh_labels:
        if not args.output.exists():
            print(f"Nothing to relabel: {args.output} does not exist")
            return 0
        before = count_labels(args.output)
        r = refresh_labels(args.output, args.label_source, args.page_delay, no_wait=args.no_wait,
                           redo=args.refresh_all)
        after = count_labels(args.output)
        print(f"Checked {r['checked']} observations: {r['relabelled']} relabelled, {r['dropped']} dropped, "
              f"{r['failed']} could not be read" + (" - stopped early, run again to continue" if r.get("stopped_early")
                                                    else ""))
        print(f"Labels before: {before[1]} signal / {before[0]} no signal; after: {after[1]} / {after[0]}")
        return 0

    done = existing_ids(args.output)
    print(f"Writing {args.output} ({len(done)} observations already there)")

    checks_done: list = []
    counts = count_labels(args.output)
    per_station: dict = {}
    print(f"Target: {args.per_class} signal + {args.per_class} no-signal "
          f"(have {counts[1]} / {counts[0]}; labels from {args.label_source})")
    state = {"done": done, "counts": counts, "per_station": per_station, "checks_done": checks_done}
    shuffle = None if args.newest_first else args.seed
    try:
        if args.preset:
            # Phase 1: our own satellites. They have far fewer observations than
            # "any satellite", so they get multi-day windows and their own quota.
            sats = [n for n in PRESETS[args.preset]]
            have = count_labels(args.output, norads=sats)
            print(f"Phase 1 - our satellites {sats}: up to {args.preset_per_class} per class "
                  f"(have {have[1]} / {have[0]})")
            harvest(args, state, check_dir, targets=sats, window_hours=72, quota=args.preset_per_class,
                    quota_counts=have, shuffle_seed=shuffle)
            print("Phase 2 - all satellites")
        harvest(args, state, check_dir, targets=list(args.norad) if args.norad else [None],
                window_hours=args.window_hours, quota=args.per_class, quota_counts=None, shuffle_seed=shuffle)
    except StopRun:
        pass
    counts = state["counts"]
    if min(counts.values()) == 0:
        print("\nWARNING: one class has no observations - training needs both. "
              "Run again (it resumes) or raise --days-back.")
    print(f"\nDataset: {args.output}")
    print(f"Parse checks (eyeball these!): {check_dir}")
    print("Next: python train_model.py --feature-set waterfall "
          f"--dataset {args.output} --output models/waterfall_rf.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
