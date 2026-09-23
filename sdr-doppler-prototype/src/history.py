"""Print the capture database as readable tables: recent captures, the
station status log, and (optionally) one capture's satellite track.

    python src/history.py                      # last 20 captures + last 20 status events
    python src/history.py --capture 12         # full detail + position history for row 12
    python src/history.py --db other.sqlite3 --limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DB_PATH  # noqa: E402
from database import get_pass_positions, get_result, list_results, list_status  # noqa: E402


def _fmt(value, width, digits=None):
    if value is None:
        text = "-"
    elif digits is not None and isinstance(value, (int, float)):
        text = f"{value:.{digits}f}"
    else:
        text = str(value)
    return text[:width].ljust(width)


def captures_table(rows) -> str:
    header = (f"{'id':>4}  {'time (UTC)':<19}  {'satellite':<14} {'MHz':>9}  {'recording':<11} "
              f"{'rule':<4} {'ML':<4} {'ML conf':>7}  {'IQ file':<24}")
    lines = [header, "-" * len(header)]
    for r in rows:
        mhz = r.get("frequency_hz") / 1e6 if r.get("frequency_hz") else None
        rule = "yes" if r.get("rule_detection_result") else ("no" if r.get("rule_detection_result") == 0 else "-")
        ml = {1: "yes", 0: "no"}.get(r.get("ml_detection_result"), "-")
        lines.append(
            f"{r['id']:>4}  {_fmt((r.get('timestamp_utc') or '')[:19], 19)}  {_fmt(r.get('satellite_name'), 14)} "
            f"{_fmt(mhz, 9, 3):>9}  {_fmt(r.get('recording_status'), 11)} {rule:<4} {ml:<4} "
            f"{_fmt(r.get('ml_confidence_score'), 7, 2):>7}  {_fmt(r.get('iq_retention'), 24)}"
        )
    if not rows:
        lines.append("(no captures yet)")
    return "\n".join(lines)


def status_table(events) -> str:
    lines = [f"{'time (UTC)':<19}  {'component':<10} {'state':<12} message", "-" * 70]
    for e in events:
        lines.append(f"{e['timestamp_utc'][:19]:<19}  {e['component']:<10} {e['state']:<12} {e.get('message') or ''}")
    if not events:
        lines.append("(no status events yet)")
    return "\n".join(lines)


def capture_detail(db_path: Path, capture_id: int) -> str:
    row = get_result(db_path, capture_id)
    if row is None:
        return f"No capture with id {capture_id}."
    lines = [f"Capture {capture_id}", "=" * 20]
    lines += [f"{k}: {v}" for k, v in row.items() if v is not None]
    track = get_pass_positions(db_path, capture_id)
    lines += ["", f"Position history ({len(track)} points)"]
    if track:
        lines.append(f"{'time (UTC)':<19}  {'az':>6} {'el':>6} {'range km':>9} {'Doppler Hz':>11}")
        for p in track:
            lines.append(f"{p['timestamp_utc'][:19]:<19}  {p['azimuth_deg']:6.1f} {p['elevation_deg']:6.1f} "
                         f"{p['range_km']:9.1f} {('-' if p['doppler_hz'] is None else format(p['doppler_hz'], '.0f')):>11}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show captures, status log and pass tracks from the database.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--capture", type=int, default=None, help="Show one capture in full, with its track")
    args = parser.parse_args(argv)
    if args.capture is not None:
        print(capture_detail(args.db, args.capture))
        return 0
    print(f"Database: {args.db}\n")
    print("Recent captures")
    print(captures_table(list_results(args.db, args.limit)))
    print("\nStation status log")
    print(status_table(list_status(args.db, args.limit)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
