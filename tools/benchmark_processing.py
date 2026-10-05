#!/usr/bin/env python3
"""Processing-time and peak-memory benchmark for one recording (NFR 5.1 evidence).

    python tools/benchmark_processing.py                          # 10 s, 60 s, 120 s captures
    python tools/benchmark_processing.py --seconds 10 60 120 600  # add a whole-pass-length file
    python tools/benchmark_processing.py --sample-rate 2048000

Each case writes a synthetic rtl_sdr-format capture (interleaved unsigned
8-bit I/Q - exactly what the recorder writes) containing a slowly drifting
carrier in noise, then runs pipeline.process_recording() on it - the same
call auto_capture.py makes after every pass: IQ read -> spectrogram ->
features -> rule detector -> ML detector (if a model exists) -> Doppler
waterfall -> JSON/PNG -> SQLite row -> retention (keep-all here).

Each case runs in its own child process so peak memory is measured per
case (Linux/macOS: ru_maxrss; Windows: psutil peak working set if
installed, otherwise not reported). The synthetic signal only exercises
the code path - it says nothing about detection accuracy.

Results go to evidence/performance/benchmark_<host>_<date>.json and .md
with Python/OS/CPU/RAM details, so runs on different laptops can be compared.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO / "evidence" / "performance"


def peak_rss_mb():
    """Peak resident memory of this process in MB, or None where unavailable."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux
    except ImportError:  # Windows
        try:
            import psutil

            return psutil.Process().memory_info().peak_wset / (1024 * 1024)
        except (ImportError, AttributeError):
            return None


def write_capture(path: Path, seconds: float, sample_rate: int, chunk_seconds: float = 5.0) -> int:
    """Synthetic cu8 capture, written in chunks so generating it never needs much memory."""
    import numpy as np

    rng = np.random.default_rng(0)
    total = int(seconds * sample_rate)
    chunk = int(chunk_seconds * sample_rate)
    phase = 0.0
    with open(path, "wb") as f:
        for start in range(0, total, chunk):
            n = min(chunk, total - start)
            t = (start + np.arange(n)) / sample_rate
            freq = 20_000.0 - 30_000.0 * t / max(seconds, 1e-9)       # slow Doppler-like drift
            inst = phase + 2 * np.pi * np.cumsum(freq) / sample_rate
            phase = float(inst[-1])
            iq = 0.3 * np.exp(1j * inst) + 0.2 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
            inter = np.empty(2 * n, dtype=np.float32)
            inter[0::2], inter[1::2] = iq.real, iq.imag
            f.write(np.clip(np.round(inter * 127.5 + 127.5), 0, 255).astype(np.uint8).tobytes())
    return path.stat().st_size


def run_case(iq_path: Path, sample_rate: int, workdir: Path, max_detection_seconds) -> dict:
    """Child-process body: process one recording and report time + peak memory."""
    sys.path.insert(0, str(REPO))
    import pipeline
    from rtl_recorder.metadata import RecordingResult
    from rtl_recorder.states import RecordingStatus

    result = RecordingResult(status=RecordingStatus.SUCCESS, output_file=str(iq_path),
                             output_file_size=iq_path.stat().st_size)
    before = peak_rss_mb()
    t0 = time.perf_counter()
    pr = pipeline.process_recording(
        result, frequency_hz=137_750_000, sample_rate_hz=sample_rate, target_frequency_hz=137_900_000,
        db_path=workdir / "bench.sqlite3", output_dir=workdir / "results", save_image=True,
        retention_policy="keep-all", max_detection_seconds=max_detection_seconds,
    )
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": round(elapsed, 2), "peak_rss_mb": None if peak_rss_mb() is None else round(peak_rss_mb(), 1),
            "baseline_rss_mb_after_imports": None if before is None else round(before, 1),
            "db_row": pr.result_id, "rule_detected": pr.detected, "ml_status": pr.ml_status,
            "waterfall_status": pr.wf_status, "success": pr.success}


def environment() -> dict:
    import numpy
    import scipy

    info = {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or None, "cpu_count": os.cpu_count(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        try:
            import psutil

            info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        except ImportError:
            info["ram_gb"] = None
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                                            check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info


def to_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Processing benchmark - one recording through pipeline.process_recording()", "",
        f"Generated {env['generated_utc']} on commit `{env['git_commit']}` by `tools/benchmark_processing.py`.", "",
        f"Machine: {env['platform']}, {env['cpu_count']} CPUs, {env['ram_gb']} GB RAM; Python {env['python']}, "
        f"numpy {env['numpy']}, scipy {env['scipy']}.", "",
        f"Input: synthetic rtl_sdr cu8 captures at {report['sample_rate']:,} samples/s (drifting carrier in noise - "
        f"exercises the code path only, not detection accuracy). Detection window cap "
        f"(`max_detection_seconds`): {report['max_detection_seconds']}.", "",
        "| Capture length | File size | Elapsed | Peak memory (RSS) | Throughput | Row written |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["cases"]:
        if c.get("error"):
            lines.append(f"| {c['seconds']:g} s | {c['file_mb']:.0f} MB | FAILED: {c['error']} | | | |")
            continue
        rss = "n/a" if c["peak_rss_mb"] is None else f"{c['peak_rss_mb']:,.0f} MB"
        lines.append(f"| {c['seconds']:g} s | {c['file_mb']:.0f} MB | {c['elapsed_s']:.1f} s | {rss} | "
                     f"{c['seconds'] / c['elapsed_s']:.1f}x real time | {'yes' if c['db_row'] else 'no'} |")
    lines += ["", "Peak memory is the whole child process (Python + numpy/scipy/matplotlib imports + processing).", ""]
    lines += [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Benchmark processing time and peak memory per recording.")
    p.add_argument("--seconds", type=float, nargs="+", default=[10, 60, 120], help="Capture lengths to test")
    p.add_argument("--sample-rate", type=int, default=1_024_000, help="Default: the capture config's 1.024 Msps")
    p.add_argument("--max-detection-seconds", type=float, default=120.0,
                   help="Same cap process_recording() uses by default (detection reads at most this much)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--keep-files", action="store_true", help="Keep the generated captures")
    p.add_argument("--_child", nargs=3, metavar=("IQ", "RATE", "WORKDIR"), help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    if args._child:
        iq, rate, workdir = args._child
        print(json.dumps(run_case(Path(iq), int(rate), Path(workdir), args.max_detection_seconds)))
        return 0

    cases = []
    with tempfile.TemporaryDirectory(prefix="sdr_bench_") as tmp:
        tmp = Path(tmp)
        for seconds in args.seconds:
            iq = tmp / f"bench_{seconds:g}s_137750000Hz.iq"
            size = write_capture(iq, seconds, args.sample_rate)
            print(f"{seconds:g} s capture ({size / 1e6:.0f} MB): processing...", flush=True)
            proc = subprocess.run([sys.executable, __file__, "--max-detection-seconds", str(args.max_detection_seconds),
                                   "--_child", str(iq), str(args.sample_rate), str(tmp / f"work_{seconds:g}")],
                                  capture_output=True, text=True)
            case = {"seconds": seconds, "file_mb": size / 1e6, "samples": size // 2}
            try:
                case.update(json.loads(proc.stdout.strip().splitlines()[-1]))
            except (IndexError, ValueError):
                tail = (proc.stderr or proc.stdout).strip().splitlines()
                case["error"] = (tail[-1] if tail else f"exit code {proc.returncode}")[:200]
                case["returncode"] = proc.returncode
            print("   ", case, flush=True)
            cases.append(case)
            if not args.keep_files:
                iq.unlink(missing_ok=True)

    notes = [
        "Detection (spectrogram + features + rule/ML) reads at most `max_detection_seconds` of the file; the Doppler "
        "waterfall streams through the whole file. Longer captures therefore cost more time but should not need more "
        "memory beyond the cap.",
        "A negative return code with no output means the child process was killed (on Linux usually the "
        "out-of-memory killer).",
    ]
    report = {"sample_rate": args.sample_rate, "max_detection_seconds": args.max_detection_seconds,
              "cases": cases, "notes": notes, "environment": environment()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{platform.system().lower()}_{datetime.now(timezone.utc):%Y%m%d}"
    (args.output_dir / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    print(f"Wrote {args.output_dir / (stem + '.json')} and .md")
    return 0 if all(not c.get("error") for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
