# Processing benchmark - one recording through pipeline.process_recording()

Generated 2026-10-05T12:13:01+00:00 on commit `040c140f28ce292f3b93299ae403604067b9327d` by `tools/benchmark_processing.py`.

Machine: Linux-6.18.44-fc-v70-x86_64-with-glibc2.39, 2 CPUs, 8.4 GB RAM; Python 3.13.16, numpy 2.5.3, scipy 1.18.1.

Input: synthetic rtl_sdr cu8 captures at 1,024,000 samples/s (drifting carrier in noise - exercises the code path only, not detection accuracy). Detection window cap (`max_detection_seconds`): 120.0.

| Capture length | File size | Elapsed | Peak memory (RSS) | Throughput | Row written |
|---|---|---|---|---|---|
| 10 s | 20 MB | 3.4 s | 717 MB | 2.9x real time | yes |
| 20 s | 41 MB | 5.6 s | 1,265 MB | 3.6x real time | yes |
| 60 s | 123 MB | 14.3 s | 3,456 MB | 4.2x real time | yes |

Peak memory is the whole child process (Python + numpy/scipy/matplotlib imports + processing).

- Detection (spectrogram + features + rule/ML) reads at most `max_detection_seconds` of the file; the Doppler waterfall streams through the whole file. Longer captures therefore cost more time but should not need more memory beyond the cap.
- A negative return code with no output means the child process was killed (on Linux usually the out-of-memory killer).

## Additional runs, same commit and machine (5 Oct 2026)

| Capture | Detection window | Result | Peak memory |
|---|---|---|---|
| 120 s (246 MB) | 60 s (`--max-detection-seconds 60`) | SUCCESS in 12.5 s, row written | 3,456 MB |
| 120 s (246 MB) | 120 s (default) | **Killed (exit -9, out of memory)** on this 8 GB machine | - |

So on an 8 GB computer set `max_detection_seconds: 60`; the default 120 s window needs more than 8 GB.

**Before reconciliation** (`775a92f`, which lacked the spectrogram-image fix `6bb4906`), measured on the same machine
the same day without trained models: 10 s 1,569 MB, 20 s 2,958 MB. After: 717 MB and 1,265 MB.

Environment: Python 3.13.16, Linux-6.18.44 x86_64 (container, 2 CPUs, 8 GB RAM); numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.5, matplotlib 3.11.2, pytest 9.1.1. Models loaded: IQ and waterfall (`ml_status` AVAILABLE) for the main table.
