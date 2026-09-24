# Processing benchmark - one recording through pipeline.process_recording()

Generated 2026-09-24T08:54:43+00:00 on commit `7ed7ec43c8b0b43b020257de6e68cc18d43eebc3` by `tools/benchmark_processing.py`.

Machine: Linux-6.18.44-fc-v37-x86_64-with-glibc2.39, 4 CPUs, 16.9 GB RAM; Python 3.11.15, numpy 2.4.6, scipy 1.17.1.

Input: synthetic rtl_sdr cu8 captures at 1,024,000 samples/s (drifting carrier in noise - exercises the code path only, not detection accuracy). Detection window cap (`max_detection_seconds`): 120.0.

| Capture length | File size | Elapsed | Peak memory (RSS) | Throughput | Row written |
|---|---|---|---|---|---|
| 10 s | 20 MB | 6.0 s | 721 MB | 1.7x real time | yes |
| 60 s | 123 MB | 35.0 s | 3,459 MB | 1.7x real time | yes |
| 120 s | 246 MB | 70.7 s | 6,745 MB | 1.7x real time | yes |
| 600 s | 1229 MB | 70.0 s | 6,745 MB | 8.6x real time | yes |

Peak memory is the whole child process (Python + numpy/scipy/matplotlib imports + processing).

- Detection (spectrogram + features + rule/ML) reads at most `max_detection_seconds` of the file; the Doppler waterfall streams through the whole file. Longer captures therefore cost more time but should not need more memory beyond the cap.
- A negative return code with no output means the child process was killed (on Linux usually the out-of-memory killer).
