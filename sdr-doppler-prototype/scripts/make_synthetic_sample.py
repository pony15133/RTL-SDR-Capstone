from pathlib import Path

import numpy as np


def main() -> None:
    sample_rate_hz = 240_000.0
    samples = 240_000
    t = np.arange(samples) / sample_rate_hz
    start_offset_hz = -40_000.0
    end_offset_hz = 40_000.0
    chirp_rate = (end_offset_hz - start_offset_hz) / t[-1]
    phase = 2 * np.pi * (start_offset_hz * t + 0.5 * chirp_rate * t * t)

    rng = np.random.default_rng(7)
    noise = 0.15 * (rng.normal(size=samples) + 1j * rng.normal(size=samples))
    iq = np.exp(1j * phase) + noise

    output = Path("data/raw/sample.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, iq.astype(np.complex64))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
