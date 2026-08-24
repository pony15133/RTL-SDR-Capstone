from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


@dataclass
class SpectrogramData:
    power_db: np.ndarray
    frequencies_hz: np.ndarray
    times_s: np.ndarray


def iq_to_spectrogram(
    iq_samples: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float,
    nperseg: int,
    noverlap: int,
) -> SpectrogramData:
    freqs, times, power = signal.spectrogram(
        iq_samples,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )
    order = np.argsort(freqs)
    freqs = freqs[order] + center_freq_hz
    power = power[order, :]
    power_db = 10.0 * np.log10(power + np.finfo(float).eps)
    return SpectrogramData(power_db=power_db.T, frequencies_hz=freqs, times_s=times)


def matrix_to_spectrogram(
    matrix: np.ndarray,
    sample_rate_hz: float,
    center_freq_hz: float,
) -> SpectrogramData:
    power_db = np.asarray(matrix, dtype=float)
    frequencies_hz = np.linspace(
        center_freq_hz - sample_rate_hz / 2.0,
        center_freq_hz + sample_rate_hz / 2.0,
        power_db.shape[1],
    )
    times_s = np.arange(power_db.shape[0], dtype=float)
    return SpectrogramData(power_db=power_db, frequencies_hz=frequencies_hz, times_s=times_s)


def save_chunked_spectrogram_images(
    spec: SpectrogramData,
    output_dir: Path,
    *,
    chunk_size: int,
    overlap: int = 0,
    prefix: str = "spectrogram",
) -> list[Path]:
    """Render a spectrogram in time chunks to stay memory-safe on large captures.

    Each chunk is written as a separate PNG with a numbered suffix.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap!r}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap must be smaller than chunk_size ({chunk_size}), got {overlap!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = len(spec.times_s)
    if total_rows <= chunk_size:
        out_path = output_dir / f"{prefix}_chunk_000.png"
        save_spectrogram_image(spec, out_path)
        return [out_path]

    chunk_paths: list[Path] = []
    start = 0
    chunk_index = 0
    while start < total_rows:
        end = min(total_rows, start + chunk_size)
        chunk_spec = SpectrogramData(
            power_db=spec.power_db[start:end, :],
            frequencies_hz=spec.frequencies_hz,
            times_s=spec.times_s[start:end],
        )
        out_path = output_dir / f"{prefix}_chunk_{chunk_index:03d}.png"
        save_spectrogram_image(chunk_spec, out_path)
        chunk_paths.append(out_path)
        if end >= total_rows:
            break
        start = max(start + 1, end - overlap)
        chunk_index += 1

    return chunk_paths


def save_spectrogram_image(spec: SpectrogramData, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    extent = [
        spec.frequencies_hz[0] / 1e6,
        spec.frequencies_hz[-1] / 1e6,
        spec.times_s[-1] if len(spec.times_s) else spec.power_db.shape[0],
        0,
    ]
    plt.imshow(spec.power_db, aspect="auto", extent=extent, cmap="viridis")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.colorbar(label="Power [dB]")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
