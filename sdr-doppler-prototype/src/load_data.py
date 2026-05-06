from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoadedData:
    path: Path
    kind: str
    values: np.ndarray


def load_input(path: Path, binary_dtype: str = "complex64") -> LoadedData:
    """Load either raw IQ samples or an existing spectrogram matrix."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        values = np.load(path, mmap_mode="r")
        if values.ndim == 1 or np.iscomplexobj(values):
            return LoadedData(path=path, kind="iq", values=values)
        if values.ndim == 2:
            return LoadedData(path=path, kind="spectrogram", values=values)
        raise ValueError(f"Unsupported .npy shape {values.shape}; expected 1D IQ or 2D spectrogram")

    if suffix in {".txt", ".csv"}:
        delimiter = "," if suffix == ".csv" else None
        values = np.loadtxt(path, delimiter=delimiter)
        if values.ndim != 2:
            raise ValueError("Text/CSV input must contain a 2D spectrogram matrix")
        return LoadedData(path=path, kind="spectrogram", values=values)

    if suffix in {".bin", ".iq", ".dat"}:
        values = np.memmap(path, dtype=np.dtype(binary_dtype), mode="r")
        return LoadedData(path=path, kind="iq", values=values)

    raise ValueError(f"Unsupported input type: {path.suffix}")


def is_raw_iq_path(path: Path) -> bool:
    return path.suffix.lower() in {".npy", ".bin", ".iq", ".dat"}
