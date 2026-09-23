from dataclasses import dataclass
from pathlib import Path

import numpy as np

from iq_io import IQReader, is_raw_iq_file, probe


@dataclass
class LoadedData:
    path: Path
    kind: str
    values: np.ndarray


def load_input(path: Path, binary_dtype: str = "auto") -> LoadedData:
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

    if is_raw_iq_file(path):
        # binary_dtype: "auto" (default) looks at SigMF/sidecar/file name/extension;
        # an explicit cu8 / ci16 / complex64 / wav always wins.
        info = probe(path, iq_format=binary_dtype)
        if info.iq_format == "complex64":
            values = np.memmap(path, dtype=np.complex64, mode="r")
        else:
            values = IQReader(path, info.iq_format).read_all()
        return LoadedData(path=path, kind="iq", values=values)

    raise ValueError(f"Unsupported input type: {path.suffix}")


def is_raw_iq_path(path: Path) -> bool:
    return Path(path).suffix.lower() == ".npy" or is_raw_iq_file(path)
