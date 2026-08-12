#!/usr/bin/env python3
"""Convenience entrypoint matching the example usage:

    python train_model.py --dataset data/training/features.csv --output models/random_forest.joblib

Delegates to ml.train; the real implementation lives in src/ml/train.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ml.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
