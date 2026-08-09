#!/usr/bin/env python3
"""Convenience entrypoint matching the spec's example usage:

    python recorder.py --frequency 137900000 --sample-rate 2400000 \
        --gain 30 --duration 60 --satellite "METEOR-M2-4"

    python recorder.py --simulate --duration 10

This just delegates to ``rtl_recorder.main``; the real implementation lives
in the ``rtl_recorder`` package.
"""

import sys

from rtl_recorder.main import main

if __name__ == "__main__":
    sys.exit(main())
