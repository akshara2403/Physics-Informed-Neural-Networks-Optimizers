#!/usr/bin/env python3
"""Thin CLI wrapper so the benchmark can be run without installing the
package, e.g. `python scripts/run_experiment.py --config configs/quick.yaml`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pinn_bench.run import main  # noqa: E402

if __name__ == "__main__":
    main()
