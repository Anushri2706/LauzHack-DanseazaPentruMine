"""Data prep for the HMM regime engine (backend plan §2): person-month table, holdout split,
outlier flags, December offsets, per-regime stats, spend regression and income/rent baselines.

Usage: python backend/prepare_data.py [--holdout-frac 0.2 --seed 0 --outlier-z 6 --out-dir DIR]
Writes to backend/artifacts/ by default.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from twin.data import main  # noqa: E402

if __name__ == "__main__":
    main()
