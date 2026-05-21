"""
Reproduce Figure 3: Image drift from deep augmentation chains.

Usage:
    uv run python experiments/figure3_augmentation_drift.py \
        --image data/sample.jpg --output figures/fig3.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("figures/fig3.pdf"))
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
