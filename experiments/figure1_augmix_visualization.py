"""
Reproduce Figure 1: Visual comparison of augmentation techniques.

Usage:
    uv run python experiments/figure1_augmix_visualization.py \
        --image data/sample.jpg --output figures/fig1.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("figures/fig1.pdf"))
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
