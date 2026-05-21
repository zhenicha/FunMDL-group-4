"""
Reproduce Figure 12: CIFAR-10-C error rate per corruption type (Appendix D).

Usage:
    uv run python experiments/figure12_per_corruption.py \
        --checkpoint-dir checkpoints/ --data-root data/ --output figures/fig12.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("figures/fig12.pdf"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
