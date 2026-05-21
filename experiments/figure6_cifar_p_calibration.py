"""
Reproduce Figure 6: CIFAR-10-P prediction stability + RMS calibration error.
 Table 5 (RMS calibration across architectures) and Table 6 (clean error + mFP across architectures).

Usage:
    uv run python experiments/figure6_cifar_p_calibration.py \
        --checkpoint-dir checkpoints/ --data-root data/ --output figures/fig6.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("figures/fig6.pdf"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
