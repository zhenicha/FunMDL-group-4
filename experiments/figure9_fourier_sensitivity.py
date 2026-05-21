"""
Reproduce Figure 9: Fourier sensitivity heatmap (Appendix B).

Method: add each of the 32x32 Fourier basis vectors to the CIFAR-10 test set
one at a time, record error rate → 32x32 heatmap per model.
Requires: Standard, Cutout, and AugMix checkpoints (WideResNet 40-2).

Usage:
    uv run python experiments/figure9_fourier_sensitivity.py \
        --checkpoint-dir checkpoints/ --data-root data/ --output figures/fig9.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("figures/fig9.pdf"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
