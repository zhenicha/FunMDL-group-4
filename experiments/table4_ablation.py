"""
Reproduce Table 4: Ablation study on CIFAR-10-C and CIFAR-100-C.

Variants: Standard → AutoAugment* → Random AutoAugment* → +JSD → +Mixing → AugMix

Usage:
    uv run python experiments/table4_ablation.py \
        --checkpoint-dir checkpoints/ --data-root data/
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
