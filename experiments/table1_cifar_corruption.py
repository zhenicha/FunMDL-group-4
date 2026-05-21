"""
Reproduce Table 1: CIFAR-10-C and CIFAR-100-C corruption error across architectures.

Architectures: AllConvNet, DenseNet-BC (k=12, d=100), WideResNet 40-2, ResNeXt-29 32x4
Saves results to JSON for use by figure5_bar_chart.py.

Usage:
    uv run python experiments/table1_cifar_corruption.py \
        --checkpoint-dir checkpoints/ --data-root data/ --dataset both
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--dataset", choices=["cifar10", "cifar100", "both"], default="both")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("checkpoints/table1_results.json"))
    p.add_argument("--batch-size", type=int, default=256)
    main(p.parse_args())
