"""
Reproduce Figure 5: CIFAR-10-C error bar chart (ResNeXt backbone).

Usage:
    uv run python experiments/figure5_bar_chart.py \
        --results checkpoints/table1_results.json --output figures/fig5.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("checkpoints/table1_results.json"))
    p.add_argument("--output", type=Path, default=Path("figures/fig5.pdf"))
    main(p.parse_args())
