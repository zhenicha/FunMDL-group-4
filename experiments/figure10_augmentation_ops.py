"""
Reproduce Figure 10: Illustration of all augmentation operations (Appendix C).

Usage:
    uv run python experiments/figure10_augmentation_ops.py \
        --image data/sample.jpg --output figures/fig10.pdf
"""

import argparse
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("figures/fig10.pdf"))
    main(p.parse_args())
