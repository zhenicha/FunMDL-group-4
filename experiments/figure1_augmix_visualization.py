"""
Reproduce Figure 1: Visual comparison of augmentation techniques.

Usage:
    # Pull bird+frog directly from CIFAR-10 (recommended):
    uv run python experiments/figure1_augmix_visualization.py \
        --from-cifar --data-root data/ --output figures/fig1.pdf

    # Or provide your own images:
    uv run python experiments/figure1_augmix_visualization.py \
        --image1 data/bird.png --image2 data/frog.png --output figures/fig1.pdf
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no GUI window
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.augmix.augmentations import augment_and_mix


def get_cifar_images(data_root: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return one bird (class 2) and one frog (class 6) from CIFAR-10."""
    from torchvision import datasets
    cifar = datasets.CIFAR10(data_root, train=True, download=True)
    rng = np.random.default_rng(seed)
    bird_idx = rng.choice([i for i, t in enumerate(cifar.targets) if t == 2])
    frog_idx = rng.choice([i for i, t in enumerate(cifar.targets) if t == 6])
    return np.array(cifar.data[bird_idx]), np.array(cifar.data[frog_idx])


def cutout(img: np.ndarray) -> np.ndarray:
    """Zero out a square patch of ~25% image width, centred randomly."""
    img = img.copy()
    h, w = img.shape[:2]
    pad_size = w // 4
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1, y2 = max(0, cy - pad_size), min(h, cy + pad_size)
    x1, x2 = max(0, cx - pad_size), min(w, cx + pad_size)
    img[y1:y2, x1:x2] = 0
    return img


def mixup(img1: np.ndarray, img2: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Linear interpolation between two images."""
    return np.clip(lam * img1.astype(np.float32) + (1 - lam) * img2.astype(np.float32), 0, 255).astype(np.uint8)


def cutmix(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Paste a random rectangular region from img2 into img1."""
    img = img1.copy()
    h, w = img.shape[:2]
    cut_ratio = np.sqrt(0.5)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]
    return img


def load_image(path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    if args.from_cifar:
        img1, img2 = get_cifar_images(args.data_root, args.seed)
    else:
        if args.image1 is None or args.image2 is None:
            raise ValueError("Provide --image1 and --image2, or use --from-cifar")
        img1 = load_image(args.image1)
        img2 = load_image(args.image2)

    pil1 = Image.fromarray(img1)
    pil2 = Image.fromarray(img2)

    rows = [
        (img1, img2, pil1),
        (img2, img1, pil2),
    ]

    col_titles = ["Original", "Cutout", "MixUp", "CutMix", "AugMix"]
    n_rows, n_cols = 2, 5

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.subplots_adjust(wspace=0.05, hspace=0.15)

    for r, (base_np, mix_np, base_pil) in enumerate(rows):
        augmix_pil = augment_and_mix(base_pil)

        images = [
            base_np,
            cutout(base_np),
            mixup(base_np, mix_np),
            cutmix(base_np, mix_np),
            np.array(augmix_pil),
        ]

        for c, img in enumerate(images):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--from-cifar", action="store_true")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--image1", type=Path, default=None, help="First source image")
    p.add_argument("--image2", type=Path, default=None, help="Second source image")
    p.add_argument("--output", type=Path, default=Path("figures/fig1.jpg"))
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
