"""
Training loop for AugMix on CIFAR-10/100.

Paper 4.1 setup:
  - SGD + Nesterov momentum, lr=0.1, cosine decay
  - WideResNet / AllConvNet: 100 epochs
  - DenseNet / ResNeXt: 200 epochs
  - weight_decay=0.0001 for Mixup, 0.0005 otherwise

Each forward pass produces three outputs (orig, aug1, aug2) — needed for JSD loss.

"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

def jsd_loss(
    logits_orig: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
    labels: torch.Tensor,
    lam: float = 12.0,
) -> torch.Tensor:
    """
    AugMix loss: cross-entropy on clean image + λ * JSD consistency term.
    Loss = CE(p_orig, y) + λ * JS(p_orig; p_augmix1; p_augmix2)

    JS is computed as:
        M = (p_orig + p_augmix1 + p_augmix2) / 3
        JS = (KL(p_orig‖M) + KL(p_augmix1‖M) + KL(p_augmix2‖M)) / 3
        CE = cross_entropy(logits_orig, labels) applies softmax internally in PyTorch so we don't pass p_orig.
    Args:
        logits_orig:  (B, C) — model output for original images
        logits_aug1:  (B, C) — model output for first AugMix sample
        logits_aug2:  (B, C) — model output for second AugMix sample
        labels:       (B,)   — ground-truth class indices
        lam:          JSD loss weight (paper default = 12)

    Returns:
        scalar loss
    """
    p_orig = F.softmax(logits_orig, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)
    M = torch.clamp((p_orig + p_aug1 + p_aug2) / 3, min=1e-7, max=1).log() # avoid log(0) exploding
    JS = (F.kl_div(M, p_orig, reduction='batchmean') +
          F.kl_div(M, p_aug1, reduction='batchmean') +
          F.kl_div(M, p_aug2, reduction='batchmean')) / 3
    return F.cross_entropy(logits_orig, labels) + lam * JS


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Return dict of scalar metrics (loss, acc) for one epoch."""
    raise NotImplementedError


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return dict of scalar metrics (loss, acc) on loader."""
    raise NotImplementedError


def train(args: argparse.Namespace) -> None:
    """Full training run; saves checkpoint to args.output_dir."""
    raise NotImplementedError


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AugMix on CIFAR")
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--arch", default="wrn40_2",
                   choices=["wrn40_2", "resnext29_32x4", "densenet100", "allconvnet"])
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--augmix", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--jsd-lambda", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    train(get_args())
