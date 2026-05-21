"""
Dataset helpers for CIFAR-10/100 and their -C / -P variants.

Downloads:
  CIFAR-10/100
  CIFAR-10-C/100-C 
  CIFAR-10-P/100-P 
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# Standard pre-processing applied before AugMix (paper §4.1)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def get_cifar_loaders(
    dataset: str,  # 'cifar10' or 'cifar100'
    data_root: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmix: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for clean CIFAR.

    AugMix is applied inside the training transform when use_augmix=True.
    Standard pre-processing: random flip + crop, then normalize.
    """
    raise NotImplementedError


class CIFAR_C(Dataset):
    """
    CIFAR-10-C or CIFAR-100-C corruption benchmark.

    data_root should contain the extracted .npy files from Zenodo.
    corruption: one of the 15 corruption names, or 'all' to iterate over all.
    severity: 1–5, or None for all severities.
    """

    CORRUPTIONS = [
        "brightness", "contrast", "defocus_blur", "elastic_transform",
        "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
        "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
        "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
    ]

    def __init__(
        self,
        data_root: str | Path,
        corruption: str,
        severity: int,
        transform=None,
    ):
        raise NotImplementedError


class CIFAR_P(Dataset):
    """
    CIFAR-10-P or CIFAR-100-P perturbation benchmark (video sequences).

    Used to compute mean Flip Probability (mFP).
    """

    PERTURBATIONS = [
        "brightness", "gaussian_noise", "motion_blur", "rotate",
        "scale", "shear", "shot_noise", "snow", "tilt", "translate",
        "zoom_blur",
    ]

    def __init__(self, data_root: str | Path, perturbation: str, transform=None):
        raise NotImplementedError
