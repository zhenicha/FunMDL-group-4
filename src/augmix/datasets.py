"""
Dataset helpers for CIFAR-10/100 and their -C / -P variants.

Downloads:
  CIFAR-10/100
  CIFAR-10-C/100-C 
  CIFAR-10-P/100-P 
"""

import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .augmentations import augment_and_mix

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

def get_normalization(dataset: str):
    if dataset == 'cifar100':
        return CIFAR100_MEAN, CIFAR100_STD
    return CIFAR10_MEAN, CIFAR10_STD


class AugMixDataset(Dataset):
    """Dataset wrapper to perform AugMix augmentation and return a 3-tuple."""
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __getitem__(self, i):
        # x is a PIL image with base transforms already applied
        x, y = self.dataset[i]
        
        x_aug1 = augment_and_mix(x)
        x_aug2 = augment_and_mix(x)
        
        im_tuple = (self.preprocess(x), self.preprocess(x_aug1), self.preprocess(x_aug2))
        return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def get_cifar_loaders(
    dataset: str, 
    data_root: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    use_augmix: bool = True,
) -> tuple[DataLoader, DataLoader]:
    
    data_root = Path(data_root)

    # Base transforms applied before AugMix (4.1)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4)
    ])
    
    mean, std = get_normalization(dataset)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    DatasetClass = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    
    train_data = DatasetClass(data_root, train=True, transform=train_transform, download=True)
    test_data = DatasetClass(data_root, train=False, transform=preprocess, download=True)

    if use_augmix:
        train_data = AugMixDataset(train_data, preprocess)
    else:
        train_data.transform = transforms.Compose([train_transform, preprocess])

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_loader


class CIFAR_C(Dataset):
    """CIFAR-10-C or CIFAR-100-C corruption benchmark."""
    CORRUPTIONS = [
        "brightness", "contrast", "defocus_blur", "elastic_transform",
        "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
        "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
        "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
    ]

    def __init__(self, data_root: str | Path, corruption: str, severity: int = None, transform=None, dataset: str = 'cifar10'):
        data_root = Path(data_root)
        assert corruption in self.CORRUPTIONS or corruption == 'all'
        mean, std = get_normalization(dataset)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        data_path = data_root / f"{corruption}.npy"
        label_path = data_root / "labels.npy"
        
        self.data = np.load(data_path)
        self.targets = np.load(label_path)

        # Zenodo structure: 50,000 images per file, 10,000 per severity (1-5)
        if severity is not None:
            assert 1 <= severity <= 5
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            self.data = self.data[start_idx:end_idx]
            self.targets = self.targets[start_idx:end_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, target = self.data[i], self.targets[i]
        img = Image.fromarray(img) #back to PIL 
        if self.transform:
            img = self.transform(img)
        return img, target