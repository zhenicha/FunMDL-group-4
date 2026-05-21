"""
AugMix augmentation operations and the AugmentAndMix procedure.

Operations are from AutoAugment, excluding contrast/color/brightness/
sharpness/Cutout to keep disjoint from ImageNet-C corruptions.
"""

import numpy as np
from PIL import Image

AUGMENTATIONS: list[str] = [
    "autocontrast",
    "equalize",
    "invert",
    "posterize",
    "rotate",
    "shear_x",
    "shear_y",
    "solarize",
    "translate_x",
    "translate_y",
]

# Excluded to stay disjoint from ImageNet-C test corruptions:
# contrast, color, brightness, sharpness, cutout


def _apply_op(image: Image.Image, op_name: str, severity: float) -> Image.Image:
    """Apply a single augmentation operation at a given severity in [0, 1]."""
    raise NotImplementedError(f"op '{op_name}' not yet implemented")


def augment_and_mix(image: Image.Image, k: int = 3, alpha: float = 1.0) -> Image.Image:
    """
    Algorithm 1 from the paper.

    Args:
        image: PIL image (original, pre-processed with standard flip/crop)
        k: number of augmentation chains (default 3)
        alpha: Dirichlet / Beta concentration parameter (default 1)

    Returns:
        x_augmix: mixed image as PIL Image
    """
    raise NotImplementedError("augment_and_mix not yet implemented")
