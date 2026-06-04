"""
AugMix augmentation operations and the AugmentAndMix procedure.

Operations are from AutoAugment, excluding contrast/color/brightness/
sharpness/Cutout to keep disjoint from ImageNet-C corruptions.
"""

import numpy as np
from PIL import Image, ImageOps
from typing import Callable

AUGMENTATIONS: list[Callable] = [
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    rotate,
    solarize,
    posterize,
    lambda image, smin, smax : autocontrast(image),
    lambda image, smin, smax : invert(image),
    lambda image, smin, smax : equalize(image),
]

# Excluded to stay disjoint from ImageNet-C test corruptions:
# contrast, color, brightness, sharpness, cutout


def augment(image: Image.Image, severity_min: float, severity_max: float) -> Image.Image:
    """
    Apply a single random augmentation operation to the given image.
    The severity of the augmentation is uniformy sampled from the interval [severity_min, severity_max], 
    which should be a sub-interval of [0, 1].
    """
    augment = np.random.choice(AUGMENTATIONS)
    return augment(image, severity_min, severity_max)


def augment_and_mix(image: Image.Image, k: int = 3, alpha: float = 1.0, severity_min: float = 0.0, severity_max: float = 1.0) -> Image.Image:
    """
    Algorithm 1 from the paper.

    Args:
        image: PIL image (original, pre-processed with standard flip/crop)
        k: number of augmentation chains (default 3)
        alpha: Dirichlet / Beta concentration parameter (default 1)

    Returns:
        x_augmix: mixed image as PIL Image
    """

    weights = np.random.dirichlet(np.repeat(alpha, k))
    m = np.random.beta(alpha, alpha)
    augmented_image = Image.new(image.mode, image.size)

    for i in range(k):
        new_image = image.copy()
        depth = np.random.choice([1, 2, 3])

        for _ in range(depth):
            new_image = augment(new_image, severity_min, severity_max)

        augmented_image = augmented_image + weights[i] * new_image

    return m * image + (1 - m) * augmented_image



def shear_x(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Shear the image horizontally by a given rate.
    The shear rate should be in the range of [-0.3, 0.3] and is calculated using the augmentation severity levels.
    """
    rate = random_severity(severity_min, severity_max) * 0.6 - 0.3
    new_image = image.transform(image.size, Image.AFFINE, (1, rate, 0, 0, 1, 0))
    return new_image


def shear_y(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Shear the image vertically by a given rate.
    The shear rate should be in the range of [-0.3, 0.3] and is calculated using the augmentation severity levels.
    """
    rate = random_severity(severity_min, severity_max) * 0.6 - 0.3
    new_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, rate, 1, 0))
    return new_image


def translate_x(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Translate the image horizontally by a given number of pixels.
    The number of pixels should be in the range of [-150, 150] and is calculated using the augmentation severity levels.
    """
    shift = np.floor(random_severity(severity_min, severity_max) * 301 - 150)
    new_image = image.transform(image.size, Image.AFFINE, (1, 0, shift, 0, 1, 0))
    return new_image


def translate_y(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Translate the image vertically by a given number of pixels.
    The number of pixels should be in the range of [-150, 150] and is calculated using the augmentation severity levels.
    """
    shift = np.floor(random_severity(severity_min, severity_max) * 301 - 150)
    new_image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, shift))
    return new_image


def rotate(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Rotate the image by a given number of degrees counter clockwise around its centre.
    The number of degrees should be in the range of [-30, 30] and is calculated using the augmentation severity levels.
    """
    angle = np.floor(random_severity(severity_min, severity_max) * 61 - 30)
    new_image = image.rotate(angle)
    return new_image


def solarize(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Invert all pixel values above a certain threshold.
    The number of pixels should be in the range of [0, 256] and the exact value is calculated using the augmentation severity levels.
    """
    threshold = np.floor(random_severity(severity_min, severity_max) * 257)
    new_image = ImageOps.solarize(image, threshold)
    return new_image
    

def posterize(image: Image.Image, severity_min: float, severity_max: float) ->  Image.Image:
    """
    Reduce the number of bits for each color channel.
    The numer of bits to keep per channel should be in the range of [4, 8] and the exact value is calculated using the augmentation severity levels.
    """
    bits = np.floor(random_severity(severity_min, severity_max) * 5 + 4)
    new_image = ImageOps.solarize(image, bits)
    return new_image


def autocontrast(image: Image.Image) ->  Image.Image:
    """
    Maximize image contrast by stretching the image histogram.
    """
    new_image = ImageOps.autocontrast(image)
    return new_image

    
def invert(image: Image.Image) ->  Image.Image:
    """
    Invert the pixel values in the image.
    """
    new_image = ImageOps.invert(image)
    return new_image


def equalize(image: Image.Image) ->  Image.Image:
    """
    Equalize the image histogram by creating a uniform distribution of the pixel values.
    """
    new_image = ImageOps.equalize(image)
    return new_image


def random_severity(severity_min: float, severity_max: float) -> float:
    """
    Helper method to generte a random severity score in the given [severity_min, severity_max] interval.
    """
    low, high = sorted((severity_min, severity_max))
    return np.random.uniform(low, high)
