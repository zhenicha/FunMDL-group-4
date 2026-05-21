"""
Model definitions used in the paper.

Paper 4.1 uses: AllConvNet, DenseNet-BC (k=12, d=100), WideResNet 40-2, ResNeXt-29 (32x4)
Paper 4.2 uses: ResNet-50 (ImageNet)

"""

import torch.nn as nn


def get_model(arch: str, num_classes: int) -> nn.Module:
    """
    Return an untrained model by architecture name.

    Supported arches: 'wrn40_2', 'resnext29_32x4', 'densenet100', 'allconvnet', 'resnet50'
    """
    raise NotImplementedError(f"arch '{arch}' not yet implemented")
