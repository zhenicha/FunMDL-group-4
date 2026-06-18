"""
Model definitions used in the paper.

Paper 4.1 uses: AllConvNet, DenseNet-BC (k=12, d=100), WideResNet 40-2, ResNeXt-29 (32x4)
Paper 4.2 uses: ResNet-50 (ImageNet)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate
        self.is_in_equal_out = (in_planes == out_planes)
        self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.is_in_equal_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        
        if self.is_in_equal_out:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
            
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        
        if not self.is_in_equal_out:
            return torch.add(self.conv_shortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0):
        super(WideResNet, self).__init__()
        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.n_channels)
        return self.fc(out)


# AllConvNet 
class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'Md':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=8)]
        elif v == 'NIN':
            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)
            layers += [conv2d, nn.BatchNorm2d(in_channels), GELU()]
        elif v == 'nopad':
            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
            layers += [conv2d, nn.BatchNorm2d(in_channels), GELU()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), GELU()]
            in_channels = v
    return nn.Sequential(*layers)

class AllConvNet(nn.Module):
    def __init__(self, num_classes):
        super(AllConvNet, self).__init__()
        self.num_classes = num_classes
        self.width1, w1 = 96, 96
        self.width2, w2 = 192, 192

        self.features = make_layers([w1, w1, w1, 'Md', w2, w2, w2, 'Md', 'nopad', 'NIN', 'NIN', 'A'])
        self.classifier = nn.Linear(self.width2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# DenseNet-BC Implementation (k=12, d=100)
# ==========================================
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([out, x], 1)

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return F.avg_pool2d(out, 2)

class DenseNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12, reduction=0.5):
        super(DenseNet, self).__init__()
        num_blocks = (depth - 4) // 6
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(Bottleneck, num_planes, num_blocks, growth_rate)
        num_planes += num_blocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(Bottleneck, num_planes, num_blocks, growth_rate)
        num_planes += num_blocks * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(Bottleneck, num_planes, num_blocks, growth_rate)
        num_planes += num_blocks * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock, growth_rate):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_planes, growth_rate))
            in_planes += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


# ----- ResNeXt-29 (32x4) -----

class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes: int, planes: int, cardinality: int, base_width: int, stride: int = 1):
        super().__init__()

        width = int(planes * (base_width / 64.0)) * cardinality

        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    def __init__(self, num_blocks: int = 3, cardinality: int = 32, base_width: int = 4, num_classes: int = 10):
        super().__init__()

        self.cardinality = cardinality
        self.base_width = base_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, num_blocks, stride=1)
        self.layer2 = self._make_layer(128, num_blocks, stride=2)
        self.layer3 = self._make_layer(256, num_blocks, stride=2)

        self.classifier = nn.Linear(256 * ResNeXtBottleneck.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(ResNeXtBottleneck(self.in_planes, planes, self.cardinality, self.base_width, s))
            self.in_planes = planes * ResNeXtBottleneck.expansion

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


# Model Retrieval
def get_model(arch: str, num_classes: int) -> nn.Module:
    """Return an untrained model by architecture name."""
    if arch == 'wrn40_2':
        return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, drop_rate=0.0)
    elif arch == 'allconvnet':
        return AllConvNet(num_classes=num_classes)
    elif arch == 'densenet100':
        return DenseNet(depth=100, num_classes=num_classes, growth_rate=12)
    elif arch == 'resnext29_32x4':
        return ResNeXt(num_blocks=3, cardinality=32, base_width=4, num_classes=num_classes)
    elif arch == 'resnet50':
        raise NotImplementedError("ResNet-50 not yet linked. Use torchvision.models.resnet50.")
    else:
        raise ValueError(f"Unknown architecture: {arch}")
