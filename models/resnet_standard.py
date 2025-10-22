"""
Standard ResNet architectures for CIFAR-100.

Implements ResNet-20, 32, 44, 56, 110 following the CIFAR ResNet design:
- 3 stages with [16, 32, 64] filters
- BasicBlock with skip connections
- No MaxPooling (uses stride-2 convolutions for downsampling)
- Global Average Pooling + Linear classifier

Reference:
"Deep Residual Learning for Image Recognition" (He et al., 2015)
"Identity Mappings in Deep Residual Networks" (He et al., 2016)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNets.

    Structure: Conv-BN-ReLU-Conv-BN + skip connection
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_batchnorm=True):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes) if use_batchnorm else nn.Identity()
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-100.

    Architecture: conv1 -> stage1 -> stage2 -> stage3 -> GAP -> FC
    """
    def __init__(self, block, num_blocks, num_classes=100, use_batchnorm=True, use_dropout=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.use_batchnorm = use_batchnorm

        # Initial conv layer (no pooling for CIFAR)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64 * block.expansion, num_classes)
            )
        else:
            self.classifier = nn.Linear(64 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_batchnorm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def ResNet20(num_classes=100, use_batchnorm=True, use_dropout=False):
    """ResNet-20 for CIFAR (n=3, total layers = 6*3+2 = 20)"""
    return ResNet(BasicBlock, [3, 3, 3], num_classes, use_batchnorm, use_dropout)


def ResNet32(num_classes=100, use_batchnorm=True, use_dropout=False):
    """ResNet-32 for CIFAR (n=5, total layers = 6*5+2 = 32)"""
    return ResNet(BasicBlock, [5, 5, 5], num_classes, use_batchnorm, use_dropout)


def ResNet44(num_classes=100, use_batchnorm=True, use_dropout=False):
    """ResNet-44 for CIFAR (n=7, total layers = 6*7+2 = 44)"""
    return ResNet(BasicBlock, [7, 7, 7], num_classes, use_batchnorm, use_dropout)


def ResNet56(num_classes=100, use_batchnorm=True, use_dropout=False):
    """ResNet-56 for CIFAR (n=9, total layers = 6*9+2 = 56)"""
    return ResNet(BasicBlock, [9, 9, 9], num_classes, use_batchnorm, use_dropout)


def ResNet110(num_classes=100, use_batchnorm=True, use_dropout=False):
    """ResNet-110 for CIFAR (n=18, total layers = 6*18+2 = 110)"""
    return ResNet(BasicBlock, [18, 18, 18], num_classes, use_batchnorm, use_dropout)
