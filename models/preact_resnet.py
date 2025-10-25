"""
PreAct ResNet architectures for CIFAR-10.

Implements PreAct ResNet-20, 32, 44, 56, 68, 80, 92, 110 following "Identity Mappings in Deep
Residual Networks" (He et al., 2016). Uses pre-activation: BN->ReLU->Conv instead
of Conv->BN->ReLU for better gradient flow.

Architecture:
- 3 stages with [16, 32, 64] filters
- PreActBlock with pre-activation and skip connections
- No MaxPooling (uses stride-2 convolutions for downsampling)
- Global Average Pooling + Linear classifier

Reference:
"Identity Mappings in Deep Residual Networks" (He et al., 2016)
"""

import torch
import torch.nn as nn


class PreActBlock(nn.Module):
    """PreActivation residual block for CIFAR.

    Structure: BN-ReLU-Conv-BN-ReLU-Conv + skip connection
    The activation happens BEFORE the weight layers.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Shortcut connection
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    """PreAct ResNet for CIFAR-10.

    Architecture: conv1 -> stage1 -> stage2 -> stage3 -> BN -> ReLU -> GAP -> FC
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 16

        # Initial conv layer (no pooling for CIFAR)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        # Residual stages
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # Final batch norm and activation
        self.bn = nn.BatchNorm2d(64 * block.expansion)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.relu(self.bn(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def PreActResNet20(num_classes=10):
    """PreAct ResNet-20 for CIFAR (n=3, total layers = 6*3+2 = 20)"""
    return PreActResNet(PreActBlock, [3, 3, 3], num_classes)


def PreActResNet32(num_classes=10):
    """PreAct ResNet-32 for CIFAR (n=5, total layers = 6*5+2 = 32)"""
    return PreActResNet(PreActBlock, [5, 5, 5], num_classes)


def PreActResNet44(num_classes=10):
    """PreAct ResNet-44 for CIFAR (n=7, total layers = 6*7+2 = 44)"""
    return PreActResNet(PreActBlock, [7, 7, 7], num_classes)


def PreActResNet56(num_classes=10):
    """PreAct ResNet-56 for CIFAR (n=9, total layers = 6*9+2 = 56)"""
    return PreActResNet(PreActBlock, [9, 9, 9], num_classes)


def PreActResNet68(num_classes=10):
    """PreAct ResNet-68 for CIFAR (n=11, total layers = 6*11+2 = 68)"""
    return PreActResNet(PreActBlock, [11, 11, 11], num_classes)


def PreActResNet80(num_classes=10):
    """PreAct ResNet-80 for CIFAR (n=13, total layers = 6*13+2 = 80)"""
    return PreActResNet(PreActBlock, [13, 13, 13], num_classes)


def PreActResNet92(num_classes=10):
    """PreAct ResNet-92 for CIFAR (n=15, total layers = 6*15+2 = 92)"""
    return PreActResNet(PreActBlock, [15, 15, 15], num_classes)


def PreActResNet110(num_classes=10):
    """PreAct ResNet-110 for CIFAR (n=18, total layers = 6*18+2 = 110)"""
    return PreActResNet(PreActBlock, [18, 18, 18], num_classes)


if __name__ == '__main__':
    # Test model creation
    for name, model_fn in [('PreActResNet20', PreActResNet20),
                            ('PreActResNet32', PreActResNet32),
                            ('PreActResNet44', PreActResNet44),
                            ('PreActResNet56', PreActResNet56),
                            ('PreActResNet68', PreActResNet68),
                            ('PreActResNet80', PreActResNet80),
                            ('PreActResNet92', PreActResNet92),
                            ('PreActResNet110', PreActResNet110)]:
        print(f"\n{name}:")
        model = model_fn(num_classes=10)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 10), f"Expected shape (1, 10), got {y.shape}"
        print(f"  Forward pass: OK (output shape: {y.shape})")
