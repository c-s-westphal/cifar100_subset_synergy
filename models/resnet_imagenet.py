"""
ImageNet-style ResNet models (10, 18, 34, 50, 68) adapted for CIFAR-100.

Based on "Deep Residual Learning for Image Recognition" (He et al., 2015)
Adapted for CIFAR-100 (32x32 images) by:
- Using 3x3 conv (stride 1) instead of 7x7 conv (stride 2)
- Removing initial max pooling layer
- Using standard ImageNet width: [64, 128, 256, 512]
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic ResNet block with two 3x3 convolutions.

    Used in ResNet-10, 18, 34.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block with 1x1 -> 3x3 -> 1x1 convolutions.

    Used in ResNet-50, 68, 101, 152.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model adapted for CIFAR-100.

    Adaptations from ImageNet version:
    - 3x3 conv with stride 1 (instead of 7x7 stride 2)
    - No max pooling after first conv
    - Standard ImageNet width: [64, 128, 256, 512]
    """
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR adaptation: 3x3 conv, stride 1, no max pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet10(num_classes=100):
    """ResNet-10: [1, 1, 1, 1] basic blocks.

    Ultra-shallow ResNet for testing minimal viable depth.
    """
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes)


def ResNet18(num_classes=100):
    """ResNet-18: [2, 2, 2, 2] basic blocks.

    Standard ImageNet ResNet-18 adapted for CIFAR.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=100):
    """ResNet-34: [3, 4, 6, 3] basic blocks.

    Standard ImageNet ResNet-34 adapted for CIFAR.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=100):
    """ResNet-50: [3, 4, 6, 3] bottleneck blocks.

    Standard ImageNet ResNet-50 adapted for CIFAR.
    Uses bottleneck blocks (1x1->3x3->1x1) for efficiency.
    """
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet68(num_classes=100):
    """ResNet-68: [3, 4, 12, 3] bottleneck blocks.

    Novel configuration between ResNet-50 and ResNet-101.
    Adds more capacity in stage 3 (most important stage).
    """
    return ResNet(Bottleneck, [3, 4, 12, 3], num_classes)


if __name__ == '__main__':
    # Test model creation
    for name, model_fn in [('ResNet10', ResNet10), ('ResNet18', ResNet18),
                            ('ResNet34', ResNet34), ('ResNet50', ResNet50),
                            ('ResNet68', ResNet68)]:
        print(f"\n{name}:")
        model = model_fn(num_classes=100)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        assert y.shape == (1, 100), f"Expected shape (1, 100), got {y.shape}"
        print(f"  Forward pass: OK (output shape: {y.shape})")
