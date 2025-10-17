"""
VGG models (9, 11, 13, 16, 19) for CIFAR-100 with Global Average Pooling.

Based on the VGG paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
Adapted for CIFAR-100 (32x32 images) with GAP instead of traditional FC classifier.
"""

import torch
import torch.nn as nn


# VGG configurations: number of output channels for each layer
# 'M' denotes max pooling
cfg = {
    'vgg9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """VGG model for CIFAR-100 with Global Average Pooling.

    Args:
        arch: Architecture name ('vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19')
        num_classes: Number of output classes (default: 100 for CIFAR-100)
        use_batchnorm: Whether to use batch normalization (default: True)
        use_dropout: Whether to use dropout before final linear layer (default: True)
    """
    def __init__(self, arch: str, num_classes: int = 100, use_batchnorm: bool = True, use_dropout: bool = True):
        super(VGG, self).__init__()
        self.arch = arch
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        self.features = self._make_layers(cfg[arch], use_batchnorm)

        # Global Average Pooling + Dropout + Linear classifier
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP: (B, 512, H, W) -> (B, 512, 1, 1)

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layers(self, cfg_list, use_batchnorm):
        """Create convolutional layers based on configuration."""
        layers = []
        in_channels = 3

        for v in cfg_list:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if use_batchnorm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)      # Conv features
        x = self.pool(x)          # (B, 512, 1, 1)
        x = torch.flatten(x, 1)   # (B, 512)
        x = self.classifier(x)    # (B, num_classes)
        return x


def VGG9(num_classes=100, use_batchnorm=True, use_dropout=True):
    """VGG 9-layer model with GAP."""
    return VGG('vgg9', num_classes, use_batchnorm, use_dropout)


def VGG11(num_classes=100, use_batchnorm=True, use_dropout=True):
    """VGG 11-layer model with GAP."""
    return VGG('vgg11', num_classes, use_batchnorm, use_dropout)


def VGG13(num_classes=100, use_batchnorm=True, use_dropout=True):
    """VGG 13-layer model with GAP."""
    return VGG('vgg13', num_classes, use_batchnorm, use_dropout)


def VGG16(num_classes=100, use_batchnorm=True, use_dropout=True):
    """VGG 16-layer model with GAP."""
    return VGG('vgg16', num_classes, use_batchnorm, use_dropout)


def VGG19(num_classes=100, use_batchnorm=True, use_dropout=True):
    """VGG 19-layer model with GAP."""
    return VGG('vgg19', num_classes, use_batchnorm, use_dropout)


if __name__ == '__main__':
    # Test model creation
    for arch_name in ['vgg9', 'vgg11', 'vgg13', 'vgg16', 'vgg19']:
        print(f"\n{arch_name.upper()}:")

        # With batch norm and dropout
        model_bn = VGG(arch_name, num_classes=100, use_batchnorm=True, use_dropout=True)
        total_params = sum(p.numel() for p in model_bn.parameters())
        print(f"  With BN+Dropout: {total_params:,} parameters")

        # Without batch norm, without dropout
        model_no_bn = VGG(arch_name, num_classes=100, use_batchnorm=False, use_dropout=False)
        total_params = sum(p.numel() for p in model_no_bn.parameters())
        print(f"  Without BN/Dropout: {total_params:,} parameters")

        # Test forward pass
        x = torch.randn(1, 3, 32, 32)
        y = model_bn(x)
        assert y.shape == (1, 100), f"Expected shape (1, 100), got {y.shape}"
        print(f"  Forward pass: OK (output shape: {y.shape})")
