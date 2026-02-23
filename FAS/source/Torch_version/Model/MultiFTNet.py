# -*- coding: utf-8 -*-
# @Time : 20-6-3 下午5:14
# @Author : zhuying
# @Company : Minivision
# @File : MultiFTNet.py
# @Software : PyCharm
import torch
from torch import nn
import torch.nn.functional as F
from Model.mnext import MXNet


class FTGenerator(nn.Module):
    """Generates Fourier feature map from intermediate backbone features."""

    def __init__(self, in_channels=128, out_channels=1):
        super(FTGenerator, self).__init__()
        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    """Multi-task FAS model.

    Architecture:
        Input → early_features (128ch) → late_features (512ch) → classification
                        ↓
                   FTGenerator → Fourier feature map

    Args:
        num_classes: number of output classes (2 for CASIA-FASD: real/fake)
        img_channel: input image channels
        embedding_size: FC embedding dimension before classifier
        ft_size: (H, W) of output Fourier feature map. None = no resize.
    """

    def __init__(self, num_classes=2, img_channel=3, embedding_size=128, ft_size=None, **kwargs):
        super(MultiFTNet, self).__init__()
        self.num_classes = num_classes
        self.ft_size = ft_size

        # Build MobileNeXt backbone
        backbone = MXNet(num_classes=num_classes, in_channels=img_channel)

        # Split features at 128-channel boundary
        features = list(backbone.features.children())
        self.early_features = nn.Sequential(*features[:6])   # output: 128 channels
        self.late_features = nn.Sequential(*features[6:])     # output: 512 channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=0.2)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

        # Fourier feature map head
        self.FTGenerator = FTGenerator(in_channels=128, out_channels=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Early features → 128 channels
        x_early = self.early_features(x)

        # Late features → classification
        x_late = self.late_features(x_early)
        x_late = self.avgpool(x_late)
        x_late = x_late.view(x_late.size(0), -1)
        x_late = self.linear(x_late)
        x_late = self.bn(x_late)
        x_late = self.drop(x_late)
        cls = self.prob(x_late)

        if self.training:
            ft = self.FTGenerator(x_early)
            # Resize FT map to target size if specified
            if self.ft_size is not None:
                ft = F.interpolate(ft, size=self.ft_size, mode='bilinear', align_corners=False)
            return cls, ft
        else:
            return cls
