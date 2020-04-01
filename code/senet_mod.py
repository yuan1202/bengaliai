
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.GELU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.GELU()
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENetMod(nn.Module):

    def __init__(
        self, n_grapheme=168, n_vowel=11, n_consonant=7,
        block=SEResNeXtBottleneck, layers=(3, 4, 6, 3), groups=32, reduction=16, inplanes=64, downsample_kernel_size=1, downsample_padding=0
    ):
        super().__init__()
        
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        self.inplanes = inplanes
        
        layer0_modules = [
            ('conv1', nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            #('relu1', nn.GELU()),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            #('relu2', nn.GELU()),
            ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(inplanes)),
            ('relu3', nn.ReLU(inplace=True)),
            #('relu3', nn.GELU()),
        ]
        
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        
        self.features = nn.Sequential(
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )
        
        self.gcls = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Linear(512, n_grapheme, bias=True),
            nn.BatchNorm1d(n_grapheme),
        )

        self.vcls = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Linear(512, n_vowel, bias=True),
            nn.BatchNorm1d(n_vowel),
        )
        
        self.ccls = nn.Sequential(
            nn.Linear(2048, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            #nn.GELU(),
            nn.Linear(512, n_consonant, bias=True),
            nn.BatchNorm1d(n_consonant),
        )

    def _make_layer(
        self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    padding=downsample_padding,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
                #nn.Conv2d(
                #    self.inplanes, planes * block.expansion,
                #    kernel_size=downsample_kernel_size, stride=stride,
                #    padding=downsample_padding, bias=False
                #),
                #nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x)
