from model import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from mish.mish import Mish
from senet_mod import SEResNeXtBottleneck
from densenet_mod import _Transition, _DenseBlock
    
# ------------------------------------------------------------------------
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
    
class seresnext_densenet(nn.Module):
    def __init__(
        self,
        n_grapheme=168, n_vowel=11, n_consonant=7,
        dropout=0.1,
        inplanes=128, 
        downsample_kernel_size=3,
        downsample_padding=1,
    ):
        super().__init__()
        
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        self.inplanes = inplanes
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(inplanes),
            Mish(),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        
        self.layer1 = self._make_layer(
            SEResNeXtBottleneck,
            planes=64,
            blocks=3,
            groups=32,
            reduction=16,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            SEResNeXtBottleneck,
            planes=128,
            blocks=4,
            stride=2,
            groups=32,
            reduction=16,
            downsample_kernel_size=3,
            downsample_padding=1
        )
        self.layer3 = self._make_layer(
            SEResNeXtBottleneck,
            planes=256,
            blocks=6,
            stride=2,
            groups=32,
            reduction=16,
            downsample_kernel_size=3,
            downsample_padding=1
        )
        self.transition = _Transition(num_input_features=1024, num_output_features=640)
        self.layer4 = _DenseBlock(
            num_layers=32,
            num_input_features=640,
            bn_size=4,
            growth_rate=32,
            drop_rate=dropout,
            memory_efficient=False,
        )
        self.feature_pool = GeM()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.last_linear = nn.Linear(1664, n_grapheme+n_vowel+n_consonant)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)
    
    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.transition(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.feature_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)