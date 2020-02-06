# Based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential

import pretrainedmodels

from fastai.vision import *

from mish import Mish

# -----------------------------------------------------------------
class LinearBlock(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False
    ):
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h
    
    
class PretrainedCNN(nn.Module):
    def __init__(self, out_dim, in_channels=1, model_name='se_resnext50_32x4d', use_bn=True, pretrained='imagenet'):
        super(PretrainedCNN, self).__init__()
        
        # convert channels to 3 to adapt to pre-trained model
        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        
        activation = F.leaky_relu
        
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h
    

class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

    def forward(self, x, y=None):
        pred = self.predictor(x)
        preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds
    

# -----------------------------------------------------------------
nunique = [168, 11, 7]
arch = models.densenet121


class Head(nn.Module):
    def __init__(self, nc, n, ps=0.5):
        super().__init__()
        layers = [AdaptiveConcatPool2d(), Mish(), Flatten()] + \
            bn_drop_lin(nc*2, 512, True, ps, Mish()) + \
            bn_drop_lin(512, n, True, ps)
        self.fc = nn.Sequential(*layers)
        self._init_weight()
        
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
        
    def forward(self, x):
        return self.fc(x)

    
#change the first conv to accept 1 chanel input
class Dnet_1ch(nn.Module):
    def __init__(self, arch=arch, n=nunique, pre=True, ps=0.5):
        super().__init__()
        m = arch(True) if pre else arch()
        
        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        w = (m.features.conv0.weight.sum(1)).unsqueeze(1)
        conv.weight = nn.Parameter(w)
        
        self.layer0 = nn.Sequential(conv, m.features.norm0, nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            m.features.denseblock1)
        self.layer2 = nn.Sequential(m.features.transition1,m.features.denseblock2)
        self.layer3 = nn.Sequential(m.features.transition2,m.features.denseblock3)
        self.layer4 = nn.Sequential(m.features.transition3,m.features.denseblock4,
                                    m.features.norm5)
        
        nc = self.layer4[-1].weight.shape[0]
        self.head1 = Head(nc,n[0])
        self.head2 = Head(nc,n[1])
        self.head3 = Head(nc,n[2])
        #to_Mish(self.layer0), to_Mish(self.layer1), to_Mish(self.layer2)
        #to_Mish(self.layer3), to_Mish(self.layer4)
        
    def forward(self, x):    
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)
        
        return x1,x2,x3