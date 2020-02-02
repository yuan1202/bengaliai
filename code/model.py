# Based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential

import pretrainedmodels


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
    def __init__(self, out_dim, in_channels=1, model_name='se_resnext101_32x4d', use_bn=True, pretrained='imagenet'):
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