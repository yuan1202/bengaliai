from collections import OrderedDict

from model import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torchvision
from torchvision.models import densenet121, wide_resnet50_2, resnet34
from densenet_mod import _DenseBlock
from mish.mish import Mish
import pretrainedmodels
    
res34 = resnet34(pretrained=True)
seresxt50 = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
dense121 = densenet121(pretrained=True)
wideres50 = wide_resnet50_2(pretrained=True)

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
    
    
class mdl_res34_localpool_small(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        layer0_modules = [
            ('conv1', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        
        layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        # feature extraction
        # 98x168 -> 4x6; 64x112 -> 4x7
        base_lists = list(res34.children())[1:8]
        self.feature_extractor = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + base_lists[0:2] + base_lists[3:]))
        
        # 4x7 -> 2x5
        self.gpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        
        self.gcls = nn.Sequential(nn.Linear(5120, n_grapheme), nn.BatchNorm1d(n_grapheme))
        self.vcls = nn.Sequential(nn.Linear(5120, n_vowel), nn.BatchNorm1d(n_vowel))
        self.ccls = nn.Sequential(nn.Linear(5120, n_consonant), nn.BatchNorm1d(n_consonant))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = self.gpool(x).view(batch_size, 5120)
        #x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x)
    
    
class mdl_res34_localpool(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        layer0_modules = [
            ('conv1', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        
        layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        # feature extraction
        # 98x168 -> 4x6; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(*([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] + list(res34.children())[1:8]))
        
        # 6x6 -> 4x4
        self.gpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)
        
        self.gcls = nn.Sequential(nn.Linear(8192, n_grapheme), nn.BatchNorm1d(n_grapheme))
        self.vcls = nn.Sequential(nn.Linear(8192, n_vowel), nn.BatchNorm1d(n_vowel))
        self.ccls = nn.Sequential(nn.Linear(8192, n_consonant), nn.BatchNorm1d(n_consonant))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = self.gpool(x).view(batch_size, 8192)
        #x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x)
    

class mdl_ResDenHybrid(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(*list(res34.children())[:8] + [_DenseBlock(), nn.BatchNorm2d(640)])
        
        #self.gpool = GeM()
        
        self.gcls = nn.Linear(640, n_grapheme)
        self.vcls = nn.Linear(640, n_vowel)
        self.ccls = nn.Linear(640, n_consonant)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        #x = self.gpool(x).view(batch_size, 640)
        x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x)
    
    
class mdl_sext50_small(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        layer0_modules = [
            ('conv1', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
        ]
        
        layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        # feature extraction
        self.feature_extractor = nn.Sequential(
            layer0,
            seresxt50.layer1,
            seresxt50.layer2,
            seresxt50.layer3,
            seresxt50.layer4,
        )
        
        # global pooling
        self.gpool = GeM()
        
        self.gcls = nn.Sequential(nn.Linear(2048, n_grapheme), nn.BatchNorm1d(n_grapheme))
        self.vcls = nn.Sequential(nn.Linear(2048, n_vowel), nn.BatchNorm1d(n_vowel))
        self.ccls = nn.Sequential(nn.Linear(2048, n_consonant), nn.BatchNorm1d(n_consonant))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = self.gpool(x).view(batch_size, 2048)
        #x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x)

    
class mdl_sext50(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        layer0_modules = [
            ('conv1', nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]
        
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=True)))
        layer0 = nn.Sequential(OrderedDict(layer0_modules))
        
        # feature extraction
        self.feature_extractor = nn.Sequential(
            layer0,
            seresxt50.layer1,
            seresxt50.layer2,
            seresxt50.layer3,
            seresxt50.layer4,
        )
        
        # global pooling
        self.gpool = GeM()
        
        self.gcls = nn.Linear(2048, n_grapheme)
        self.vcls = nn.Linear(2048, n_vowel)
        self.ccls = nn.Linear(2048, n_consonant)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = self.gpool(x).view(batch_size, 2048)
        #x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x)
        
    
class mdl_res34Cmplx(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6 64x118 -> 2x4;
        # 98x168 -> 4x6
        self.feature_extractor = nn.Sequential(*list(res34.children())[:8])
        
        # grapheme root branch
        self.g_conv_s = nn.Sequential(
            # outputs 3x5
            nn.Conv2d(512, 256, kernel_size=2),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((3, 5)),
        )
        self.g_conv_m = nn.Sequential(
            # outputs 2x4
            nn.Conv2d(512, 256, kernel_size=3),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((2, 4)),
        )
        self.g_conv_l = nn.Sequential(
            # outputs 1x3
            nn.Conv2d(512, 256, kernel_size=4),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((1, 3)),
        )
        self.g_cls = nn.Sequential(nn.Linear(256, n_grapheme), nn.BatchNorm1d(n_grapheme))
        
        # vowel and consonant branch
        self.vc_conv_sq = nn.Sequential(
            # outputs 3x5
            nn.Conv2d(512, 256, kernel_size=2),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((3, 5)),
        )
        self.vc_conv_vs = nn.Sequential(
            # outputs 2x5
            nn.Conv2d(512, 256, kernel_size=(3, 2)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((2, 5)),
        )
        self.vc_conv_vl = nn.Sequential(
            # outputs 1x4
            nn.Conv2d(512, 256, kernel_size=(4, 3)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((1, 4)),
        )
        self.vc_conv_hs = nn.Sequential(
            # outputs 3x4
            nn.Conv2d(512, 256, kernel_size=(2, 3)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((3, 4)),
        )
        self.vc_conv_hl = nn.Sequential(
            # outputs 3x3
            nn.Conv2d(512, 256, kernel_size=(2, 4)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((3, 3)),
        )
        self.vc_cls = nn.Sequential(nn.Linear(256, n_vowel+n_consonant), nn.BatchNorm1d(n_vowel+n_consonant))
    
    def forward(self, x):
        x = self.feature_extractor(x)
        
        x_vc_sq = self.vc_conv_sq(x).view(-1, 256, 1)
        x_vc_vs = self.vc_conv_vs(x).view(-1, 256, 1)
        x_vc_vl = self.vc_conv_vl(x).view(-1, 256, 1)
        x_vc_hs = self.vc_conv_hs(x).view(-1, 256, 1)
        x_vc_hl = self.vc_conv_hl(x).view(-1, 256, 1)
        x_vc = torch.cat((x_vc_sq, x_vc_vs, x_vc_vl, x_vc_hs, x_vc_hl), -1).sum(-1)
        x_vc = self.vc_cls(x_vc)
        
        x_g_s = self.g_conv_s(x).view(-1, 256, 1)
        x_g_m = self.g_conv_m(x).view(-1, 256, 1)
        x_g_l = self.g_conv_l(x).view(-1, 256, 1)
        x_g = torch.cat((x_g_s, x_g_m, x_g_l), -1).sum(-1)
        x_g = self.g_cls(x_g)
        
        return (x_g,) + torch.split(x_vc, (self.n_vowel, self.n_consonant), dim=1)
    
    
class mdl_sext50_AdvLss(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            seresxt50.layer0,
            seresxt50.layer1,
            seresxt50.layer2,
            seresxt50.layer3,
            seresxt50.layer4,
            GeM(),
            nn.Flatten(),
        )
        self.loss_head = nn.Linear(2048, 512)
        self.gcls = nn.Linear(2048, n_grapheme)
        self.vcls = nn.Linear(2048, n_vowel)
        self.ccls = nn.Linear(2048, n_consonant)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = F.dropout(x, 0.1, self.training)
        return self.gcls(x), self.vcls(x), self.ccls(x), self.loss_head(x)

