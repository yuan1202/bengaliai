from model import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torchvision
from torchvision.models import densenet121, wide_resnet50_2, resnet34
from mish.mish import Mish
from senet_mod import se_resnext50_32x4d
    
base34 = resnet34()
base50 = se_resnext50_32x4d()

# ------------------------------------------------------------------------
def gem_spatial(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (2, 2), 2, padding=0).pow(1./p)


class GeM_Spatial(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM_Spatial, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem_spatial(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    

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
    
    
class Simple50(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
        )
        
        # global pooling
        self.gpoolmax = nn.AdaptiveMaxPool2d((1, 1))
        self.gpoolavg = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gcls = nn.Linear(2048, n_grapheme)
        self.vcls = nn.Linear(2048, n_vowel)
        self.ccls = nn.Linear(2048, n_consonant)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x_mp = self.gpoolmax(x)
        x_ap = self.gpoolavg(x)
        x = (x_mp + x_ap).view(-1, 2048)
        return self.gcls(x), self.vcls(x), self.ccls(x)
    
    
class Seresnext50AM(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
        )
        
        # global pooling
        self.gpoolmax = nn.AdaptiveMaxPool2d((1, 1))
        self.gpoolavg = nn.AdaptiveAvgPool2d((1, 1))
        
        self.gcls = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, n_grapheme),
        )
        self.vcls = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, n_vowel),
        )
        self.ccls = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, n_consonant),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x_mp = self.gpoolmax(x)
        x_ap = self.gpoolavg(x)
        x = (x_mp + x_ap).view(-1, 2048)
        return self.gcls(x), self.vcls(x), self.ccls(x)
    
    
class Res34ComplexWide(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor =  self.feature_extractor = nn.Sequential(*list(base34.children())[:8])
        
        # grapheme root branch
        self.g_conv_s = nn.Sequential(
            # outputs 3x3
            nn.Conv2d(512, 256, kernel_size=2),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d(3),
        )
        self.g_conv_m = nn.Sequential(
            # outputs 2x2
            nn.Conv2d(512, 256, kernel_size=3),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d(2),
        )
        self.g_conv_l = nn.Sequential(
            # outputs 1x1
            nn.Conv2d(512, 256, kernel_size=4),
            nn.PReLU(),
            nn.BatchNorm2d(256),
        )
        self.g_cls_vc_bn = nn.BatchNorm1d(n_vowel+n_consonant)
        self.g_cls = nn.Linear(256*3+n_vowel+n_consonant, n_grapheme)
        
        # vowel and consonant branch
        self.vc_conv_s = nn.Sequential(
            # outputs 3x3
            nn.Conv2d(512, 256, kernel_size=2),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d(3),
        )
        self.vc_conv_v = nn.Sequential(
            # outputs 1x3
            nn.Conv2d(512, 256, kernel_size=(4, 2)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((1, 3)),
        )
        self.vc_conv_h = nn.Sequential(
            # outputs 3x1
            nn.Conv2d(512, 256, kernel_size=(2, 4)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((3, 1)),
        )
        self.vc_cls = nn.Linear(256*3, n_vowel+n_consonant)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        
        x_vc_s = self.vc_conv_s(x).view(-1, 256)
        x_vc_v = self.vc_conv_v(x).view(-1, 256)
        x_vc_h = self.vc_conv_h(x).view(-1, 256)
        x_vc = torch.cat((x_vc_s, x_vc_v, x_vc_h), -1)
        x_vc = self.vc_cls(x_vc)
        
        x_vc_detach = self.g_cls_vc_bn(x_vc.detach())
        
        x_g_s = self.g_conv_s(x).view(-1, 256)
        x_g_m = self.g_conv_m(x).view(-1, 256)
        x_g_l = self.g_conv_l(x).view(-1, 256)
        x_g = torch.cat((x_g_s, x_g_m, x_g_l), -1)
        
        x_g = self.g_cls(torch.cat((x_g, x_vc_detach), -1))
        
        return (x_g,) + torch.split(x_vc, (self.n_vowel, self.n_consonant), dim=1)
    
    
class Res34Complex(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor =  self.feature_extractor = nn.Sequential(*list(base34.children())[:8])
        
        # grapheme root branch
        self.g_conv_s = nn.Sequential(
            # outputs 3x3
            nn.Conv2d(512, 256, kernel_size=2),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d(3),
        )
        self.g_conv_m = nn.Sequential(
            # outputs 2x2
            nn.Conv2d(512, 256, kernel_size=3),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d(2),
        )
        self.g_conv_l = nn.Sequential(
            # outputs 1x1
            nn.Conv2d(512, 256, kernel_size=4),
            nn.PReLU(),
            nn.BatchNorm2d(256),
        )
        #self.g_cls_vc_bn = nn.BatchNorm1d(n_vowel+n_consonant)
        self.g_cls = nn.Sequential(nn.Linear(256+n_vowel+n_consonant, n_grapheme), nn.BatchNorm1d(n_grapheme))
        #self.g_cls = nn.Linear(512+n_vowel+n_consonant, n_grapheme)
        
        # vowel and consonant branch
        self.vc_conv_s = nn.Sequential(
            # outputs 3x3
            nn.Conv2d(512, 256, kernel_size=2),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d(3),
        )
        self.vc_conv_vs = nn.Sequential(
            # outputs 2x4
            nn.Conv2d(512, 256, kernel_size=(3, 1)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((2, 4)),
        )
        self.vc_conv_vl = nn.Sequential(
            # outputs 1x3
            nn.Conv2d(512, 256, kernel_size=(4, 2)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((1, 3)),
        )
        self.vc_conv_hs = nn.Sequential(
            # outputs 3x2
            nn.Conv2d(512, 256, kernel_size=(1, 3)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # 1x1
            nn.AvgPool2d((4, 2)),
        )
        self.vc_conv_hl = nn.Sequential(
            # outputs 3x1
            nn.Conv2d(512, 256, kernel_size=(2, 4)),
            nn.PReLU(),
            nn.BatchNorm2d(256),
            # outputs 1x1
            nn.AvgPool2d((3, 1)),
        )
        self.vc_cls = nn.Sequential(nn.Linear(256, n_vowel+n_consonant), nn.BatchNorm1d(n_vowel+n_consonant))
        #self.vc_cls = nn.Linear(256, n_vowel+n_consonant)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        
        x_vc_s = self.vc_conv_s(x).view(-1, 256, 1)
        x_vc_vs = self.vc_conv_vs(x).view(-1, 256, 1)
        x_vc_hs = self.vc_conv_hs(x).view(-1, 256, 1)
        x_vc_vl = self.vc_conv_vl(x).view(-1, 256, 1)
        x_vc_hl = self.vc_conv_hl(x).view(-1, 256, 1)
        x_vc = torch.cat((x_vc_s, x_vc_vs, x_vc_hs, x_vc_vl, x_vc_hl), -1).sum(-1)
        x_vc = self.vc_cls(x_vc)
        
        x_vc_detach = x_vc.detach()
        
        x_g_s = self.g_conv_s(x).view(-1, 256, 1)
        x_g_m = self.g_conv_m(x).view(-1, 256, 1)
        x_g_l = self.g_conv_l(x).view(-1, 256, 1)
        x_g = torch.cat((x_g_s, x_g_m, x_g_l), -1).sum(-1)
        x_g = self.g_cls(torch.cat((x_g, x_vc_detach), -1))
        
        return (x_g,) + torch.split(x_vc, (self.n_vowel, self.n_consonant), dim=1)
    
    
class Seresnext50MishSumSpatial(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
            nn.MaxPool2d(2, stride=1),
        )
        self.g_cls = nn.Sequential(nn.Linear(2048, n_grapheme), nn.BatchNorm1d(n_grapheme))
        self.vc_cls = nn.Sequential(nn.Linear(2048, n_vowel+n_consonant), nn.BatchNorm1d(n_vowel+n_consonant))
        #self.vc_cls = nn.Sequential(
        #    nn.Linear(2048, 512),
        #    nn.BatchNorm1d(512),
        #    Mish(),
        #    nn.Linear(512, n_vowel+n_consonant),
        #    nn.BatchNorm1d(n_vowel+n_consonant),
        #)
        
        self.g_slice = torch.Tensor([1, 3, 4, 5, 6, 7, 8]).long()
        self.vc_slice = torch.Tensor([0, 1, 2, 3, 5, 6, 7, 8]).long()
    
    def forward(self, x):
        # 16 for 128x128
        x = self.feature_extractor(x).view(-1, 2048, 9)
        x_g = torch.mean(x[:, :, self.g_slice], -1)
        x_vc = torch.mean(x[:, :, self.vc_slice], -1)
        return (self.g_cls(x_g),) + torch.split(self.vc_cls(x_vc), [self.n_vowel, self.n_consonant], dim=1)
    
    
class Seresnext50MishGeM(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
            GeM(),
            nn.Flatten(),
        )
        self.cls = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Linear(512, n_grapheme+n_vowel+n_consonant),
            nn.BatchNorm1d(n_grapheme+n_vowel+n_consonant),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cls(x)
        return torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
    
    
class Seresnext50MishGeM_AdvLss(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
            GeM(),
            nn.Flatten(),
        )
        self.lss_head = nn.Linear(2048, 256)
        self.cls = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            Mish(),
            nn.Linear(1024, n_grapheme+n_vowel+n_consonant),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        feat_loss = self.lss_head(x)
        cls = self.cls(x)
        return torch.split(cls, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1) + (feat_loss,)
    
    
class Seresnext50MishGeM_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
        )
        self.neck = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=0, padding=0),
            nn.Flatten(),
            nn.Linear(2048, 1024),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.neck(x)
        return x
    
    
class Simple34(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(*list(base34.children())[:8])
        
        # 512x2x2
        self.gpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 512x3x3
        self.gpool3x3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6656, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_grapheme+n_vowel+n_consonant),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x_fc = self.gpool2x2(x).view(-1, 2048)
        x_gp = self.gpool3x3(x).view(-1, 4608)
        x = self.cls(torch.cat((x_fc, x_gp), dim=1))
        return torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
    
    
class Seresnext50MishFrac(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base50.layer0,
            base50.layer1,
            base50.layer2,
            base50.layer3,
            base50.layer4,
            nn.FractionalMaxPool2d(3, output_size=(2, 2)),
            nn.Flatten(),
        )
        self.cls = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.BatchNorm1d(1024),
            Mish(),
            nn.Linear(1024, n_grapheme+n_vowel+n_consonant),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.cls(x)
        return torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)