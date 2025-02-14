from model import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torchvision
from torchvision.models import densenet121, wide_resnet50_2
from mish.mish import Mish
from senet_mod import se_resnext50_32x4d
    
    
base = se_resnext50_32x4d()

# ------------------------------------------------------------------------
def gem_spatial(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (2, 2), 2, padding=1).pow(1./p)


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
    

class SERESXT50MishGeM_Embedding(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=0.):
        super().__init__()
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base.layer0,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.transit = nn.Sequential(
            GeM(),
            nn.Flatten(),
        )
        # classifier
        self.cls_g = nn.Linear(2048, 256)
        self.cls_v = nn.Linear(2048, 256)
        self.cls_c = nn.Linear(2048, 256)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.transit(x)
        x_g = self.cls_g(x)
        x_v = self.cls_v(x)
        x_c = self.cls_c(x)
        return x_g, x_v, x_c
    
    
class Simple50Embedding(nn.Module):
    def __init__(self, embedding_dimension=256):
        super().__init__()
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = nn.Sequential(
            base.layer0,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.pool = nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Linear(2048, embedding_dimension),
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return x
    
    
class Simple50GeMArc(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=.0):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.dropout = dropout
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = se_resnext50_32x4d()
        # dense feature
        self.gfeats = nn.Sequential(GeM(), nn.Flatten())
        self.arc_head = nn.Sequential(nn.Linear(2048, 256), nn.BatchNorm1d(256))
        # classifier
        self.cls_g = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, self.n_grapheme),
        )
        self.cls_v = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, self.n_vowel),
        )
        self.cls_c = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, self.n_consonant),
        )
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.feature_extractor.features(x)
        feats = self.gfeats(x)
        x_g = self.cls_g(feats)
        x_v = self.cls_v(feats)
        x_c = self.cls_c(feats)
        
        return x_g, x_v, x_c, self.arc_head(feats)
    
    
class Simple50GeMSpatialOrdered(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=.0):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.dropout = dropout
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = se_resnext50_32x4d()
        # dense feature
        self.gfeats = nn.Sequential(GeM_Spatial(), nn.Flatten(), nn.Linear(8192, 512), nn.BatchNorm1d(512), Mish())
        self.vcfeats = nn.Sequential(nn.BatchNorm1d(self.n_vowel+self.n_consonant), Mish())
        # classifier
        self.cls_g = nn.Linear(512+self.n_vowel+self.n_consonant, self.n_grapheme)
        self.cls_v = nn.Linear(512, self.n_vowel)
        self.cls_c = nn.Linear(512, self.n_consonant)
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.feature_extractor.features(x)
        feats = self.gfeats(x)
        x_v = self.cls_v(feats)
        x_c = self.cls_c(feats)
        x_vc = self.vcfeats(torch.cat((x_v, x_c), 1))
        x_g = self.cls_g(torch.cat((feats, x_vc), 1))
        
        return x_g, x_v, x_c
    
    
class Simple50GeMClsPool(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=.0):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.dropout = dropout
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = se_resnext50_32x4d()
        # dense feature
        self.cls_pool = nn.Sequential(
            # 2048x2x2
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.BatchNorm2d(2048), 
            Mish(),
            # 8192
            nn.Flatten(),
        )
        self.gfeats = nn.Sequential(GeM(), nn.Flatten(), nn.BatchNorm1d(n_grapheme+n_vowel+n_consonant), Mish())
        
        # classifier
        self.cls_g = nn.Linear(8192, n_grapheme)
        self.cls_v = nn.Linear(8192, n_vowel)
        self.cls_c = nn.Linear(8192, n_consonant)
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.feature_extractor.features(x)
        x = self.cls_conv(x)
        x = self.gfeats(x)
        x_g, x_v, x_c = torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        x_g = self.cls_g(x_g)
        x_v = self.cls_v(x_v)
        x_c = self.cls_c(x_c)
        
        return x_g, x_v, x_c
    
    
class Simple50GeMSpatial(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=.0):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.dropout = dropout
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = se_resnext50_32x4d()
        # dense feature
        self.gfeats = nn.Sequential(GeM_Spatial(), nn.Flatten(), nn.Linear(8192, 512), nn.BatchNorm1d(512), Mish())
        # classifier
        self.cls_g = nn.Linear(512, self.n_grapheme)
        self.cls_v = nn.Linear(512, self.n_vowel)
        self.cls_c = nn.Linear(512, self.n_consonant)
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.feature_extractor.features(x)
        feats = self.gfeats(x)
        x_g = self.cls_g(feats)
        x_v = self.cls_v(feats)
        x_c = self.cls_c(feats)
        
        return x_g, x_v, x_c
    
    
class Simple50GeM(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=.0):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.dropout = dropout
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = se_resnext50_32x4d()
        # dense feature
        self.gfeats = nn.Sequential(GeM(), nn.Flatten())
        # classifier
        self.cls_g = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, self.n_grapheme),
        )
        self.cls_v = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, self.n_vowel),
        )
        self.cls_c = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(256, self.n_consonant),
        )
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.feature_extractor.features(x)
        feats = self.gfeats(x)
        x_g = self.cls_g(feats)
        x_v = self.cls_v(feats)
        x_c = self.cls_c(feats)
        
        return x_g, x_v, x_c
    
    
class Simple50AnM(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7, dropout=.1, output_features=False):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.dropout = dropout
        self.feat_out = output_features
        
        # feature extraction
        # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
        self.feature_extractor = se_resnext50_32x4d()
        # global pooling
        self.gpoolmax = nn.AdaptiveMaxPool2d((1, 1))
        self.gpoolavg = nn.AdaptiveAvgPool2d((1, 1))
        # classifier
        self.cls_g = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, self.n_grapheme),
        )
        self.cls_v = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, self.n_vowel),
        )
        self.cls_c = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Mish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, self.n_consonant),
        )
    
    def pooling(self, x):
        pmax = self.gpoolmax(x)
        pavg = self.gpoolavg(x)
        return .5 * (pmax + pavg).view(-1, 2048)
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.feature_extractor.features(x)
        feats = self.pooling(x)
        x_g = self.cls_g(feats)
        x_v = self.cls_v(feats)
        x_c = self.cls_c(feats)
        
        if self.feat_out:
            return x_g, x_v, x_c, feats
        else:
            return x_g, x_v, x_c
    
    
# ------------------------------------------------------------------------
class Simple50(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        # classifiers
        # 168x168 -> 6x6
        self.feature_extractor = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        self.cls = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),  
            nn.Linear(2048, 512),
            # 168+11+7x1x1
            nn.Conv2d(1024, n_grapheme+n_vowel+n_consonant, kernel_size=3),
            nn.Flatten(),
        )
    
    def forward(self, x):

        # Perform the usual forward pass
        x = self.adapter(x)
        x = self.feature_extractor.features(x)
        x = self.cls(x)
        
        return torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
    

# ------------------------------------------------------------------------
class LocNCls(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        # classifiers
        self.feature_extractor = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        self.cls = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            # 128x128
            #nn.Linear(2048*2*2, 1024),
            # 168x168
            nn.Linear(2048*3*3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_grapheme+n_vowel+n_consonant),
        )
        
        # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             # 128->42
#             nn.Conv2d(1, 8, kernel_size=5, stride=3),
#             nn.ReLU(True),
#             # 42->14
#             nn.Conv2d(8, 16, kernel_size=5, stride=3, padding=1),
#             nn.ReLU(True),
#             # 14=>7
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#         )
        self.localization = nn.Sequential(
            # 168->114
            nn.Conv2d(1, 8, kernel_size=5, stride=3, padding=1),
            nn.ReLU(True),
            # 42->18
            nn.Conv2d(8, 16, kernel_size=5, stride=3),
            nn.ReLU(True),
            # 18=>9
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        # Regressors for each category for the 3 * 2 affine matrix
        self.loc_fc = nn.Sequential(
            # 128
            #nn.Linear(1568, 768),
            nn.Linear(2592, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )
            
        # Initialize the weights/bias with identity transformation
        self.loc_fc[2].weight.data.zero_()
        self.loc_fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 7 * 7)
        theta = self.loc_fc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x
    
    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.adapter(x)
        x = self.feature_extractor.features(x)
        x = self.cls(x)
        
        return torch.split(x, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)


class LocNCls_Single(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        # classifiers
        self.feature_extractor = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        self.cls = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(2048*2*2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, n_classes),
        )
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # 128->42
            nn.Conv2d(1, 8, kernel_size=5, stride=3),
            nn.ReLU(True),
            # 42->14
            nn.Conv2d(8, 16, kernel_size=5, stride=3, padding=1),
            nn.ReLU(True),
            # 14=>7
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        # Regressors for each category for the 3 * 2 affine matrix
        self.loc_fc = nn.Sequential(
            nn.Linear(1568, 768),
            nn.ReLU(True),
            nn.Linear(768, 3 * 2)
        )
            
        # Initialize the weights/bias with identity transformation
        self.loc_fc[2].weight.data.zero_()
        self.loc_fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 7 * 7)
        theta = self.loc_fc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = self.adapter(x)
        # 128x128->
        x = self.feature_extractor.features(x)
        x = self.cls(x)
        
        return [x]
    
    
# ------------------------------------------------------------------------
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, target_size, bilinear=True):
        super(up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(size=target_size, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


class MskNCls(nn.Module):
    """base model is the pretrained model se_resnext50_32x4d
    """
    def __init__(self, seresnext, category=0):
        super().__init__()
        
        self.cat = category
        
        self.base = seresnext.requires_grad_(False)
        
        childrens0 = list(self.base.children())
        childrens1 = list(childrens0[1].children())
        
        self.adapter = childrens0[0]
        
        # based on 168*168 input
        # out 64x42x42
        self.down0 = childrens1[0]
        # out 256x42x42
        self.down1 = childrens1[1]
        # out 512x21x21
        self.down2 = childrens1[2]
        # out 1024x11x11
        self.down3 = childrens1[3]
        # out 2048x6x6
        self.down4 = childrens1[4]
        # out 1024x11x11
        self.up0 = up(2048+1024, 1024, 11)
        # out 512x21x21
        self.up1 = up(1024+512, 1, 21)
        # out 1x21x21
        self.mask = nn.Sequential(
            nn.Sigmoid(),
            nn.Upsample(size=168, mode='bilinear', align_corners=True),
        )
        
    def forward_mask(self, x):
        x0 = self.adapter(x)
        x1 = self.down0(x0)      # 64   x 42 x 42
        x2 = self.down1(x1)      # 256  x 42 x 42
        x3 = self.down2(x2)      # 512  x 21 x 21
        x4 = self.down3(x3)      # 1024 x 11 x 11
        x5 = self.down4(x4)      # 2048 x 6  x 6
        x4u = self.up0(x5, x4)   # 1024 x 11 x 11
        x3u = self.up1(x4u, x3)  # 512  x 21 x 21
        return self.mask(x3u)    # 1    x 168x 168
        
    def forward(self, x):
        
        mask = self.forward_mask(x)
        
        masked = x * mask
        
        y = self.base(masked)
        
        return torch.split(y, [168, 11, 7], dim=1)[self.cat:self.cat+1]
    
        
# ------------------------------------------------------------------------
class LocNCls_Light(nn.Module):
    def __init__(self, base_model, n_classes):
        super().__init__()
        
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        # base feature extractor
        self.base = base_model
        
        # classifiers
        self.cls = densenet121(num_classes=n_classes)
        
        # localization
        # input 160x160 -> 5x5; 168x168 -> 6x6
        # output 3x3
        self.localization = nn.Conv2d(2048, 512, kernel_size=2, stride=2)
        
        # Regressors for each category for the 3 * 2 affine matrix
        self.loc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )
            
        # Initialize the weights/bias with identity transformation
        self.loc[2].weight.data.zero_()
        self.loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.base(x)
        xs = self.localization(xs)
        xs = xs.view(-1, 512 * 3 * 3)
        theta = self.loc_grapheme(xs)
        theta = theta_grapheme.view(-1, 2, 3)

        grid_grapheme = F.affine_grid(theta_grapheme, x.size())
        grid_vowel = F.affine_grid(theta_vowel, x.size())
        grid_conconant = F.affine_grid(theta_consonant, x.size())
        
        x_grapheme = F.grid_sample(x, grid_grapheme)
        x_vowel = F.grid_sample(x, grid_grapheme)
        x_conconant = F.grid_sample(x, grid_grapheme)

        return x_grapheme, x_vowel, x_conconant
    
    def forward(self, x):
        # adapt 1 channel to 3 channels
        x = self.adapter(x)
        # transform the input
        x_g, x_v, x_c = self.stn(x)

        y_g = self.cls_grapheme(x_g)
        y_v = self.cls_vowel(x_v)
        y_c = self.cls_consonant(x_c)
        
        return y_g, y_v, y_c


# class Simple50GeM_ArcFace_Single(nn.Module):
#     def __init__(self, n_classes=168):
#         super().__init__()
#         self.n_classes = n_classes
        
#         self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
#         # feature extraction
#         # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
#         self.feature_extractor = se_resnext50_32x4d()
#         self.gpool = GeM()
#         # classifier
#         self.cls = nn.Linear(2048, n_classes)
#         # advanced learnable loss
#         # center loss
#         self.arcface = AngularPenaltySMLoss(in_features=2048, out_features=n_classes)
    
#     def forward(self, x, y=None):

#         # Perform the usual forward pass
#         x = self.adapter(x)
#         x = self.feature_extractor.features(x)
#         feats = self.gpool(x).view(-1, 2048)
#         x = self.cls(feats)
        
#         if y == None:
#             cntr_lss = 0.
#         else:
#             cntr_lss = self.arcface(feats, y[:, -1])
        
#         return (x, cntr_lss, )
    
    
# class Simple50GeM_CenterLoss_Single(nn.Module):
#     def __init__(self, n_classes=168):
#         super().__init__()
#         self.n_classes = n_classes
        
#         self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
#         # feature extraction
#         # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
#         self.feature_extractor = se_resnext50_32x4d()
#         self.gpool = GeM()
#         # classifier
#         self.cls = nn.Linear(2048, n_classes)
#         # advanced learnable loss
#         # center loss
#         self.center_loss = CenterLoss(num_classes=n_classes)
    
#     def forward(self, x, y=None):

#         # Perform the usual forward pass
#         x = self.adapter(x)
#         x = self.feature_extractor.features(x)
#         feats = self.gpool(x).view(-1, 2048)
#         x = self.cls(feats)
        
#         if y == None:
#             cntr_lss = 0.
#         else:
#             cntr_lss = self.center_loss(feats, y[:, -1])
        
#         return (x, cntr_lss, )
    
    
# class Simple50GeM_Single(nn.Module):
#     def __init__(self, n_classes=168):
#         super().__init__()
        
#         self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
#         # feature extraction
#         # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
#         self.feature_extractor = se_resnext50_32x4d()
#         # classifier
#         self.cls = nn.Sequential(
#             GeM(),
#             nn.Flatten(),
#             nn.Linear(2048, n_classes)
#         )
    
#     def forward(self, x):

#         # Perform the usual forward pass
#         x = self.adapter(x)
#         x = self.feature_extractor.features(x)
#         x = self.cls(x)
        
#         return (x,)
    
    
# class Simple50GeM_CenterLoss(nn.Module):
#     def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
#         super().__init__()
#         self.n_grapheme = n_grapheme
#         self.n_vowel = n_vowel
#         self.n_consonant = n_consonant
        
#         # feature extraction
#         # input 128x128 -> 4x4; 160x160 -> 5x5; 168x168 -> 6x6
#         self.feature_extractor = se_resnext50_32x4d()
#         # global pooling
#         self.pooling = nn.Sequential(GeM(), Mish(), nn.Flatten())
#         # center loss
#         self.c_loss = CenterLoss()
#         # classifier
#         self.cls_g = nn.Sequential(
#             nn.BatchNorm1d(2048),
#             nn.Dropout(p=dropout, inplace=True),
#             nn.Linear(2048, 512),
#             Mish(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(p=dropout, inplace=True),
#             nn.Linear(512, self.n_grapheme),
#         )
#         self.cls_v = nn.Sequential(
#             nn.BatchNorm1d(2048),
#             nn.Dropout(p=dropout, inplace=True),
#             nn.Linear(2048, 512),
#             Mish(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(p=dropout, inplace=True),
#             nn.Linear(512, self.n_vowel),
#         )
#         self.cls_c = nn.Sequential(
#             nn.BatchNorm1d(2048),
#             nn.Dropout(p=dropout, inplace=True),
#             nn.Linear(2048, 512),
#             Mish(),
#             nn.BatchNorm1d(512),
#             nn.Dropout(p=dropout, inplace=True),
#             nn.Linear(512, self.n_consonant),
#         )
    
#     def forward(self, x, y=None):

#         # Perform the usual forward pass
#         x = self.feature_extractor.features(x)
#         feats = self.pooling(x).view(-1, 2048)
#         x_g = self.cls_g(x)
#         x_v = self.cls_v(x)
#         x_c = self.cls_c(x)
        
#         if y == None:
#             adv_lss = 0.
#         else:
#             adv_lss = self.c_loss(feats, y[:, -1])
        
#         return x_g, x_v, x_c, adv_lss
    
    
# class LocNCls(nn.Module):
#     def __init__(self, n_base, n_grapheme, n_vowel, n_consonant):
#         super().__init__()
        
#         # base feature extractor
#         self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
#         self.base = nn.Sequential(*list(wide_resnet50_2().children())[:-2])
        
#         # classifiers
#         self.cls_base = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(in_features=2048, out_features=n_base))
#         self.cls_grapheme = densenet121(num_classes=n_grapheme)
#         self.cls_vowel = densenet121(num_classes=n_vowel)
#         self.cls_consonant = densenet121(num_classes=n_consonant)
        
#         # localization
#         # 160x160 -> 5x5; 168x168 -> 6x6
#         self.localization = nn.Conv2d(2048, 512, kernel_size=2, stride=2)
        
#         # Regressors for each category for the 3 * 2 affine matrix
#         self.loc_grapheme = nn.Sequential(
#             nn.Linear(512 * 3 * 3, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 3 * 2)
#         )
#         self.loc_vowel = nn.Sequential(
#             nn.Linear(512 * 3 * 3, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 3 * 2)
#         )
#         self.loc_consonant = nn.Sequential(
#             nn.Linear(512 * 3 * 3, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 3 * 2)
#         )
        
#         # Initialize the weights/bias with identity transformation
#         self.loc_grapheme[2].weight.data.zero_()
#         self.loc_grapheme[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         self.loc_vowel[2].weight.data.zero_()
#         self.loc_vowel[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         self.loc_consonant[2].weight.data.zero_()
#         self.loc_consonant[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#     # Spatial transformer network forward function
#     def stn(self, x):
#         x_feat = self.base(x)
#         xs = self.localization(x_feat.detach())
#         xs = xs.view(-1, 512 * 3 * 3)
#         theta_grapheme = self.loc_grapheme(xs)
#         theta_grapheme = theta_grapheme.view(-1, 2, 3)
#         theta_vowel = self.loc_vowel(xs)
#         theta_vowel = theta_vowel.view(-1, 2, 3)
#         theta_consonant = self.loc_consonant(xs)
#         theta_consonant = theta_consonant.view(-1, 2, 3)

#         grid_grapheme = F.affine_grid(theta_grapheme, x.size())
#         grid_vowel = F.affine_grid(theta_vowel, x.size())
#         grid_conconant = F.affine_grid(theta_consonant, x.size())
        
#         x_grapheme = F.grid_sample(x, grid_grapheme)
#         x_vowel = F.grid_sample(x, grid_grapheme)
#         x_conconant = F.grid_sample(x, grid_grapheme)

#         return x_grapheme, x_vowel, x_conconant, x_feat
    
#     def forward(self, x):
#         # adapt 1 channel to 3 channels
#         x = self.adapter(x)
#         # transform the input
#         x_g, x_v, x_c, x_feat = self.stn(x)

#         # Perform the usual forward pass
#         y_base = self.cls_base(x_feat)
#         y_g = self.cls_grapheme(x_g)
#         y_v = self.cls_vowel(x_v)
#         y_c = self.cls_consonant(x_c)
        
#         return y_base, y_g, y_v, y_c

