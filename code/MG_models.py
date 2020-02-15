from model import *

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import densenet121, wide_resnet50_2
import pretrainedmodels


# base = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None).to('cuda:0')
# ------------------------------------------------------------------------
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
    
    
class LocNCls(nn.Module):
    def __init__(self, n_base, n_grapheme, n_vowel, n_consonant):
        super().__init__()
        
        # base feature extractor
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base = nn.Sequential(*list(wide_resnet50_2().children())[:-2])
        
        # classifiers
        self.cls_base = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(), nn.Linear(in_features=2048, out_features=n_base))
        self.cls_grapheme = densenet121(num_classes=n_grapheme)
        self.cls_vowel = densenet121(num_classes=n_vowel)
        self.cls_consonant = densenet121(num_classes=n_consonant)
        
        # localization
        # 160x160 -> 5x5; 168x168 -> 6x6
        self.localization = nn.Conv2d(2048, 512, kernel_size=2, stride=2)
        
        # Regressors for each category for the 3 * 2 affine matrix
        self.loc_grapheme = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )
        self.loc_vowel = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )
        self.loc_consonant = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 2)
        )
        
        # Initialize the weights/bias with identity transformation
        self.loc_grapheme[2].weight.data.zero_()
        self.loc_grapheme[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.loc_vowel[2].weight.data.zero_()
        self.loc_vowel[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.loc_consonant[2].weight.data.zero_()
        self.loc_consonant[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        x_feat = self.base(x)
        xs = self.localization(x_feat.detach())
        xs = xs.view(-1, 512 * 3 * 3)
        theta_grapheme = self.loc_grapheme(xs)
        theta_grapheme = theta_grapheme.view(-1, 2, 3)
        theta_vowel = self.loc_vowel(xs)
        theta_vowel = theta_vowel.view(-1, 2, 3)
        theta_consonant = self.loc_consonant(xs)
        theta_consonant = theta_consonant.view(-1, 2, 3)

        grid_grapheme = F.affine_grid(theta_grapheme, x.size())
        grid_vowel = F.affine_grid(theta_vowel, x.size())
        grid_conconant = F.affine_grid(theta_consonant, x.size())
        
        x_grapheme = F.grid_sample(x, grid_grapheme)
        x_vowel = F.grid_sample(x, grid_grapheme)
        x_conconant = F.grid_sample(x, grid_grapheme)

        return x_grapheme, x_vowel, x_conconant, x_feat
    
    def forward(self, x):
        # adapt 1 channel to 3 channels
        x = self.adapter(x)
        # transform the input
        x_g, x_v, x_c, x_feat = self.stn(x)

        # Perform the usual forward pass
        y_base = self.cls_base(x_feat)
        y_g = self.cls_grapheme(x_g)
        y_v = self.cls_vowel(x_v)
        y_c = self.cls_consonant(x_c)
        
        return y_base, y_g, y_v, y_c