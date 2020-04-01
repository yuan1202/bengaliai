import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from mish.mish import Mish

# --------------------------------------------------------------------------------------------


class ConvBn2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channel, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=4,):
        super(SqueezeExcite, self).__init__()

        self.fc1 = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x,1)
        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = self.fc2(s)
        x = x*torch.sigmoid(s)
        return x
    

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

    
class SENextBottleneck(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, group=32,
                 reduction=16, pool=None, is_shortcut=False):
        super(SENextBottleneck, self).__init__()

        self.conv_bn1 = ConvBn2d(in_channel, channel[0], kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(channel[0], channel[1], kernel_size=3, padding=1, stride=1, groups=group)
        self.conv_bn3 = ConvBn2d(channel[1], out_channel, kernel_size=1, padding=0, stride=1)
        self.scale    = SqueezeExcite(out_channel, reduction)

        #---
        self.is_shortcut = is_shortcut
        self.stride = stride
        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)

        if stride==2:
            if pool=='max' : self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            if pool=='avg' : self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        z = F.relu(self.conv_bn1(x), inplace=True)
        z = F.relu(self.conv_bn2(z), inplace=True)
        if self.stride==2:
            z = self.pool(z)

        z = self.scale(self.conv_bn3(z))
        if self.is_shortcut:
            if self.stride==2:
                x = F.avg_pool2d(x, 2, 2)
            x = self.shortcut(x)

        z += x
        z = F.relu(z,inplace=True)
        return z


# --------------------------------------------------------------------------------------------
class seresxt50heng(nn.Module):

    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7):
        super().__init__()

        self.block0  = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            
        )
        
        self.block1  = nn.Sequential(
             SENextBottleneck( 64, [128,128], 256, stride=2, is_shortcut=True, pool='max', ),
          * [SENextBottleneck(256, [128,128], 256, stride=1, is_shortcut=False,) for i in range(1, 3)],
        )
        
        self.block2  = nn.Sequential(
             SENextBottleneck(256, [256,256], 512, stride=2, is_shortcut=True, pool='max', ),
          * [SENextBottleneck(512, [256,256], 512, stride=1, is_shortcut=False,) for i in range(1, 4)],
        )
        
        self.block3  = nn.Sequential(
             SENextBottleneck( 512,[512,512],1024, stride=2, is_shortcut=True, pool='max', ),
          * [SENextBottleneck(1024,[512,512],1024, stride=1, is_shortcut=False,) for i in range(1, 6)],
        )
        
        self.block4 = nn.Sequential(
             SENextBottleneck(1024,[1024,1024],2048, stride=2, is_shortcut=True,pool='avg', ),
          * [SENextBottleneck(2048,[1024,1024],2048, stride=1, is_shortcut=False) for i in range(1, 3)],
        )
        
        self.gcls = nn.Linear(2048, n_grapheme)
        self.vcls = nn.Linear(2048, n_vowel)
        self.ccls = nn.Linear(512, n_consonant)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        #x = self.gpool(x).reshape(batch_size, -1)
        #x = torch.sum(x, dim=(-1, -2))
        x = F.dropout(x, 0.1, self.training)
        
        return self.gcls(x), self.vcls(x), self.ccls(x)
    