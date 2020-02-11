import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import densenet121, wide_resnet50_2


class LocNCls_Light(nn.Module):
    def __init__(self, n_base, n_grapheme, n_vowel, n_consonant):
        super().__init__()
        
        # base feature extractor
        self.adapter = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.base = nn.Sequential(*list(wide_resnet50_2().children())[:-2])
        
        # classifiers
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
        xs = self.localization(x_feat)
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