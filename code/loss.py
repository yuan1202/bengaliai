import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------------------------------------------
# https://github.com/vandit15/Class-balanced-loss-pytorch.git
def focal_loss(labels, logits, alpha=.25, gamma=2.):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, beta=.999, gamma=2.):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().cuda()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)
    
    return F.cross_entropy(input=logits, target=labels, weight=weights)


grapheme_weights = np.load('../features/grapheme_roots_count.npy').tolist()
vowel_weights = np.load('../features/vowels_count.npy').tolist()
consonant_weights = np.load('../features/consonants_count.npy').tolist()


class CBLoss_combine_weighted(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, reduction='mean'):
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return 2*CB_loss(y[:,0], x1, grapheme_weights, 168) + \
               1*CB_loss(y[:,1], x2, vowel_weights, 11) + \
               1*CB_loss(y[:,2], x3, consonant_weights, 7)
    
# ----------------------------------------------------------------------------------------------------------------------------------
# https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch.git
class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -L
    

# ----------------------------------------------------------------------------------------------------------------------------------
# https://blog.csdn.net/weixin_40671425/article/details/98068190
# https://github.com/KaiyangZhou/pytorch-center-loss.git
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=1295, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss
    
    
# ----------------------------------------------------------------------------------------------------------------------------------
# Cross entropy loss is applied independently to each part of the prediction and the result is summed with the corresponding weight.
class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target,reduction='mean'):
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return (F.cross_entropy(x1, y[:,0], reduction=reduction) + \
                F.cross_entropy(x2, y[:,1], reduction=reduction) + \
                F.cross_entropy(x3, y[:,2], reduction=reduction)) / 3.


class Loss_combine_weighted(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target,reduction='mean'):
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return 0.7*F.cross_entropy(x1, y[:,0], reduction=reduction) + \
               0.1*F.cross_entropy(x2, y[:,1], reduction=reduction) + \
               0.2*F.cross_entropy(x3, y[:,2], reduction=reduction)
    
    
class Loss_combine_weighted_v2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target,reduction='mean'):
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return .50*F.cross_entropy(x1, y[:,0], reduction=reduction) + \
               .25*F.cross_entropy(x2, y[:,1], reduction=reduction) + \
               .25*F.cross_entropy(x3, y[:,2], reduction=reduction)
    
    
class Loss_single(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, target, reduction='mean'):
        y = target.long()
        return F.cross_entropy(x[0], y.view(-1), reduction=reduction)
    
    
# ----------------------------------------------------------------------------------------------------------------------------------
# center loss
class AdvancedLoss_Single(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, reduction='mean'):
        x, y0 = input
        y1 = target.long()
        return F.cross_entropy(x, y1[:, 0], reduction=reduction) + .1 * y0
    

class AdvancedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, reduction='mean'):
        x1, x2, x3, y0 = input
        y1 = target.long()
        return 0.7 * F.cross_entropy(x1, y1[:, 0], reduction=reduction) + \
               0.1 * F.cross_entropy(x2, y1[:, 1], reduction=reduction) + \
               0.2 * F.cross_entropy(x3, y1[:, 2], reduction=reduction) + \
               y0
    
# arcface loss
# AF = 

# class CE_with_CL(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, input, target, reduction='mean'):
#         x1, x2, x3, y0 = input
#         y1 = target.long()
#         return 0.7 * F.cross_entropy(x1, y1[:, 0], reduction=reduction) + \
#                0.1 * F.cross_entropy(x2, y1[:, 1], reduction=reduction) + \
#                0.2 * F.cross_entropy(x3, y1[:, 2], reduction=reduction) + \
#                0.1 * y0

# ----------------------------------------------------------------------------------------------------------------------------------
# replace CE with focal loss
        
class CEFL_combine_weighted(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target, reduction='mean'):
        x0, x1, x2 = input
        y0, y1, y2 = target[:, 0].long(), target[:, 1].long(), target[:, 2].long()
        
        y0oh = F.one_hot(y0, 168).float()
        y1oh = F.one_hot(y1, 11).float()
        y2oh = F.one_hot(y2, 7).float()
        
        return 2. * (focal_loss(y0oh, x0) + F.cross_entropy(x0, y0, reduction=reduction)) + \
               1. * (focal_loss(y1oh, x1) + F.cross_entropy(x1, y1, reduction=reduction)) + \
               1. * (focal_loss(y2oh, x2) + F.cross_entropy(x2, y2, reduction=reduction))