from functools import wraps
from itertools import accumulate
import random
import skimage
import cv2

import torch
from torch import nn
import torch.nn.functional as F

import fastai
from fastai.vision import *
from fastai.callbacks import *


# -------------------------------------------------------------------------------------
# The code below modifies fast.ai MixUp calback to make it compatible with the current data.
class MixUpLoss(Module):
    "Adapt the loss function `crit` to go with mixup."
    
    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'): 
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else: 
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.shape) == 2 and target.shape[1] == 7:
            loss1, loss2 = self.crit(output, target[:, 0:3].long()), self.crit(output, target[:, 3:6].long())
            d = loss1 * target[:,-1] + loss2 * (1-target[:,-1])
        else:  d = self.crit(output, target)
        if self.reduction == 'mean':    return d.mean()
        elif self.reduction == 'sum':   return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

        
def rand_bboxes_FlantIndices(size, lam, clamp=None):
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    
    index_addition = np.arange(B) * C * H * W
    
    if clamp is None:
        cut_ratios = np.sqrt(1. - lam)
        cut_ws = np.round(W * cut_ratios).astype(int)
        cut_hs = np.round(H * cut_ratios).astype(int)
    else:
        assert isinstance(clamp, tuple)
        # varying aspect ratios
        base_size = min(H, W)
        cut_ws = np.round(np.random.uniform(clamp[0], clamp[1], size=B) * base_size).astype(np.int)
        cut_hs = np.round(np.random.uniform(clamp[0], clamp[1], size=B) * base_size).astype(np.int)

    # uniform
    cx = np.random.randint(W, size=B)
    cx = np.clip(cx, cut_ws // 2, W - cut_ws // 2)
    cy = np.random.randint(H, size=B)
    cy = np.clip(cy, cut_hs // 2, H - cut_hs // 2)

    bbx0s = cx - cut_ws // 2
    bby0s = cy - cut_hs // 2
    bbx1s = cx + cut_ws // 2
    bby1s = cy + cut_hs // 2
    
    multi_indices = [np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)) for x0, x1, y0, y1 in zip(bbx0s, bbx1s, bby0s, bby1s)]
    multi_indices_rvl = [np.ravel_multi_index(lst[::-1], (H, W)).flatten() for lst in multi_indices]
    
    boxes_in_flattened_indices = np.concatenate([index_r + addition for index_r, addition in zip(multi_indices_rvl, index_addition)])
    lamb = 1 - ((bbx1s - bbx0s) * (bby1s - bby0s) / (H * W))
    
    return boxes_in_flattened_indices, lamb


def half_bboxes_FlantIndices(size, lam, clamp=None):
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    
    index_addition = np.arange(B) * H * W
    
    # 0 - left new; 1 - right new; 2 - top new; 3 = bottom new
    # x0, x1, y0, y1
    combination = {
        0: (0, int(W//2), 0, H),
        1: (int(W//2), W, 0, H),
        2: (0, W, 0, int(H//2)),
        3: (0, W, int(H//2), H)
    }
    
    opt = np.random.randint(0, 3, size=B)
    
    boxes = [combination[o] for o in opt]
    
    multi_indices = [np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)) for x0, x1, y0, y1 in boxes]
    multi_indices_rvl = [np.ravel_multi_index(lst[::-1], (H, W)).flatten() for lst in multi_indices]
    
    boxes_in_flattened_indices = np.concatenate([index_r + addition for index_r, addition in zip(multi_indices_rvl, index_addition)])
    lamb = np.array([0.5] * B)
    
    return boxes_in_flattened_indices, lamb


def random_half_FlantIndices(size, lam=None, clamp=None):
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    
    canvas = np.ones((H*2, W*2))
    canvas[:, W:] = 0
    
    index_addition = np.arange(B) * H * W
    
    angles = np.random.randint(0, 359, size=B)
    
    prints = [skimage.transform.rotate(canvas, a) for a in angles]
    crops = [np.round(p[(H//2):(H//2)+H, (W//2):(W//2)+W]).astype('uint8') for p in prints]
    
    multi_indices = [np.where(c.reshape(-1) == 1)[0] for c in crops]
    
    boxes_in_flattened_indices = np.concatenate([index_r + addition for index_r, addition in zip(multi_indices, index_addition)])
    lamb = np.array([0.5] * B)
    
    return boxes_in_flattened_indices, lamb


def grid_mask(size):
    b = size[0]
    c = size[1]
    h = size[2]
    w = size[3]
    s = np.random.uniform(min(h, w), max(h, w))
    
    index_addition = np.arange(b) * h * w

    mask = np.zeros((h*2, w*2))

    #density_h = random.randint(s//8, s//5)
    #density_w = random.randint(s//8, s//5)
    density_h = random.randint(s//15, s//5)
    density_w = random.randint(s//15, s//5)
    #factor_h = random.randint(2, 3)
    #factor_w = random.randint(2, 3)
    factor_h = np.random.uniform(1.5, 2)
    factor_w = np.random.uniform(1.5, 2)

    mask[::density_h, ::density_w] = 1

    mask = cv2.dilate(mask, np.ones((np.round(density_h/factor_h).astype(int), np.round(density_w/factor_w).astype(int)), np.uint8), iterations=1)
        
    random_h = np.random.randint(0, h//2, size=b)
    random_w = np.random.randint(0, w//2, size=b)

    individual_masks = [np.round(mask[random_h[i]:h+random_h[i], random_w[i]:w+random_w[i]]).astype('uint8') for i in range(b)]
    
    multi_indices = [np.where(m.reshape(-1) == 0)[0] for m in individual_masks]
    flattened_indices = np.concatenate([index_r + addition for index_r, addition in zip(multi_indices, index_addition)])

    return flattened_indices
    
        
# -------------------------------------------------------------------------------------  
class MuGmCallback_InnerPeace(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=.4, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_y = alpha, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        # split for MU, and Null
        B = last_input.size(0)
        divide = 2
        unit = B // divide
        residual = B % divide
        split = [unit if i < divide - 1 else unit + residual for i in range(divide)]
        split = list(accumulate(split))
        
        mu_rng = torch.arange(0, split[0], dtype=torch.long)
        #gm_rng = torch.arange(split[0], split[1], dtype=torch.long)
        #null_rng = torch.arange(split[1], split[2], dtype=torch.long)
        null_rng = torch.arange(split[0], split[1], dtype=torch.long)
        #imgs_mu, imgs_gm, imgs_null = last_input[mu_rng], last_input[gm_rng], last_input[null_rng]
        #lbls_mu, lbls_gm, lbls_null = last_target[mu_rng], last_target[gm_rng], last_target[null_rng]
        imgs_mu, imgs_null = last_input[mu_rng], last_input[null_rng]
        lbls_mu, lbls_null = last_target[mu_rng], last_target[null_rng]
        
        # ---------------- mixup ----------------
        perm_mu = torch.randperm(imgs_mu.size(0)).to(imgs_mu.device)
        perm_imgs_mu = imgs_mu[perm_mu]
        perm_lbls_mu = lbls_mu[perm_mu]

        gamma_mu = np.random.beta(self.alpha, self.alpha, imgs_mu.size(0))
        gamma_mu = np.concatenate([gamma_mu[:, None], 1-gamma_mu[:, None]], 1).max(1)

        gamma_mu = imgs_mu.new(gamma_mu)
        gamma_mu_newview = gamma_mu.view(-1, 1, 1, 1)

        new_imgs_mu = imgs_mu * gamma_mu_newview + perm_imgs_mu * (1 - gamma_mu_newview)
        
        # ---------------- gridmask ----------------
        flattened_indices = grid_mask(imgs_gm.size())
        gamma_gm = np.ones(imgs_gm.size(0))
        gamma_gm = imgs_gm.new(gamma_gm)
        imgs_gm_newview = imgs_gm.view(-1)
        flattened_indices_torch = torch.from_numpy(flattened_indices)
        imgs_gm_newview[flattened_indices_torch] = np.random.uniform(imgs_gm.min().item(), imgs_gm.max().item())
        
        new_imgs_gm = imgs_gm
        
        # ---------------- null ----------------
        gamma_null = np.ones(imgs_null.size(0))
        gamma_null = imgs_null.new(gamma_null)

        # ---------------- back together and shuffle ----------------
        new_imgs = torch.cat((new_imgs_mu, new_imgs_gm, imgs_null), dim=0)
        new_lbls = torch.cat((perm_lbls_mu, lbls_gm, lbls_null), dim=0)
        new_gamma = torch.cat([gamma_mu, gamma_gm, gamma_null])
        
        #new_imgs = torch.cat((new_imgs_mu, new_imgs_gm), dim=0)
        #new_lbls = torch.cat((perm_lbls_mu, lbls_gm), dim=0)
        #new_gamma = torch.cat([gamma_mu, gamma_gm])

        perm_final = torch.randperm(new_imgs.size(0)).to(new_imgs.device)
        
        new_imgs = new_imgs[perm_final]
        lbls = last_target[perm_final]
        new_lbls = new_lbls[perm_final]
        new_gamma = new_gamma[perm_final]
            
        if self.stack_y:
            new_lbls = torch.cat([lbls.float(), new_lbls.float(), new_gamma[:, None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                new_gamma = new_gamma.unsqueeze(1).float()
            new_lbls = lbls.float() * new_gamma + new_lbls.float() * (1-new_gamma)
            
        return {'last_input': new_imgs, 'last_target': new_lbls}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=.4, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_y = alpha, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:, None], 1-lambd[:, None]], 1).max(1)
        
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        
        # randomly choose between mixup or cutmix
        opt = 0#random.randint(0, 2)
        # 1/3 mixup
        if opt == 0:
            lambd = last_input.new(lambd)
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        # 1/3 chance cutout
        elif opt == 1:
            flattened_indices, _ = rand_bboxes_FlantIndices(last_input.size(), lambd)
            mix_indices = torch.from_numpy(flattened_indices)
            lambd = np.ones(last_target.size(0))
            lambd = last_input.new(lambd)
            last_input_newview = last_input.view(-1)
            last_input_newview[mix_indices] = 0
            new_input = last_input
        # 1/3 chance doing nothing
        else:
            lambd = np.ones(last_target.size(0))
            lambd = last_input.new(lambd)
            new_input = last_input
            
        if self.stack_y:
            new_target = torch.cat([last_target.float(), y1.float(), lambd[:, None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
            
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


class MuCmCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=.4, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_y = alpha, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:, None], 1-lambd[:, None]], 1).max(1)
        
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        input_wip, target_wip = last_input[shuffle], last_target[shuffle]
        
        # randomly choose between mixup or cutmix
        opt = random.randint(0, 1)
        # 1/3 chance cutmix
        if opt == 0:
            flattened_indices, lambd = rand_bboxes_FlantIndices(last_input.size(), lambd)
            mix_indices = torch.from_numpy(flattened_indices)
            lambd = last_input.new(lambd)
            last_input_newview = last_input.view(-1)
            last_input_newview[mix_indices] = input_wip.view(-1)[mix_indices]
            new_input = last_input
        # 1/3 chance mixup
        elif opt == 1:
            lambd = last_input.new(lambd)
            out_shape = [lambd.size(0)] + [1 for _ in range(len(input_wip.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + input_wip * (1-lambd).view(out_shape))
        # 1/4 chance cutout
        #elif opt == 2:
        #    flattened_indices, _ = rand_bboxes_FlantIndices(last_input.size(), lambd)
        #    mix_indices = torch.from_numpy(flattened_indices)
        #    lambd = last_input.new(np.ones(last_target.size(0)))
        #    last_input_newview = last_input.view(-1)
        #    last_input_newview[mix_indices] = 0
        #    new_input = last_input
        # 1/4 chance doing nothing
        else:
            lambd = np.ones(last_target.size(0))
            lambd = last_input.new(lambd)
            new_input = last_input
        
        if self.stack_y:
            new_target = torch.cat([last_target.float(), target_wip.float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + target_wip.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
            
            
class CmCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=.4, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_y = alpha, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:, None], 1-lambd[:, None]], 1).max(1)
        
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        input_wip, target_wip = last_input[shuffle], last_target[shuffle]
        
        # randomly choose between mixup or cutmix
        opt = 0#random.randint(0, 1)
        # 1/2 chance cutmix
        if opt == 0:
            flattened_indices, lambd = rand_bboxes_FlantIndices(last_input.size(), lambd)
            mix_indices = torch.from_numpy(flattened_indices)
            lambd = last_input.new(lambd)
            last_input_newview = last_input.view(-1)
            last_input_newview[mix_indices] = input_wip.view(-1)[mix_indices]
            new_input = last_input
        # 1/2 chance doing nothing
        else:
            lambd = np.ones(last_target.size(0))
            lambd = last_input.new(lambd)
            new_input = last_input
        
        if self.stack_y:
            new_target = torch.cat([last_target.float(), target_wip.float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + target_wip.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


