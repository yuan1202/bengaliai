from functools import wraps
import random

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

        
class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_x, self.stack_y = alpha, stack_x, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        #lambd_cap = np.random.choice(2, last_target.size(0), p=[.9, .1])
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        #lambd = np.max([lambd, lambd_cap], 0)
            
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
            
        if self.stack_y:
            new_target = torch.cat([last_target.float(), y1.float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
            
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()

            
# -------------------------------------------------------------------------------------
class MixUpLoss_Single(Module):
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
        if len(target.shape) == 2 and target.shape[1] == 3:
            loss1, loss2 = self.crit(output, target[:,0].long()), self.crit(output, target[:,1].long())
            d = loss1 * target[:,-1] + loss2 * (1-target[:,-1])
        else:  d = self.crit(output, target[:,0])
        if self.reduction == 'mean':    return d.mean()
        elif self.reduction == 'sum':   return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

        
class MixUpCallback_Single(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_x, self.stack_y = alpha, stack_x, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss_Single(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target.float(), y1.float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + y1.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
            
            
# -------------------------------------------------------------------------------------      
def rand_bboxes_FlantIndices(size, lam):
    B = size[0]
    H = size[2]
    W = size[3]
    
    index_addition = np.arange(B) * H * W
    
    # varying aspect ratios
    #w_ratios = np.random.uniform(low=.2, high=.8, size=B)
    #h_ratios = np.random.uniform(low=.2, high=.8, size=B)
    #cut_ws = np.round(W * w_ratios).astype(int)
    #cut_hs = np.round(H * h_ratios).astype(int)
    
    cut_ratios = np.sqrt(1. - lam)
    cut_ws = np.round(W * cut_ratios).astype(int)
    cut_hs = np.round(H * cut_ratios).astype(int)

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
        cm = random.randint(0, 2)
        # 1/3 chance cutmix
        if cm==0:
            flattened_indices, lambd = rand_bboxes_FlantIndices(last_input.size(), lambd)
            lambd = last_input.new(lambd)
            last_input_newview = last_input.view(-1)
            last_input_newview[torch.from_numpy(flattened_indices)] = input_wip.view(-1)[torch.from_numpy(flattened_indices)]
            new_input = last_input
        # 1/3 chance mixup
        elif cm == 1:
            lambd = last_input.new(lambd)
            out_shape = [lambd.size(0)] + [1 for _ in range(len(input_wip.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + input_wip * (1-lambd).view(out_shape))
        # 1/3 chance doing nothing
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
    def __init__(self, learn:Learner, stack_y:bool=True):
        super().__init__(learn)
        self.stack_y = stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        input_wip, target_wip = last_input[shuffle], last_target[shuffle]
        
        # randomly choose between mixup or cutmix
        cm = random.randint(0, 1)
        
        # 1/3 chance cutmix
        if cm == 0:
            flattened_indices, lambd = rand_bboxes_FlantIndices(last_input.size())
            lambd = last_input.new(lambd)
            last_input_newview = last_input.view(-1)
            last_input_newview[torch.from_numpy(flattened_indices)] = input_wip.view(-1)[torch.from_numpy(flattened_indices)]
            new_input = last_input
        # 1/3 chance doing nothing
        elif cm == 1:
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


# -------------------------------------------------------------------------------------
class MixUpLoss_Advlss_Single(Module):
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
        if len(target.shape) == 2 and target.shape[1] == 3:
            loss1, loss2 = self.crit(output, target[:,0].long()), self.crit(output, target[:,1].long())
            d = loss1 * target[:,-1] + loss2 * (1-target[:,-1])
        else:  d = self.crit(output, target[:,0])
        if self.reduction == 'mean':    return d.mean()
        elif self.reduction == 'sum':   return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'):  return self.old_crit
        elif hasattr(self, 'old_red'): 
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit
        
             
class MuCmCallback_AdvLss_Single(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_y:bool=True):
        super().__init__(learn)
        self.alpha, self.stack_y = alpha, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        
        lambd = np.random.beta(self.alpha, self.alpha, last_target.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        input_wip, target_wip = last_input[shuffle], last_target[shuffle]
        
        # randomly choose between mixup or cutmix
        cm = random.randint(0, 1)
        if cm:
            flattened_indices, lambd = rand_bboxes_FlantIndices(last_input.size(), lambd)
            lambd = last_input.new(lambd)
            last_input_newview = last_input.view(-1)
            last_input_newview[torch.from_numpy(flattened_indices)] = input_wip.view(-1)[torch.from_numpy(flattened_indices)]
            new_input = last_input
            
        else:
            lambd = last_input.new(lambd)
            out_shape = [lambd.size(0)] + [1 for _ in range(len(input_wip.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + input_wip * (1-lambd).view(out_shape))
        
        if self.stack_y:
            new_target = torch.cat([last_target.float(), target_wip.float(), lambd[:,None].float()], 1)
        else:
            if len(last_target.shape) == 2:
                lambd = lambd.unsqueeze(1).float()
            new_target = last_target.float() * lambd + target_wip.float() * (1-lambd)
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()
