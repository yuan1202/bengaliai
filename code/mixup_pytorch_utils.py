import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ALPHA = .4


def rand_bboxes_FlantIndices(size, lam):
    H = size[2]
    W = size[3]
    
    index_addition = np.arange(size[0]) * H * W
    
    cut_ratios = np.sqrt(1. - lam)
    cut_ratios = np.clip(cut_ratios, .2, .8)
    
    cut_ws = np.round(W * cut_ratios).astype(int)
    cut_hs = np.round(H * cut_ratios).astype(int)

    # uniform
    cx = np.random.randint(W, size=cut_ratios.shape[0])
    cx = np.clip(cx, cut_ws // 2, W - cut_ws // 2)
    cy = np.random.randint(H, size=cut_ratios.shape[0])
    cy = np.clip(cy, cut_hs // 2, H - cut_hs // 2)

    bbx0s = cx - cut_ws // 2
    bby0s = cy - cut_hs // 2
    bbx1s = cx + cut_ws // 2
    bby1s = cy + cut_hs // 2
    
    multi_indices = [np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)) for x0, x1, y0, y1 in zip(bbx0s, bbx1s, bby0s, bby1s)]
    multi_indices_rvl = [np.ravel_multi_index(lst[::-1], (H, W)).flatten() for lst in multi_indices]
    
    boxes_in_flattened_indices = np.concatenate([index_r + addition for index_r, addition in zip(multi_indices_rvl, index_addition)])
    new_lambda = 1 - ((bbx1s - bbx0s) * (bby1s - bby0s) / (H * W))
    
    return boxes_in_flattened_indices, new_lambda


def MuCm(imgs, lbls):
    
    # op = np.random.choice(2, 1, p=(.5, .5)).item()
    
    gamma = np.random.beta(ALPHA, ALPHA, imgs.size(0))
    gamma = np.concatenate([gamma[:, None], 1-gamma[:, None]], 1).max(1)
    
    shuffle = torch.randperm(imgs.size(0)).to(imgs.device)
    shfl_imgs, shfl_lbls = imgs[shuffle], lbls[shuffle]
        
    # if op == 0:
    gamma = imgs.new(gamma)
    gamma_img = gamma.view(-1, 1, 1, 1)
    new_imgs = imgs * gamma_img + shfl_imgs * (1-gamma_img)
            
    # elif op == 1:
    #     flattened_indices, gamma = rand_bboxes_FlantIndices(imgs.size(), gamma)
    #     gamma = imgs.new(gamma)
    #     imgs_newview = imgs.view(-1)
    #     flattened_indices_torch = torch.from_numpy(flattened_indices)
    #     imgs_newview[flattened_indices_torch] = shfl_imgs.view(-1)[flattened_indices_torch]
    #     new_imgs = imgs
            
    return new_imgs, shfl_lbls, gamma
                                           
def mixup_loss(preds, lbls, shfl_lbls, gamma):
    return (gamma * F.cross_entropy(preds, lbls, reduction='none') + (1-gamma) * F.cross_entropy(preds, shfl_lbls, reduction='none')).mean()


# def to_onehot(truth, num_class):
#     batch_size = len(truth)
#     onehot = torch.zeros(batch_size, num_class).to(truth.device)
#     onehot.scatter_(dim=1, index=truth.view(-1,1), value=1)
#     return onehot


# def cross_entropy_onehot_loss(logit, onehot):
#     batch_size, num_class = logit.shape
#     log_probability = -F.log_softmax(logit, 1)
#     loss = (log_probability*onehot)
#     loss = loss.sum(1)
#     loss = loss.mean()
#     return loss


# def criterion(logit, truth):
#     loss = []
    
#     for l, t in zip(logit, truth):
#         e = cross_entropy_onehot_loss(l, t)
#         loss.append(e)

#     return loss

# #https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
# def metric(probability, truth):

#     correct = []
#     for p,t in zip(probability,truth):
#         p = p.data.cpu().numpy()
#         t = t.data.cpu().numpy()
#         y = p.argmax(-1)
#         c = np.mean(y==t)
#         correct.append(c)

#     return correct


# def logit_to_probability(logit):
#     probability = [ F.softmax(l, 1) for l in logit ]
#     return probability


# def mixup(data, onehot):
#     batch_size = len(data)

#     alpha = 0.4
#     gamma = np.random.beta(alpha, alpha)
#     gamma = max(1-gamma, gamma)

#     # #mixup https://github.com/moskomule/mixup.pytorch/blob/master/main.py
#     perm = torch.randperm(batch_size).to(data.device)
#     perm_input  = data[perm]
#     perm_onehot = [t[perm] for t in onehot]
#     mix_input  = gamma*data + (1-gamma)*perm_input
#     mix_onehot = [gamma*t + (1-gamma)*perm_t for t, perm_t in zip(onehot, perm_onehot)]
#     return mix_input, mix_onehot


# def rand_bbox(size, lam):
#     W = size[2]
#     H = size[3]
#     cut_rat = np.sqrt(1. - lam)
#     cut_w = np.int(W * cut_rat)
#     cut_h = np.int(H * cut_rat)

#     # uniform
#     cx = np.random.randint(W)
#     cy = np.random.randint(H)

#     bbx1 = np.clip(cx - cut_w // 2, 0, W)
#     bby1 = np.clip(cy - cut_h // 2, 0, H)
#     bbx2 = np.clip(cx + cut_w // 2, 0, W)
#     bby2 = np.clip(cy + cut_h // 2, 0, H)

#     return bbx1, bby1, bbx2, bby2


# def cutmix(data, targets1, targets2, targets3, alpha):
#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_targets1 = targets1[indices]
#     shuffled_targets2 = targets2[indices]
#     shuffled_targets3 = targets3[indices]

#     lam = np.random.beta(alpha, alpha)
#     bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
#     data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
#     # adjust lambda to exactly match pixel ratio
#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

#     targets = (targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam)
#     return data, targets


# def cutmix_criterion(preds1, preds2, preds3, targets):
#     targets1, targets2, targets3, targets4, targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
#     loss = .7 * (lam * cross_entropy_onehot_loss(preds1, targets1) + (1 - lam) * cross_entropy_onehot_loss(preds1, targets2)) + \
#            .1 * (lam * cross_entropy_onehot_loss(preds2, targets3) + (1 - lam) * cross_entropy_onehot_loss(preds2, targets4)) + \
#            .2 * (lam * cross_entropy_onehot_loss(preds3, targets5) + (1 - lam) * cross_entropy_onehot_loss(preds3, targets6))
#     return loss.mean()