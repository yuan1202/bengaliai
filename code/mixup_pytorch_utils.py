import numpy as np
import skimage
import torch
import torch.nn.functional as F
from torch import nn


ALPHA = 1.


def rand_bboxes_FlantIndices(size, lam, clamp=None):
    B = size[0]
    H = size[2]
    W = size[3]
    
    index_addition = np.arange(B) * H * W
    
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


def random_half_FlantIndices(size, lam=None):
    B = size[0]
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


def half_bboxes_FlantIndices(size, lam):
    B = size[0]
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


def logit_to_probability(logit):
    probability = [F.softmax(l, 1) for l in logit]
    return probability


def to_onehot(truth, num_class):
    batch_size = len(truth)
    onehot = torch.zeros(batch_size, num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1, 1), value=1)
    return onehot


def cross_entropy_onehot_loss(logit, onehot):
    batch_size, num_class = logit.shape
    log_probability = -F.log_softmax(logit, 1)
    loss = (log_probability * onehot)
    loss = loss.sum(1)
    loss = loss.mean()
    return loss


def criterion(logits, truths):
    return [cross_entropy_onehot_loss(l, t) for l, t in zip(logits, truths)]


def mixup_loss(preds, lbls, shfl_lbls, gamma):
    return gamma * F.cross_entropy(preds, lbls, reduction='none') + (1-gamma) * F.cross_entropy(preds, shfl_lbls, reduction='none')


def Mu_InnerPeace_fastai(imgs, lbls):
    '''do MU/CM/Null in one batch
    '''
    # split for MU, and Null
    B = imgs.size(0)
    divide = 2
    unit = B // divide
    residual = B % divide
    split = [unit if i < divide - 1 else unit + residual for i in range(divide)]
    
    imgs_mu, imgs_null = torch.split(imgs, split, dim=0)
    lbls_mu, lbls_null = torch.split(lbls, split, dim=0)
    
    # ---------------- mixup ----------------
    perm_mu = torch.randperm(imgs_mu.size(0)).to(imgs_mu.device)
    perm_imgs_mu = imgs_mu[perm_mu]
    perm_lbls_mu = lbls_mu[perm_mu]
    
    gamma_mu = np.random.beta(ALPHA, ALPHA, imgs_mu.size(0))
    gamma_mu = np.concatenate([gamma_mu[:, None], 1-gamma_mu[:, None]], 1).max(1)
    
    gamma_mu = imgs_mu.new(gamma_mu)
    gamma_mu_newview = gamma_mu.view(-1, 1, 1, 1)
        
    new_imgs_mu = imgs_mu * gamma_mu_newview + perm_imgs_mu * (1 - gamma_mu_newview)
    gamma_mu_newview = gamma_mu.view(-1, 1)
    
    # ---------------- null ----------------
    gamma_null = np.ones(imgs_null.size(0))
    gamma_null = imgs_null.new(gamma_null)
    
    # ---------------- back together and shuffle ----------------
    new_imgs = torch.cat((new_imgs_mu, imgs_null), dim=0)
    new_lbls = torch.cat((perm_lbls_mu, lbls_null), dim=0)
    new_gamma = torch.cat([gamma_mu, gamma_null])
    
    perm_final = torch.randperm(new_imgs.size(0)).to(new_imgs.device)
            
    return new_imgs[perm_final], lbls[perm_final], new_lbls[perm_final], new_gamma[perm_final]


def MuCm_InnerPeace_fastai(imgs, lbls):
    '''do MU/CM/Null in one batch
    '''
    # split for MU, CM and Null
    B = imgs.size(0)
    divide = 3
    unit = B // divide
    residual = B % divide
    split = [unit if i < divide - 1 else unit + residual for i in range(divide)]
    
    imgs_mu, imgs_cm, imgs_null = torch.split(imgs, split, dim=0)
    lbls_mu, lbls_cm, lbls_null = torch.split(lbls, split, dim=0)
    
    # ---------------- mixup ----------------
    perm_mu = torch.randperm(imgs_mu.size(0)).to(imgs_mu.device)
    perm_imgs_mu = imgs_mu[perm_mu]
    perm_lbls_mu = lbls_mu[perm_mu]
    
    gamma_mu = np.random.beta(ALPHA, ALPHA, imgs_mu.size(0))
    gamma_mu = np.concatenate([gamma_mu[:, None], 1-gamma_mu[:, None]], 1).max(1)
    
    gamma_mu = imgs_mu.new(gamma_mu)
    gamma_mu_newview = gamma_mu.view(-1, 1, 1, 1)
        
    new_imgs_mu = imgs_mu * gamma_mu_newview + perm_imgs_mu * (1 - gamma_mu_newview)
    gamma_mu_newview = gamma_mu.view(-1, 1)
            
    # ---------------- cutmix ----------------
    perm_cm = torch.randperm(imgs_cm.size(0)).to(imgs_cm.device)
    perm_imgs_cm = imgs_cm[perm_cm]
    perm_lbls_cm = lbls_cm[perm_cm]
    
    gamma_cm = np.random.beta(ALPHA, ALPHA, imgs_cm.size(0))
    gamma_cm = np.concatenate([gamma_cm[:, None], 1-gamma_cm[:, None]], 1).max(1)
    
    flattened_indices, gamma_cm = half_bboxes_FlantIndices(imgs_cm.size(), gamma_cm)
    gamma_cm = imgs_cm.new(gamma_cm)
    gamma_cm_newview = gamma_cm.view(-1, 1)
    imgs_cm_newview = imgs_cm.view(-1)
    flattened_indices_torch = torch.from_numpy(flattened_indices)
    imgs_cm_newview[flattened_indices_torch] = perm_imgs_cm.view(-1)[flattened_indices_torch]
    
    new_imgs_cm = imgs_cm
    
    # ---------------- null ----------------
    gamma_null = np.ones(imgs_null.size(0))
    gamma_null = imgs_null.new(gamma_null)
    # ---------------- back together and shuffle ----------------
    new_imgs = torch.cat((new_imgs_mu, new_imgs_cm, imgs_null), dim=0)
    new_lbls = torch.cat((perm_lbls_mu, perm_lbls_cm, lbls_null), dim=0)
    new_gamma = torch.cat([gamma_mu, gamma_cm, gamma_null])
    
    perm_final = torch.randperm(new_imgs.size(0)).to(new_imgs.device)
            
    return new_imgs[perm_final], lbls[perm_final], new_lbls[perm_final], new_gamma[perm_final]


def MuCm_InnerPeace(imgs, lbls_oh):
    '''do MU/CM/Null in one batch
    '''
    # split for MU, CM and Null
    B = imgs.size(0)
    divide = 3
    unit = B // divide
    residual = B % divide
    split = [unit if i < divide - 1 else unit + residual for i in range(divide)]
    
    imgs_mu, imgs_cm, imgs_null = torch.split(imgs, split, dim=0)
    lbls_oh_g, lbls_oh_v, lbls_oh_c = lbls_oh
    lbls_oh_g_mu, lbls_oh_g_cm, lbls_oh_g_null = torch.split(lbls_oh_g, split, dim=0)
    lbls_oh_v_mu, lbls_oh_v_cm, lbls_oh_v_null = torch.split(lbls_oh_v, split, dim=0)
    lbls_oh_c_mu, lbls_oh_c_cm, lbls_oh_c_null = torch.split(lbls_oh_c, split, dim=0)
    
    # ---------------- mixup ----------------
    perm_mu = torch.randperm(imgs_mu.size(0)).to(imgs_mu.device)
    perm_imgs_mu = imgs_mu[perm_mu]
    perm_lbls_oh_g_mu, perm_lbls_oh_v_mu, perm_lbls_oh_c_mu = lbls_oh_g_mu[perm_mu], lbls_oh_v_mu[perm_mu], lbls_oh_c_mu[perm_mu]
    
    gamma_mu = np.random.beta(ALPHA, ALPHA, imgs_mu.size(0))
    gamma_mu = np.concatenate([gamma_mu[:, None], 1-gamma_mu[:, None]], 1).max(1)
    
    gamma_mu = imgs_mu.new(gamma_mu)
    gamma_mu = gamma_mu.view(-1, 1, 1, 1)
        
    new_imgs_mu = imgs_mu * gamma_mu + perm_imgs_mu * (1 - gamma_mu)
    gamma_mu = gamma_mu.view(-1, 1)
    new_lbls_oh_g_mu = gamma_mu * lbls_oh_g_mu + (1 - gamma_mu) * perm_lbls_oh_g_mu
    new_lbls_oh_v_mu = gamma_mu * lbls_oh_v_mu + (1 - gamma_mu) * perm_lbls_oh_v_mu
    new_lbls_oh_c_mu = gamma_mu * lbls_oh_c_mu + (1 - gamma_mu) * perm_lbls_oh_c_mu
            
    # ---------------- cutmix ----------------
    perm_cm = torch.randperm(imgs_cm.size(0)).to(imgs_cm.device)
    perm_imgs_cm = imgs_cm[perm_cm]
    perm_lbls_oh_g_cm, perm_lbls_oh_v_cm, perm_lbls_oh_c_cm = lbls_oh_g_cm[perm_cm], lbls_oh_v_cm[perm_cm], lbls_oh_c_cm[perm_cm]
    
    gamma_cm = np.random.beta(ALPHA, ALPHA, imgs_cm.size(0))
    gamma_cm = np.concatenate([gamma_cm[:, None], 1-gamma_cm[:, None]], 1).max(1)
    
    flattened_indices, gamma_cm = random_half_FlantIndices(imgs_cm.size(), gamma_cm)
    gamma_cm = imgs_cm.new(gamma_cm).view(-1, 1)
    imgs_cm_newview = imgs_cm.view(-1)
    flattened_indices_torch = torch.from_numpy(flattened_indices)
    imgs_cm_newview[flattened_indices_torch] = perm_imgs_cm.view(-1)[flattened_indices_torch]
    
    new_imgs_cm = imgs_cm
    new_lbls_oh_g_cm = gamma_cm * lbls_oh_g_cm + (1 - gamma_cm) * perm_lbls_oh_g_cm
    new_lbls_oh_v_cm = gamma_cm * lbls_oh_v_cm + (1 - gamma_cm) * perm_lbls_oh_v_cm
    new_lbls_oh_c_cm = gamma_cm * lbls_oh_c_cm + (1 - gamma_cm) * perm_lbls_oh_c_cm
    
    # ---------------- null ----------------
    
    # ---------------- back together and shuffle ----------------
    new_imgs = torch.cat((new_imgs_mu, new_imgs_cm, imgs_null), dim=0)
    new_lbls_oh_g = torch.cat((new_lbls_oh_g_mu, new_lbls_oh_g_cm, lbls_oh_g_null), dim=0)
    new_lbls_oh_v = torch.cat((new_lbls_oh_v_mu, new_lbls_oh_v_cm, lbls_oh_v_null), dim=0)
    new_lbls_oh_c = torch.cat((new_lbls_oh_c_mu, new_lbls_oh_c_cm, lbls_oh_c_null), dim=0)
    
    perm_final = torch.randperm(new_imgs.size(0)).to(new_imgs.device)
            
    return new_imgs[perm_final], (new_lbls_oh_g[perm_final], new_lbls_oh_v[perm_final], new_lbls_oh_c[perm_final])


def Mu_InnerPeace(imgs, lbls_oh):
    '''do MU/CM/Null in one batch
    '''
    # split for MU, CM and Null
    B = imgs.size(0)
    divide = 2
    unit = B // divide
    residual = B % divide
    split = [unit if i < divide - 1 else unit + residual for i in range(divide)]
    
    imgs_mu, imgs_null = torch.split(imgs, split, dim=0)
    lbls_oh_g, lbls_oh_v, lbls_oh_c = lbls_oh
    lbls_oh_g_mu, lbls_oh_g_null = torch.split(lbls_oh_g, split, dim=0)
    lbls_oh_v_mu, lbls_oh_v_null = torch.split(lbls_oh_v, split, dim=0)
    lbls_oh_c_mu, lbls_oh_c_null = torch.split(lbls_oh_c, split, dim=0)
    
    # ---------------- mixup ----------------
    perm_mu = torch.randperm(imgs_mu.size(0)).to(imgs_mu.device)
    perm_imgs_mu = imgs_mu[perm_mu]
    perm_lbls_oh_g_mu, perm_lbls_oh_v_mu, perm_lbls_oh_c_mu = lbls_oh_g_mu[perm_mu], lbls_oh_v_mu[perm_mu], lbls_oh_c_mu[perm_mu]
    
    gamma_mu = np.random.beta(ALPHA, ALPHA, imgs_mu.size(0))
    gamma_mu = np.concatenate([gamma_mu[:, None], 1-gamma_mu[:, None]], 1).max(1)
    
    gamma_mu = imgs_mu.new(gamma_mu)
    gamma_mu = gamma_mu.view(-1, 1, 1, 1)
        
    new_imgs_mu = imgs_mu * gamma_mu + perm_imgs_mu * (1 - gamma_mu)
    gamma_mu = gamma_mu.view(-1, 1)
    new_lbls_oh_g_mu = gamma_mu * lbls_oh_g_mu + (1 - gamma_mu) * perm_lbls_oh_g_mu
    new_lbls_oh_v_mu = gamma_mu * lbls_oh_v_mu + (1 - gamma_mu) * perm_lbls_oh_v_mu
    new_lbls_oh_c_mu = gamma_mu * lbls_oh_c_mu + (1 - gamma_mu) * perm_lbls_oh_c_mu
    
    # ---------------- null ----------------
    
    # ---------------- back together and shuffle ----------------
    new_imgs = torch.cat((new_imgs_mu, imgs_null), dim=0)
    new_lbls_oh_g = torch.cat((new_lbls_oh_g_mu, lbls_oh_g_null), dim=0)
    new_lbls_oh_v = torch.cat((new_lbls_oh_v_mu, lbls_oh_v_null), dim=0)
    new_lbls_oh_c = torch.cat((new_lbls_oh_c_mu, lbls_oh_c_null), dim=0)
    
    perm_final = torch.randperm(new_imgs.size(0)).to(new_imgs.device)
            
    return new_imgs[perm_final], (new_lbls_oh_g[perm_final], new_lbls_oh_v[perm_final], new_lbls_oh_c[perm_final])


def MuCm(imgs, lbls_oh):
    
    B = imgs.size(0)
    
    gamma = np.random.beta(ALPHA, ALPHA, imgs.size(0))
    gamma = np.concatenate([gamma[:, None], 1-gamma[:, None]], 1).max(1)
    
    perm = torch.randperm(imgs.size(0)).to(imgs.device)
    perm_imgs = imgs[perm]
    perm_lbls_oh = [l[perm] for l in lbls_oh]
    
    op = np.random.choice(3, 1, p=(.3, .3, .4)).item()
    if op == 0:
        gamma = imgs.new(gamma)
        gamma = gamma.view(-1, 1, 1, 1)
        new_imgs = imgs * gamma + perm_imgs * (1 - gamma)
        gamma = gamma.view(-1, 1)
        new_lbls_oh = [gamma * l + (1 - gamma) * perm_l for l, perm_l in zip(lbls_oh, perm_lbls_oh)]
            
    elif op == 1:
        flattened_indices, gamma = rand_bboxes_FlantIndices(imgs.size(), gamma)
        gamma = imgs.new(gamma).view(-1, 1)
        imgs_newview = imgs.view(-1)
        flattened_indices_torch = torch.from_numpy(flattened_indices)
        imgs_newview[flattened_indices_torch] = perm_imgs.view(-1)[flattened_indices_torch]
        new_imgs = imgs
        new_lbls_oh = [gamma * l + (1 - gamma) * perm_l for l, perm_l in zip(lbls_oh, perm_lbls_oh)]
    else:
        gamma = np.ones(imgs.size(0))
        gamma = imgs.new(gamma)
        new_imgs = imgs
        new_lbls_oh = lbls_oh
            
    return new_imgs, new_lbls_oh


def Mu(imgs, lbls_oh):
    
    gamma = np.random.beta(ALPHA, ALPHA, imgs.size(0))
    gamma = np.concatenate([gamma[:, None], 1-gamma[:, None]], 1).max(1)
    
    perm = torch.randperm(imgs.size(0)).to(imgs.device)
    perm_imgs = imgs[perm]
    perm_lbls_oh = [l[perm] for l in lbls_oh]
        
    op = np.random.choice(2, 1, p=(.5, .5)).item()
    if op == 0:
        gamma = imgs.new(gamma)
        gamma = gamma.view(-1, 1, 1, 1)
        new_imgs = imgs * gamma + perm_imgs * (1 - gamma)
        gamma = gamma.view(-1, 1)
        new_lbls_oh = [gamma * l + (1 - gamma) * perm_l for l, perm_l in zip(lbls_oh, perm_lbls_oh)]
    elif op == 1:
        flattened_indices, _ = rand_bboxes_FlantIndices(imgs.size(), gamma)
        gamma = np.ones(imgs.size(0))
        imgs_newview = imgs.view(-1)
        flattened_indices_torch = torch.from_numpy(flattened_indices)
        imgs_newview[flattened_indices_torch] = 0
        new_imgs = imgs
    else:
        new_imgs = imgs
        new_lbls_oh = lbls_oh
            
    return new_imgs, new_lbls_oh