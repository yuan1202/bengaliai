#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, random
import numpy as np
import pandas as pd
import bloscpack as bp
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.metrics import recall_score

import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from optim import Over9000

from data import Bengaliai_DS
from models_mg import Simple50GeMSingle
from mixup_pytorch_utils import mixup, mixup_loss
from loss import CenterLoss, AngularPenaltySMLoss
import utils

import cv2
cv2.setNumThreads(1)


# ---

# In[2]:


SEED = 19841202

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# ---
# ### data

# #### augmentation

# In[3]:

augs =  iaa.SomeOf(
    (0, 3),
    [
        iaa.OneOf(
            [
                iaa.Affine(scale={"x": (0.8, 1.), "y": (0.8, 1.)}, rotate=(-15, 15), shear=(-15, 15)),
                iaa.PerspectiveTransform(scale=.08, keep_size=True),
            ]
        ),
        iaa.PiecewiseAffine(scale=(0.02, 0.04)),
        iaa.CoarseDropout(p=(.1, .3), size_percent=(0.05, 0.1))
    ],
    random_order=True
)

# #### stratification

# In[4]:


pdf = pd.read_csv('../input/train.csv')
pdf['combo'] = pdf.apply(lambda row: '_'.join([str(row['grapheme_root']), str(row['vowel_diacritic']), str(row['consonant_diacritic'])]), axis=1)
unique_grapheme = pdf['combo'].unique() # 1292
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['combo']]

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=19841202)
for trn_ndx, vld_ndx in skf.split(pdf['grapheme_code'], pdf['grapheme_code']):
    break
    
trn_pdf = pdf.iloc[trn_ndx, :]
trn_pdf.reset_index(inplace=True, drop=True)
imgs = bp.unpack_ndarray_from_file('../features/train_images_size128_pad3.bloscpack')
lbls = pdf.loc[:, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic', 'grapheme_code']].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]


# In[5]:


training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls)

batch_size = 96

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=6, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=6, shuffle=False)


# ---
# ### model

# In[6]:


N_EPOCHS = 120

reduction = 'mean'

checkpoint_name = 'pytorch_model_test_epoch{:d}.pth'


# In[8]:


classifier = Simple50GeMSingle().cuda()

optimizer = Over9000(classifier.parameters(), lr=.005)

lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=.0001,)

# In[9]:


logger = utils.csv_logger(['training_loss', 'validation_loss', 'GRAPHEME_Recall', 'VOWEL_Recall', 'CONSONANT_Recall', 'Final_Recall'])


# In[10]:


for i in range(N_EPOCHS):
    logger.new_epoch()
    # train
    classifier.train()
    
    epoch_trn_loss = []
    epoch_vld_loss = []
    epoch_vld_recall_g, epoch_vld_recall_v, epoch_vld_recall_c, epoch_vld_recall_all = [], [], [], []
    
    for j, (trn_imgs_batch, trn_lbls_batch) in enumerate(training_loader):
        # move to device
        trn_imgs_batch_device = trn_imgs_batch.cuda()
        trn_lbls_batch_device = trn_lbls_batch.cuda()
        
        # mixup
        trn_imgs_batch_device_mixup, trn_lbls_batch_device_shfl, gamma = mixup(trn_imgs_batch_device, trn_lbls_batch_device, 1.)
        
        # forward pass
        logits_g, logits_v, logits_c = classifier(trn_imgs_batch_device_mixup)
        
        loss_g = mixup_loss(logits_g, trn_lbls_batch_device[:, 0], trn_lbls_batch_device_shfl[:, 0], gamma)
        loss_v = mixup_loss(logits_v, trn_lbls_batch_device[:, 1], trn_lbls_batch_device_shfl[:, 1], gamma)
        loss_c = mixup_loss(logits_c, trn_lbls_batch_device[:, 2], trn_lbls_batch_device_shfl[:, 2], gamma)
        
        #break
        
        total_trn_loss = .5*loss_g + .25*loss_v + .25*loss_c
        
        optimizer.zero_grad()
        
        total_trn_loss.backward()
        #clip_grad_value_(classifier.parameters(), 1.0)
        
        optimizer.step()
        
        # record
        epoch_trn_loss.append(total_trn_loss.item())
        
        utils.display_progress(len(training_loader), j+1, {'training_loss': epoch_trn_loss[-1]})
    
    #break
    # validation
    classifier.eval()
    #feat_loser.eval()
    
    with torch.no_grad():
        for k, (vld_imgs_batch, vld_lbls_batch) in enumerate(validation_loader):
            
            # move to device
            vld_imgs_batch_device = vld_imgs_batch.cuda()
            vld_lbls_batch_device = vld_lbls_batch.cuda()
            vld_lbls_batch_numpy = vld_lbls_batch.detach().cpu().numpy()
            
            # forward pass
            logits_g, logits_v, logits_c = classifier(vld_imgs_batch_device)
            
            # loss
            loss_g = F.cross_entropy(logits_g, vld_lbls_batch_device[:, 0], reduction=reduction)
            loss_v = F.cross_entropy(logits_v, vld_lbls_batch_device[:, 1], reduction=reduction)
            loss_c = F.cross_entropy(logits_c, vld_lbls_batch_device[:, 2], reduction=reduction)
            
            total_vld_loss = .5*loss_g + .25*loss_v + .25*loss_c
            # record
            epoch_vld_loss.append(total_vld_loss.item())
            
            # metrics
            pred_g, pred_v, pred_c = logits_g.argmax(axis=1).detach().cpu().numpy(), logits_v.argmax(axis=1).detach().cpu().numpy(), logits_c.argmax(axis=1).detach().cpu().numpy()
            epoch_vld_recall_g.append(recall_score(pred_g, vld_lbls_batch_numpy[:, 0], average='macro', zero_division=0))
            epoch_vld_recall_v.append(recall_score(pred_v, vld_lbls_batch_numpy[:, 1], average='macro', zero_division=0))
            epoch_vld_recall_c.append(recall_score(pred_c, vld_lbls_batch_numpy[:, 2], average='macro', zero_division=0))
            
            # display progress
            utils.display_progress(len(validation_loader), k+1, {'validation_loss': epoch_vld_loss[-1]})
    #break
    epoch_vld_recall_g, epoch_vld_recall_v, epoch_vld_recall_c = np.mean(epoch_vld_recall_g), np.mean(epoch_vld_recall_v), np.mean(epoch_vld_recall_c)
    final_recall = np.average([epoch_vld_recall_g, epoch_vld_recall_v, epoch_vld_recall_c], weights=[2, 1, 1])
    
    entry = {
        'training_loss': np.mean(epoch_trn_loss),
        'validation_loss': np.mean(epoch_vld_loss),
        'GRAPHEME_Recall': epoch_vld_recall_g,
        'VOWEL_Recall': epoch_vld_recall_v,
        'CONSONANT_Recall': epoch_vld_recall_c,
        'Final_Recall': final_recall,
    }
    
    utils.display_progress(N_EPOCHS, i+1, entry, postfix='Epochs', persist=True)
    
    # ----------------------------------
    # save model
    if entry['validation_loss'] < np.nan_to_num(logger.log['validation_loss'].min(), nan=100.):
        print('Saving new best weight.')
        torch.save(
            {
                'epoch': i,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 
            os.path.join('./', checkpoint_name.format(i)),
        )
    
    # ----------------------------------
    # log
    logger.enter(entry)
    logger.save('./{}.csv'.format(checkpoint_name))
    lr_scheduler.step(total_vld_loss.item())

