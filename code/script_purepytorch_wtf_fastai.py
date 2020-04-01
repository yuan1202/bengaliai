
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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# from apex import amp

# from optim import Over9000
from torch.optim import SGD, Adam

from data import Bengaliai_DS
from models_mg import mdl_sext50
# from senet_heng import seresxt50heng
from mixup_pytorch_utils import *
# from loss import CenterLoss, AngularPenaltySMLoss
from senet_mod import SENetMod

import utils


# =========================================================================================================================

SEED = 19550423

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)


# =========================================================================================================================
augs = iaa.OneOf(
    [
        iaa.Affine(
            rotate=(-15, 15),
            shear={'x': (-10, 10), 'y': (-10, 10)},
            translate_percent={"x": (-.1, .1), "y": (-.1, .1)},
        ),
        iaa.PerspectiveTransform(scale=.09, keep_size=True),
    ]
)

# =========================================================================================================================

pdf = pd.read_csv('../input/train.csv')
unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=19550423)
for fold, (trn_ndx, vld_ndx) in enumerate(skf.split(pdf['image_id'].values.reshape(-1, 1), pdf.loc[:, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)):
    if fold == 4:
        break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_raw_98168.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]


training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs, RGB=True)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls, RGB=True)

batch_size = 64

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4, shuffle=False)

# =========================================================================================================================

N_EPOCHS = 150

checkpoint_name = 'purepytorch_wtf_sext50_98168_lessaug_mu_sgd_cosanneal_fld4'

reduction = 'mean'
# =========================================================================================================================

classifier = mdl_sext50().cuda()
# classifier.load_state_dict(torch.load('purepytorch_wtf_sext50_lessaug_mucu_sgd_cosanneal_fld0_0.pth')['model'])
optimizer = SGD(classifier.parameters(), lr=.05, weight_decay=0.0, momentum=.5)

lr_scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=.00001)

# =========================================================================================================================

logger = utils.csv_logger(['training_loss', 'validation_loss', 'GRAPHEME_Recall', 'VOWEL_Recall', 'CONSONANT_Recall', 'Final_Recall'])

for i in range(N_EPOCHS):
    logger.new_epoch()
    # train
    classifier.train()
    
    epoch_trn_loss = []
    epoch_vld_loss = []
    epoch_vld_recall_g, epoch_vld_recall_v, epoch_vld_recall_c, epoch_vld_recall_all = [], [], [], []
    
    for j, (trn_imgs_batch, trn_lbls_batch) in enumerate(training_loader):
        
        # mixup / cutmix
        trn_imgs_batch_mixup, trn_lbls_batch, trn_lbls_batch_mixup, gamma = Mu_InnerPeace_fastai(trn_imgs_batch, trn_lbls_batch)
        
        # move to device
        trn_imgs_batch_mixup_device = trn_imgs_batch_mixup.cuda()
        trn_lbls_batch_device = trn_lbls_batch.cuda()
        trn_lbls_batch_mixup_device = trn_lbls_batch_mixup.cuda()
        gamma_device = gamma.cuda()
        
        # ready optimizer
        optimizer.zero_grad()
        
        # forward pass
        logits_g, logits_v, logits_c = classifier(trn_imgs_batch_mixup_device)

        loss_g = mixup_loss(logits_g, trn_lbls_batch_device[:, 0], trn_lbls_batch_mixup_device[:, 0], gamma_device).mean()
        loss_v = mixup_loss(logits_v, trn_lbls_batch_device[:, 1], trn_lbls_batch_mixup_device[:, 1], gamma_device).mean()
        loss_c = mixup_loss(logits_c, trn_lbls_batch_device[:, 2], trn_lbls_batch_mixup_device[:, 2], gamma_device).mean()
        
        total_trn_loss = .5*loss_g + .25*loss_v + .25*loss_c
        
        total_trn_loss.backward()
        optimizer.step()
        
        # record
        epoch_trn_loss.append(total_trn_loss.item())
        
        utils.display_progress(len(training_loader), j+1, {'training_loss': epoch_trn_loss[-1]})
    
    # validation
    classifier.eval()
    
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
            pred_g = logits_g.argmax(axis=1).detach().cpu().numpy()
            pred_v = logits_v.argmax(axis=1).detach().cpu().numpy()
            pred_c = logits_c.argmax(axis=1).detach().cpu().numpy()
            epoch_vld_recall_g.append(recall_score(pred_g, vld_lbls_batch_numpy[:, 0], average='macro', zero_division=0))
            epoch_vld_recall_v.append(recall_score(pred_v, vld_lbls_batch_numpy[:, 1], average='macro', zero_division=0))
            epoch_vld_recall_c.append(recall_score(pred_c, vld_lbls_batch_numpy[:, 2], average='macro', zero_division=0))
            
            # display progress
            utils.display_progress(len(validation_loader), k+1, {'validation_loss': epoch_vld_loss[-1]})
    
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
            os.path.join('./', checkpoint_name + '.pth'),
        )
    
    # ----------------------------------
    # log
    logger.enter(entry)
    logger.save('./{}.csv'.format(checkpoint_name))
    lr_scheduler.step()

