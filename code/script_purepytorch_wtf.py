
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

# augs = iaa.SomeOf(
#     (0, 2),
#     [
#         iaa.OneOf(
#             [
#                 iaa.Affine(
#                     scale={"x": (0.8, 1.1), "y": (0.8, 1.1)},
#                     rotate=(-15, 15),
#                     shear={'x': (-15, 15), 'y': (-15, 15)},
#                 ),
#                 iaa.PerspectiveTransform(scale=.09, keep_size=True),
#             ]
#         ),
#     ],
#     random_order=True,
# )

augs = iaa.Affine(
    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
    rotate=(-15, 15),
    shear={'x': (-10, 10), 'y': (-10, 10)},
)

# =========================================================================================================================

pdf = pd.read_csv('../input/train.csv')
unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=19550423)
for fold, (trn_ndx, vld_ndx) in enumerate(skf.split(pdf['image_id'].values.reshape(-1, 1), pdf.loc[:, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)):
    if fold == 0:
        break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_size128_pad8_max_noclean.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]


training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs, split_label=True, RGB=True)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls, split_label=True, RGB=True)

batch_size = 64

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4, shuffle=False)

# =========================================================================================================================

N_EPOCHS = 150

checkpoint_name = 'purepytorch_wtf_sext50_lessaug_mucu_sgd_cosanneal_fld0'

# =========================================================================================================================

classifier = mdl_sext50().cuda()
classifier.load_state_dict(torch.load('purepytorch_wtf_sext50_lessaug_mucu_sgd_cosanneal_fld0_0.pth')['model'])
optimizer = Adam(classifier.parameters(), lr=.0005, weight_decay=0.0)

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
        
        optimizer.zero_grad()
        
        # pre-process before mixup
        trn_lbls_oh_batch = [to_onehot(l, c) for l, c in zip(trn_lbls_batch, (168, 11, 7))]
        
        # mixup
        trn_imgs_batch_mixup, trn_lbls_oh_batch_mixup, _ = Mu_InnerPeace(trn_imgs_batch, trn_lbls_oh_batch)
        
        # move to device
        trn_imgs_batch_mixup_device = trn_imgs_batch_mixup.cuda()
        trn_lbls_oh_batch_mixup_device = [l.cuda() for l in trn_lbls_oh_batch_mixup]
        
        # forward pass
        logits = classifier(trn_imgs_batch_mixup_device)

        losses = criterion(logits, trn_lbls_oh_batch_mixup_device)
        
        total_trn_loss = .8*losses[0] + .1*losses[1] + .1*losses[2]
        
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
            vld_lbls_batch_device = [l.cuda() for l in vld_lbls_batch]
            vld_lbls_batch_numpy = [l.detach().cpu().numpy() for l in vld_lbls_batch]
            
            # forward pass
            logits = classifier(vld_imgs_batch_device)
            
            # loss
            vld_lbls_batch_oh = [to_onehot(l, c) for l, c in zip(vld_lbls_batch_device, (168, 11, 7))]
            losses = criterion(logits, vld_lbls_batch_oh)
            
            total_vld_loss = .8*losses[0] + .1*losses[1] + .1*losses[2]
            # record
            epoch_vld_loss.append(total_vld_loss.item())
            
            # metrics
            pred_g = logits[0].argmax(axis=1).detach().cpu().numpy()
            pred_v = logits[1].argmax(axis=1).detach().cpu().numpy()
            pred_c = logits[2].argmax(axis=1).detach().cpu().numpy()
            epoch_vld_recall_g.append(recall_score(pred_g, vld_lbls_batch_numpy[0], average='macro', zero_division=0))
            epoch_vld_recall_v.append(recall_score(pred_v, vld_lbls_batch_numpy[1], average='macro', zero_division=0))
            epoch_vld_recall_c.append(recall_score(pred_c, vld_lbls_batch_numpy[2], average='macro', zero_division=0))
            
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

