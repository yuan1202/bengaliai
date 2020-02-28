#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import bloscpack as bp

from sklearn.model_selection import StratifiedKFold

import imgaug as ia
import imgaug.augmenters as iaa

from torch.utils.data.dataloader import DataLoader

import fastai
from fastai.vision import *
from fastai.callbacks import *

from optim import Over9000
from data import Bengaliai_DS
from heng.model import Net
from callback_utils import SaveModelCallback
from mixup_fastai_utils import CmCallback, MuCmCallback, MixUpCallback
from loss import Loss_combine_weighted_v2
from metric import Metric_grapheme, Metric_vowel, Metric_consonant, Metric_tot


# ---

# In[2]:


SEED = 42

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

# In[3]:


pdf = pd.read_csv('../input/train.csv')
unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
for trn_ndx, vld_ndx in skf.split(pdf['grapheme_code'], pdf['grapheme_code']):
    break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_raw_half.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]


# In[4]:


# augs = iaa.SomeOf(
#     (0, 2),
#     [
#         iaa.OneOf(
#             [
#                 iaa.Affine(scale={"x": (0.8, 1.), "y": (0.8, 1.)}, rotate=(-15, 15), shear=(-15, 15)),
#                 iaa.PerspectiveTransform(scale=.08, keep_size=True),
#             ]
#         ),
#         iaa.PiecewiseAffine(scale=(.02, .03)),
#     ],
#     random_order=True
# )

augs = iaa.SomeOf(
    (1, 3),
    [
        iaa.Affine(scale={"x": (0.8, 1.), "y": (0.8, 1.)}, rotate=(-15, 15), shear=(-15, 15)),
        iaa.PiecewiseAffine(scale=(0.02, 0.04)),
        iaa.DirectedEdgeDetect(alpha=(.01, .99), direction=(0.0, 1.0)),
    ],
    random_order=True
)


# In[5]:


batch_size = 64 # 64 is important as the fit_one_cycle arguments are probably tuned for this batch size

training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs, RGB=False)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls, RGB=False)

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=6, shuffle=True) # , sampler=sampler , shuffle=True
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=6, shuffle=False)

data_bunch = DataBunch(train_dl=training_loader, valid_dl=validation_loader)


# ---
# ### model

# In[6]:


device = 'cuda:0'
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant


# In[7]:


classifier = Net()


# In[8]:


learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_combine_weighted_v2(),
    opt_func=Over9000,
    metrics=[Metric_grapheme(), Metric_vowel(), Metric_consonant(), Metric_tot()]
)

name = 'new_baseline_hengs_LessAugs_211_Mu10_Wd0_fit160epochs'

logger = CSVLogger(learn, name)

# learn.unfreeze()


# In[ ]:


# learn.fit_one_cycle(
#     64,
#     max_lr=.01,
#     wd=0.,
#     pct_start=0.0,
#     div_factor=100,
#     callbacks=[logger, SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), MixUpCallback(learn, alpha=1.)]
# )

learn.fit(
    160,
    lr=.05,
    wd=0.,
    callbacks=[
        logger, 
        SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), 
        # MixUpCallback(learn, alpha=1.),
        ReduceLROnPlateauCallback(learn, patience=5, factor=0.5, min_lr=.0001)
    ]
)