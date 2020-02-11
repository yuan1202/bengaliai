#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import bloscpack as bp

import imgaug as ia
import imgaug.augmenters as iaa

from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from torch.utils.data.dataloader import DataLoader

import fastai
from fastai.vision import *

from optim import Over9000
from data import Bengaliai_DS, Balanced_Sampler
from model import *
from model_utils import *


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


augs = iaa.SomeOf(
    (0, 2),
    [
        iaa.SomeOf(
            (1, 2),
            [
                iaa.OneOf(
                    [
                        iaa.Affine(scale={"x": (0.8, 1.1), "y": (0.8, 1.1)}, rotate=(-10, 10), shear=(-10, 10)),
                        iaa.PerspectiveTransform(scale=.09, keep_size=True),
                    ]
                ),
                iaa.PiecewiseAffine(scale=(0.02, 0.03)),
            ],
            random_order=True
        ),
        iaa.OneOf(
            [
                iaa.DirectedEdgeDetect(alpha=(.6, .8), direction=(0.0, 1.0)),
                iaa.Emboss(alpha=(.5, 1.), strength=(.1, 4)),
            ]
        ),
    ],
    random_order=True
)


# In[4]:


pdf = pd.read_csv('../input/train.csv')

unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
for trn_ndx, vld_ndx in skf.split(pdf['grapheme_code'], pdf['grapheme_code']):
    break

# skf = MultilabelStratifiedKFold(n_splits=4, shuffle=True, random_state=42)
# for fold, (trn_ndx, vld_ndx) in enumerate(skf.split(pdf['image_id'].values.reshape(-1, 1), pdf.loc[:, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values.reshape(-1, 3))):
#     if fold == 0:
#         break
    
trn_pdf = pdf.iloc[trn_ndx, :]
trn_pdf.reset_index(inplace=True, drop=True)
imgs = bp.unpack_ndarray_from_file('../features/train_images.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx, 0:1]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx, 0:1]


# In[5]:


#sampler = Balanced_Sampler(trn_pdf, count_column='image_id', primary_group='grapheme_root', secondary_group=['vowel_diacritic', 'consonant_diacritic'], size=trn_imgs.shape[0])

training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=None)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls)

training_loader = DataLoader(training_set, batch_size=64, num_workers=6, shuffle=True) # , sampler=sampler , shuffle=True
validation_loader = DataLoader(validation_set, batch_size=64, num_workers=6, shuffle=False)

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


predictor = PretrainedCNN(out_dim=n_grapheme)
classifier = BengaliClassifier_Single(predictor)


# In[8]:


learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_single(),
    opt_func=Over9000,
    metrics=[Metric_grapheme()]
)

logger = CSVLogger(learn, 'Seresnext50_Multilabelstrat_Lightaugs_Singleclass_Stratbygrapheme_Mixup_Noaug')

learn.clip_grad = 1.0
learn.split([classifier.predictor.lin_layers])
# learn.split([classifier.head1])
learn.unfreeze()


# In[9]:


learn.fit_one_cycle(
    32,
    max_lr=slice(0.2e-2,1e-2),
    wd=[1e-3, 0.1e-1],
    pct_start=0.0,
    div_factor=100,
    callbacks=[logger, SaveModelCallback(learn, monitor='metric_idx', mode='max', name='Seresnext50_Multilabelstrat_Lightaugs_Singleclass_Stratbygrapheme_Mixup_Noaug'), MixUpCallback_Single(learn)] # 
)

