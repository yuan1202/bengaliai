
import numpy as np
import pandas as pd
import bloscpack as bp

from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import imgaug as ia
import imgaug.augmenters as iaa

from torch.utils.data.dataloader import DataLoader

import fastai
from fastai.vision import *
from fastai.callbacks import *

from optim import Over9000
from torch.optim import SGD, Adam

from data import Bengaliai_DS
# from senet_heng import seresxt50heng
from models_mg import mdl_sext50, mdl_res34Cmplx, mdl_stndense121
# from model import *

from callback_utils import SaveModelCallback
from mixup_fastai_utils import CmCallback, MuCmCallback, MixUpCallback
from loss import Loss_combine_weighted_v2
from metric import Metric_grapheme, Metric_vowel, Metric_consonant, Metric_tot

# =========================================================================================

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

# =========================================================================================

pdf = pd.read_csv('../input/train.csv')
unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=19550423)
for fold, (trn_ndx, vld_ndx) in enumerate(skf.split(pdf['image_id'].values.reshape(-1, 1), pdf.loc[:, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)):
    if fold == 8:
        break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_raw_98168.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]

# =========================================================================================

augs = iaa.OneOf(
    [
        iaa.Cutout(nb_iterations=1, position='uniform', size=(.09, .25), squared=False, fill_mode='constant', cval=0),
        iaa.PerspectiveTransform(scale=.1, keep_size=True),
        iaa.DirectedEdgeDetect(alpha=(.01, .99), direction=(0.0, 1.0)),
    ]
)

# =========================================================================================

batch_size = 64 # 64 is important as the fit_one_cycle arguments are probably tuned for this batch size

training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs, RGB=True)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls, RGB=True)

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4, shuffle=False)

data_bunch = DataBunch(train_dl=training_loader, valid_dl=validation_loader)

# =========================================================================================

classifier = mdl_stndense121()

# n_grapheme = 168
# n_vowel = 11
# n_consonant = 7
# n_total = n_grapheme + n_vowel + n_consonant
# predictor = PretrainedCNN(out_dim=n_total)
# classifier = BengaliClassifier(predictor)

learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_combine_weighted_v2(),
    opt_func=SGD,
    metrics=[Metric_grapheme(), Metric_vowel(), Metric_consonant(), Metric_tot()]
)

# learn.clip_grad = 1.0
# to be run
name = 'mdl_stndense121_muchlessaugs_nomucm'

logger = CSVLogger(learn, name)

# =========================================================================================

# learn.lr_find()

learn.fit_one_cycle(
    180,
    max_lr=0.05,
    wd=0.0,
    moms=(0.5, 0.8),
    pct_start=0.05,
    div_factor=50.,
    final_div=100.,
    callbacks=[logger, SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name)]
)

# learn.fit(
#     160,
#     lr=.05,
#     wd=0.0,
#     callbacks=[
#         logger, 
#         SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), 
#         ReduceLROnPlateauCallback(learn, patience=5, factor=0.67, min_lr=.0001)
#     ]
# )