
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

from torch.optim import SGD

from data import Bengaliai_DS
from models_mg import mdl_ResDenHybrid

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

skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=19550423)
for fold, (trn_ndx, vld_ndx) in enumerate(skf.split(pdf['image_id'].values.reshape(-1, 1), pdf.loc[:, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values)):
    if fold == 1:
        break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_raw.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]

# =========================================================================================

augs = iaa.OneOf(
    [
        iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            rotate=(-15, 15),
            shear={'x': (-15, 15), 'y': (-15, 15)},
        ),
        iaa.PerspectiveTransform(scale=.09, keep_size=True),
    ]
)

# =========================================================================================

batch_size = 64

training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs, RGB=True)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls, RGB=True)

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4, shuffle=False)

data_bunch = DataBunch(train_dl=training_loader, valid_dl=validation_loader)

# =========================================================================================

classifier = mdl_ResDenHybrid()

class SGD_m5(SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(momentum=0.5, *args, **kwargs)

learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_combine_weighted_v2(),
    opt_func=SGD_m5,
    metrics=[Metric_grapheme(), Metric_vowel(), Metric_consonant(), Metric_tot()]
)

name = 'mdl_ResDenHybrid_sgd_lessaugs_mucm_fixed_raw_onecycle_fld1of5'

logger = CSVLogger(learn, name)

# =========================================================================================

learn.fit_one_cycle(
    160,
    max_lr=0.05,
    wd=0.0,
    pct_start=0.0,
    div_factor=50.,
    final_div=100.,
    callbacks=[logger, SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), MuCmCallback(learn)]
)