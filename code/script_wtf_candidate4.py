
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

from models_mg import mdl_res34_localpool_small

from callback_utils import SaveModelCallback
from mixup_fastai_utils import CmCallback
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
    if fold == 0:
        break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_raw_64112.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]

# =========================================================================================
# augs = iaa.OneOf(
#     [
#         iaa.Affine(rotate=(-15, 15)),# translate_percent={"x": (-.1, .1), "y": (-.1, .1)}),
#         iaa.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}),# translate_percent={"x": (-.1, .1), "y": (-.1, .1)}),
#         iaa.PerspectiveTransform(scale=.08, keep_size=True),
#     ]
# )

augs = iaa.SomeOf(
    (0, 2),
    [
        iaa.OneOf(
            [
                iaa.Affine(rotate=(-15, 15), translate_percent={"x": (-.07, .07), "y": (-.07, .07)}),
                iaa.Affine(shear={'x': (-10, 10), 'y': (-10, 10)}, translate_percent={"x": (-.07, .07), "y": (-.07, .07)}),
                iaa.PerspectiveTransform(scale=.09, keep_size=True),
            ]
        ),
        iaa.DirectedEdgeDetect(alpha=(.05, .95), direction=(0.0, 1.0)),
    ],
    random_order=True,
)

# =========================================================================================

batch_size = 64 # 64 is important as the fit_one_cycle arguments are probably tuned for this batch size

training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=augs, RGB=False)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls, RGB=False)

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4, shuffle=False)

data_bunch = DataBunch(train_dl=training_loader, valid_dl=validation_loader)

# =========================================================================================

classifier = mdl_res34_localpool_small()

# classifier.load_state_dict(torch.load('mdl_res34localpool_168168_lessaugs_mucm_fixed_adam_onecycle_fld3of5_backup.pth'))

learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_combine_weighted_v2(),
    opt_func=Adam,
    metrics=[Metric_grapheme(), Metric_vowel(), Metric_consonant(), Metric_tot()]
)

# learn.clip_grad = 1.0
name = 'mdl_res34localpoolsmall_64112_lessaugs_cm_fixed_adam_onecycle_fld0of5'

logger = CSVLogger(learn, name)

# =========================================================================================

learn.fit_one_cycle(
    160,
    max_lr=0.001,
    wd=0.0,
    moms=.5,
    pct_start=0.0,
    div_factor=100.,
    #final_div=100.,
    callbacks=[logger, SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), CmCallback(learn, alpha=1.)]
)

# learn.fit(
#     100,
#     lr=0.001,
#     wd=0.0,
#     callbacks=[
#         logger, 
#         SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), 
#         MuCmCallback(learn),
#         ReduceLROnPlateauCallback(learn, patience=10, factor=0.5, min_lr=.00001)
#     ]
# )