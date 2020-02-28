
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

# from optim import Over9000
from torch.optim import SGD

from data import Bengaliai_DS_Heng
from heng.model import Net

from callback_utils import SaveModelCallback
from mixup_fastai_utils import CmCallback, MuCmCallback, MixUpCallback
from loss import Loss_combine_weighted_v2
from metric import Metric_grapheme, Metric_vowel, Metric_consonant, Metric_tot

# =========================================================================================

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

# =========================================================================================

pdf = pd.read_csv('../input/train.csv')
unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
for trn_ndx, vld_ndx in skf.split(pdf['grapheme_code'], pdf['grapheme_code']):
    break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_size128_raw.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]

# =========================================================================================

# augs = iaa.Sequential(
#     [
#         iaa.OneOf(
#             [
#                 iaa.Identity(),
#                 iaa.Affine(scale={"x": (0.7, 1.1), "y": (0.7, 1.1)}, rotate=(-15, 15), shear=(-15, 15)),
#             ]
#         ),
#         iaa.SomeOf(
#             (0, 2),
#             [
#                 iaa.PerspectiveTransform(scale=.1, keep_size=True),
#                 iaa.PiecewiseAffine(scale=(.02, .03)),
#                 iaa.imgcorruptlike.Snow(severity=1),
#                 #iaa.Cutout(nb_iterations=(1, 3), size=(.1, .2), squared=False, fill_mode="constant", cval=0),
#                 #iaa.CoarseDropout((.1, .2), size_percent=(.12, .2)),
#             ],
#             random_order=True
#         )
#     ]
# )

# =========================================================================================

batch_size = 64 # 64 is important as the fit_one_cycle arguments are probably tuned for this batch size

training_set = Bengaliai_DS_Heng(trn_imgs, trn_lbls, transform=True)
validation_set = Bengaliai_DS_Heng(vld_imgs, vld_lbls)

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=4, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=4, shuffle=False)

data_bunch = DataBunch(train_dl=training_loader, valid_dl=validation_loader)

# =========================================================================================

classifier = Net()

learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_combine_weighted_v2(),
    opt_func=SGD,
    metrics=[Metric_grapheme(), Metric_vowel(), Metric_consonant(), Metric_tot()]
)

name = 'fastai_heng_onecycle160epochs_hengsSize'

logger = CSVLogger(learn, name)

# =========================================================================================

learn.fit_one_cycle(
    cyc_len=128, max_lr=.01, wd=0.0, moms=0.5, div_factor=25, final_div=100, pct_start=0.02,
    callbacks=[logger, SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), MixUpCallback(learn, alpha=0.4)],
)