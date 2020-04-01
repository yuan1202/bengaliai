# =========================================================================================================================

import numpy as np
import pandas as pd
import bloscpack as bp

from sklearn.model_selection import StratifiedKFold

# import imgaug as ia
# import imgaug.augmenters as iaa

from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as pttf

import fastai
from fastai.vision import *
from fastai.callbacks import *

# from optim import Over9000
from torch.optim import Adam
from data import Bengaliai_DS
from models_mg import SimpleDense
from callback_utils import SaveModelCallback
from mixup_fastai_utils import CmCallback, MuCmCallback, MixUpCallback
from loss import Loss_combine_weighted_v2
from metric import Metric_grapheme, Metric_vowel, Metric_consonant, Metric_tot

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

pdf = pd.read_csv('../input/train.csv')
unique_grapheme = pdf['grapheme'].unique()
grapheme_code = dict([(g, c) for g, c in zip(unique_grapheme, np.arange(unique_grapheme.shape[0]))])
pdf['grapheme_code'] = [grapheme_code[g] for g in pdf['grapheme']]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19550423)
for trn_ndx, vld_ndx in skf.split(pdf['grapheme_code'], pdf['grapheme_code']):
    break
    
imgs = bp.unpack_ndarray_from_file('../features/train_images_size128_pad10_max_noclean.bloscpack')
lbls = pd.read_csv('../input/train.csv').iloc[:, 1:4].values

trn_imgs = imgs[trn_ndx]
trn_lbls = lbls[trn_ndx]
vld_imgs = imgs[vld_ndx]
vld_lbls = lbls[vld_ndx]

# =========================================================================================================================

tsfm = pttf.Compose(
    [
        pttf.ToPILImage(),
        pttf.RandomChoice(
            [
                pttf.RandomAffine(degrees=15, scale=(.9, 1.1), shear=15, resample=3),
                pttf.RandomPerspective(distortion_scale=0.5, p=1.),
            ],
        ),
        pttf.ToTensor(),
        #pttf.RandomErasing(p=0.5, scale=(.05, .1), ratio=(0.3, 3.3), value=0, inplace=False),
    ]
)

# =========================================================================================================================

batch_size = 64 # 64 is important as the fit_one_cycle arguments are probably tuned for this batch size

training_set = Bengaliai_DS(trn_imgs, trn_lbls, transform=tsfm)
validation_set = Bengaliai_DS(vld_imgs, vld_lbls)

training_loader = DataLoader(training_set, batch_size=batch_size, num_workers=6, shuffle=True) # , sampler=sampler , shuffle=True
validation_loader = DataLoader(validation_set, batch_size=batch_size, num_workers=6, shuffle=False)

data_bunch = DataBunch(train_dl=training_loader, valid_dl=validation_loader)

# =========================================================================================================================

classifier = SimpleDense()

#weight = torch.load('pytorch_model_short34.pth')
#classifier.load_state_dict(weight, strict=False)

learn = Learner(
    data_bunch,
    classifier,
    loss_func=Loss_combine_weighted_v2(),
    opt_func=Adam,
    metrics=[Metric_grapheme(), Metric_vowel(), Metric_consonant(), Metric_tot()]
)

name = 'pytorch_model_short34'

logger = CSVLogger(learn, name)

# =========================================================================================================================

# learn.fit_one_cycle(
#     cyc_len=120, max_lr=.01, wd=0.0, moms=0.5, div_factor=25, final_div=100, pct_start=0.,
#     callbacks=[logger, SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), MixUpCallback(learn)],
# )

learn.fit(
    120,
    lr=.01, wd=0.00001,
    callbacks=[
        logger, 
        SaveModelCallback(learn, monitor='metric_tot', mode='max', name=name), 
        MixUpCallback(learn),
        ReduceLROnPlateauCallback(learn, patience=10, factor=0.5, min_lr=.00001)
    ]
)