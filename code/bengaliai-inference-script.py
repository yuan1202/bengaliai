# Setup
!pip install ../input/pretrainedmodels/pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/ > /dev/null

import os
from tqdm import tqdm

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
import torchvision

from torch.utils.data import Dataset

import pretrainedmodels


# ==================================================================
# Params
BATCH_SIZE = 32
N_WORKERS = 4
N_EPOCHS = 5
CLEAN_THRESHOLD = 20
RESIZE = 128

HEIGHT = 137
WIDTH = 236
TARGET_SIZE = 256

INPUT_PATH = '/kaggle/input/bengaliai-cv19'
# INPUT_PATH = '../input/'

WEIGHTS_FILE = '/kaggle/input/seresnext-densenetstartersetup-noaug-mixup/Seresnext_DensenetStarterSetup_Noaug_Mixup.pth'
# WEIGHTS_FILE = './outputs/Seresnext_DensenetStarterSetup_Noaug_Mixup.pth'

print('==================================================================')
print('Setup dataset...')
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, pad=16):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < CLEAN_THRESHOLD] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img, (RESIZE, RESIZE))


class BengaliParquetDataset(Dataset):

    def __init__(self, dataframe):

        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp = 255 - self.data.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH).astype('uint8')
        img = crop_resize(np.round(tmp * (255. / tmp.max())).astype(np.uint8))[np.newaxis, :]

        image_id = self.data.iloc[idx, 0]

        return {
            'image_id': image_id,
            'image': img
        }


print('==================================================================')
print('Setup model...')
class LinearBlock(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False
    ):
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h
    
    
class PretrainedCNN(nn.Module):
    def __init__(self, out_dim, in_channels=1, model_name='se_resnext101_32x4d', use_bn=True, pretrained=None):
        super(PretrainedCNN, self).__init__()
        
        # convert channels to 3 to adapt to pre-trained model
        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        
        activation = F.leaky_relu
        
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h
    

class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

    def forward(self, x, y=None):
        pred = self.predictor(x)
        preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds


print('==================================================================')
print('Load model...')
n_grapheme = 168
n_vowel = 11
n_consonant = 7
n_total = n_grapheme + n_vowel + n_consonant

device = torch.device("cuda:0")

predictor = PretrainedCNN(out_dim=n_total)
model = BengaliClassifier(predictor)

model.load_state_dict(torch.load(WEIGHTS_FILE))
model.to(device)

results = []

print('==================================================================')
print('Initialise data pipeline...')
test_df = pd.read_csv(INPUT_PATH + '/test.csv')
submission_df = pd.read_csv(INPUT_PATH + '/sample_submission.csv')

pqs = [f for f in os.listdir(INPUT_PATH) if ('parquet' in f) and ('test' in f)]
pdf_tst = pd.concat([pd.read_parquet(os.path.join('../input', f)) for f in pqs], ignore_index=True)

test_dataset = BengaliParquetDataset(dataframe=pdf_tst)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)

print('==================================================================')
print('Run inference...')
model.eval()

for step, batch in enumerate(data_loader_test):
    inputs = batch["image"]
    image_ids = batch["image_id"]
    inputs = inputs.to(device, dtype=torch.float)

    out_graph, out_vowel, out_conso = model(inputs)
    out_graph = F.softmax(out_graph, dim=1).data.cpu().numpy().argmax(axis=1)
    out_vowel = F.softmax(out_vowel, dim=1).data.cpu().numpy().argmax(axis=1)
    out_conso = F.softmax(out_conso, dim=1).data.cpu().numpy().argmax(axis=1)

    for idx, image_id in enumerate(image_ids):
        results.append((image_id + '_consonant_diacritic', out_conso[idx]))
        results.append((image_id + '_grapheme_root', out_graph[idx]))
        results.append((image_id + '_vowel_diacritic', out_vowel[idx]))

                      
pd.DataFrame(results, columns=['row_id', 'target']).to_csv('./submission.csv', index=False)
