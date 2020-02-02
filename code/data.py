# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import numpy
import six

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler

from config import *


class DS_TRN(Dataset):

    def __init__(self, img_arr, lbl_arr, sampling_weight=None, transform=None):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        if self.transform:
            imgs = self.transform(self.img_arr[index])
        else:
            imgs = self.img_arr[index]
            
        imgs = imgs / 255.
            
        return imgs.astype('float32'), self.labels[index]

    def __len__(self):
        """Returns the number of data points."""
        return self.img_arr.shape[0]
    
    @property
    def is_empty(self):
        return self.__len__() == 0


class DS_TST(Dataset):
    pass


# -----------------------------------------------------------------
# list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 3, 0.1], 5, replacement=False))