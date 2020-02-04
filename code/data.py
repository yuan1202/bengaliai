# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler

from config import *


class DS_TRN(Dataset):

    def __init__(self, img_arr, lbl_arr, sampling_weight=None, transform=None, norm=True):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.norm              = norm

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        if self.transform:
            img = self.transform(image=self.img_arr[index][0])[np.newaxis, :]
        else:
            img = self.img_arr[index]
            
        img = img / 255.
        
        if self.norm:
            img = (img - 0.0692) / 0.2051
            
        return img.astype('float32'), self.labels[index]

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