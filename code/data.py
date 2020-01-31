# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import numpy
import six
import torch
from torch.utils.data.dataset import Dataset

from config import *


class DS_TRN(Dataset):

    def __init__(self, img_arr, lbl_arr, sampling_weight=None, transform=None):
        self.img_arr            = img_arr
        self.labels             = lbl_arr
        self.transform          = transform

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        return self.img_lst[index], self.labels[index]

    def __len__(self):
        """Returns the number of data points."""
        self.img_lst.shape[0]


class DS_TST(Dataset):
    pass