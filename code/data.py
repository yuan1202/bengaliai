# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import itertools, random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

from config import *


class Bengaliai_DS(Dataset):

    def __init__(self, img_arr, lbl_arr, sampling_weight=None, transform=None, norm=True):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.norm              = norm

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        if self.transform:
            img = self.transform(image=self.img_arr[index])[np.newaxis, :]
        else:
            img = self.img_arr[index][np.newaxis, :]
            
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


# -----------------------------------------------------------------
class Balanced_Sampler(Sampler):
    
    def __init__(self, pdf, count_column, primary_group, secondary_group, size):
        
        self.gb = pdf.groupby(primary_group)
        
        self.size = size
        
        self._g_indices = []
        self._g_weights = []
        
        # equal opportunity at this level
        for g in self.gb:
            indices = []
            weights = []
            sub_gb = g[1].groupby(secondary_group)
            # weighted drawing between sub-groups
            for sub_g in sub_gb:
                indices.append(sub_g[1].index.tolist())
                weights.append(sub_g[1].shape[0])
                
            # post process weights for this group
            weights = np.array(weights)
            weights = weights.sum() / weights
            
            self._g_weights.append(
                list(
                    itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
                )
            )
            self._g_indices.append(list(itertools.chain(*indices)))
            
    def __len__(self):
        return self.size
    
    def __iter__(self):
        samples_per_group = np.round(self.size / len(self.gb)).astype(int)
        samples = [random.choices(population=p, weights=w, k=samples_per_group) for p, w in zip(self._g_indices, self._g_weights)]
        random.shuffle(samples)
        return iter([val for tup in zip(*samples) for val in tup])