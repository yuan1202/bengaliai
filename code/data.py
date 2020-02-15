# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import os, itertools, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

from config import *


class Bengaliai_DS(Dataset):

    def __init__(self, img_arr, lbl_arr, transform=None, scale=True, norm=True, split_label=False, dtype='float32'):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.scale              = scale
        self.norm               = norm
        self.split_label        = split_label
        self.dtype              = dtype

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        if self.transform:
            img = self.transform(image=self.img_arr[index])[np.newaxis, :]
        else:
            img = self.img_arr[index][np.newaxis, :]
            
        img = img.astype(float)
        
        if self.scale:
            img = img / 255.
        
        if self.norm:
            img = (img - 0.0692) / 0.2051
            #img = (img - img.mean()) / img.std()
            
        lbl = self.labels[index].astype(int)
        if self.split_label:
            lbl = lbl.tolist()
            
        return img.astype(self.dtype), lbl

    def __len__(self):
        """Returns the number of data points."""
        return self.img_arr.shape[0]
    
    @property
    def is_empty(self):
        return self.__len__() == 0


class Bengaliai_DS_LIT(Dataset):

    def __init__(self, pdf, img_dir='../features/grapheme-imgs-128x128/', transform=None, scale=True, norm=True, split_label=False):
        self.dir                = img_dir
        assert all([col in pdf for col in ['image_id', 'grapheme_root',	'vowel_diacritic', 'consonant_diacritic', 'grapheme']])
        self.pdf                = pdf
        self.transform          = transform
        self.scale              = scale
        self.norm               = norm
        self.split_label        = split_label

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        img = Image.open(os.path.join('../features/grapheme-imgs-128x128/', self.pdf.iloc[index, 0] + '.png')).convert('L')
        img = np.array(img)
        
        if self.transform:
            img = self.transform(image=img)[np.newaxis, :]
        else:
            img = img[np.newaxis, :]
            
        img = img.astype(float)
        
        if self.scale:
            img = img / 255.
        
        if self.norm:
            img = (img - 0.0692) / 0.2051
            #img = (img - img.mean()) / img.std()
            
        lbl = self.pdf.iloc[index, 1:4].values.astype(int)
        if self.split_label:
            lbl = lbl.tolist()
            
        return img.astype('float32'), lbl

    def __len__(self):
        """Returns the number of data points."""
        return self.pdf.shape[0]
    
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
    

class Balanced_Sampler_v2(Sampler):
    
    def __init__(self, pdf, size, block_size):
        
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