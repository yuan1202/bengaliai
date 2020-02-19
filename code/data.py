# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import os, itertools, random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

from config import *


class Bengaliai_DS_AL(Dataset):
    """ Advanced Loss (AL)
    """

    def __init__(self, img_arr, lbl_arr, double_classes=None, transform=None, scale=True, norm=True, split_label=False, dtype='float32'):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.double_classes     = double_classes
        self.transform          = transform
        self.scale              = scale
        self.norm               = norm
        self.split_label        = split_label
        self.dtype              = dtype

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        
        img = self.img_arr[index]
        lbl = self.labels[index].astype(int)
        
        if isinstance(self.double_classes, int) & (random.randint(0, 1)):
            img = img[:, ::-1]
            lbl += self.double_classes
        
        if self.transform:
            
            k = np.ones((random.randint(2, 3), random.randint(2, 3)), np.uint8)
            if random.randint(0, 1):
                img = cv2.dilate(img, k, iterations=1)
            else:
                img = cv2.erode(img, k, iterations=1)
                
            img = np.array(self.transform(image=img))
            
        img = img[np.newaxis, :].astype(float)
        
        if self.scale:
            img = img / 255.
        
        if self.norm:
            img = (img - 0.0692) / 0.2051
            #img = (img - img.mean()) / img.std()
            
        if self.split_label:
            lbl = lbl.tolist()
            
        return (img.astype(self.dtype), lbl), lbl

    def __len__(self):
        """Returns the number of data points."""
        return self.img_arr.shape[0]
    
    @property
    def is_empty(self):
        return self.__len__() == 0


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
        
        img = self.img_arr[index]
        
        if self.transform:
            
            k = np.ones((random.randint(2, 3), random.randint(2, 3)), np.uint8)
            if random.randint(0, 1):
                img = cv2.dilate(img, k, iterations=1)
            else:
                img = cv2.erode(img, k, iterations=1)
                
            img = self.transform(image=img)
            
        img = np.repeat(img[np.newaxis, :, :], 3, axis=0).astype(float)
        
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
    """ Load in time (LIT)
    """

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
    
    def __init__(self, pdf, size, block_size, group_draw_weights=[1, 1, 1, 1]):
        
        # grapheme_root, vowel_diacritic, consonant_diacritic
        self.gb0 = pdf.groupby(['grapheme_root', 'vowel_diacritic'])
        self.gb0_draw_ndces = np.arange(len(self.gb0)).tolist()
        self.gb1 = pdf.groupby(['vowel_diacritic', 'consonant_diacritic'])
        self.gb1_draw_ndces = np.arange(len(self.gb1)).tolist()
        self.gb2 = pdf.groupby(['consonant_diacritic', 'grapheme_root'])
        self.gb2_draw_ndces = np.arange(len(self.gb2)).tolist()
        
        self.size = size
        self.block_size = block_size
        self.gb_draw_weights = group_draw_weights
        
        self.gb0_subg_indices = []
        self.gb0_subg_weights = []
        self.gb1_subg_indices = []
        self.gb1_subg_weights = []
        self.gb2_subg_indices = []
        self.gb2_subg_weights = []
        
        self.gb_len = [0, 0, 0, 1]
        
        indices_super = []
        weights_super = []
        
        # equal opportunity at this level
        for g in self.gb0:
            indices = []
            weights = []
            sub_gb = g[1].groupby('consonant_diacritic')
            
            if len(sub_gb) == 1:
                for sub_g in sub_gb:
                    indices_super.append(sub_g[1].index.tolist())
                    weights_super.append(sub_g[1].shape[0])
            else:
                # weighted drawing between sub-groups
                for sub_g in sub_gb:
                    indices.append(sub_g[1].index.tolist())
                    weights.append(sub_g[1].shape[0])

                # post process weights for this group
                weights = np.array(weights)
                weights = weights.sum() / weights

                self.gb0_subg_weights.append(
                    list(
                        itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
                    )
                )
                self.gb0_subg_indices.append(list(itertools.chain(*indices)))
            
                self.gb_len[0] += 1
            
        for g in self.gb1:
            indices = []
            weights = []
            sub_gb = g[1].groupby('grapheme_root')
            
            if len(sub_gb) == 1:
                for sub_g in sub_gb:
                    indices_super.append(sub_g[1].index.tolist())
                    weights_super.append(sub_g[1].shape[0])
            else:
                # weighted drawing between sub-groups
                for sub_g in sub_gb:
                    indices.append(sub_g[1].index.tolist())
                    weights.append(sub_g[1].shape[0])

                # post process weights for this group
                weights = np.array(weights)
                weights = weights.sum() / weights

                self.gb1_subg_weights.append(
                    list(
                        itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
                    )
                )
                self.gb1_subg_indices.append(list(itertools.chain(*indices)))
                
                self.gb_len[1] += 1
            
        self.gb_len.append(len(self.gb1))
        
        for g in self.gb2:
            indices = []
            weights = []
            sub_gb = g[1].groupby('vowel_diacritic')
            
            if len(sub_gb) == 1:
                for sub_g in sub_gb:
                    indices_super.append(sub_g[1].index.tolist())
                    weights_super.append(sub_g[1].shape[0])
            else:
                # weighted drawing between sub-groups
                for sub_g in sub_gb:
                    indices.append(sub_g[1].index.tolist())
                    weights.append(sub_g[1].shape[0])

                # post process weights for this group
                weights = np.array(weights)
                weights = weights.sum() / weights

                self.gb2_subg_weights.append(
                    list(
                        itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
                    )
                )
                self.gb2_subg_indices.append(list(itertools.chain(*indices)))
                
                self.gb_len[2] += 1
            
        self.gb_len.append(len(self.gb2))
        
        weights_super = np.array(weights_super)
        weights_super = weights_super.sum() / weights_super
        self.gbsuper_subg_weights = [list(
            itertools.chain(*[[w] * len(i) for w, i in zip(weights_super.tolist(), indices_super)])
        )]
        self.gbsuper_subg_indices = [list(itertools.chain(*indices_super))]
        self.gb_len.append(1)
        
        self.gb_draw_ndcs = [0, 1, 2, 3]
            
        self.gb_sampling_pw = (
            (self.gb0_subg_indices, self.gb0_subg_weights),
            (self.gb1_subg_indices, self.gb1_subg_weights),
            (self.gb2_subg_indices, self.gb2_subg_weights),
            (self.gbsuper_subg_indices, self.gbsuper_subg_weights)
        )
        
        
            
    def __len__(self):
        return self.size
    
    def __iter__(self):
        
        n_iters = self.size // self.block_size + (1 if self.size % self.block_size > 0 else 0)
        gb_draws = random.choices(population=[0,1,2,3], weights=self.gb_draw_weights, k=n_iters)
        
        samples = []
        for g in gb_draws:
            gb_pick = random.randint(0, self.gb_len[g]-1)
            samples += random.choices(
                population=self.gb_sampling_pw[g][0][gb_pick],
                weights=self.gb_sampling_pw[g][1][gb_pick],
                k=self.block_size
            )
            
        samples = samples[:self.size]
        return iter(samples)