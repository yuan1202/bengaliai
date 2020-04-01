# based on https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
# and https://www.kaggle.com/iafoss/image-preprocessing-128x128/output

import os, itertools, random
import numpy as np
from scipy import ndimage
from PIL import Image
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

from config import *
from heng.augmentation import *


class Bengaliai_DS(Dataset):

    def __init__(self, img_arr, lbl_arr, transform=None, split_label=False, RGB=False, raw=False, dtype='float32'):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.split_label        = split_label
        self.RGB                = RGB
        self.raw                = raw
        self.dtype              = dtype
        self.k                  = np.ones((2, 2), np.uint8)

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        
        img = self.img_arr[index]
        
        # some boundry cleanup
        if self.raw:
            img[:3, :] = 0
            img[-3:, :] = 0
            img[:, :3] = 0
            img[:, -3:] = 0
        
        if self.transform:
            img = self.transform(image=img)
            op = random.randint(0, 2)
            if op == 0:
                img = self.do_random_line(img)
            elif op == 1:
                if random.randint(0, 1):
                    img = cv2.dilate(img, self.k, iterations=1)
                else:
                    img = cv2.erode(img, self.k, iterations=1)
                     
        #    op = random.randint(0, 3)
        #    if (op == 0) or (op == 1):
        #        # cutout or perspective
        #        img = self.transform(image=img)
        #    elif op == 2:
        #        # line
        #        img = self.do_random_line(img)
        #    
        #    op = random.randint(0, 2)
        #    if op == 0:
        #        img = cv2.dilate(img, self.k, iterations=1)
        #    elif op ==1:
        #        img = cv2.erode(img, self.k, iterations=1)
                
        # scale
        img = img / 255.
        # norm
        img = (img - 0.055373063765223995) / 0.17266245915644673
        # make channel
        img = img[None, :, :]
        if self.RGB:
            img = np.repeat(img, 3, axis=0)
            
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
    
    def do_random_line(self, image):
        magnitude=0.5

        num_lines = int(round(1 + np.random.randint(8)*magnitude))

        # ---
        height,width = image.shape
        image = image.copy()

        def line0():
            return (0,0),(width-1,0)

        def line1():
            return (0,height-1),(width-1,height-1)

        def line2():
            return (0,0),(0,height-1)

        def line3():
            return (width-1,0),(width-1,height-1)

        def line4():
            x0,x1 = np.random.choice(width,2)
            return (x0,0),(x1,height-1)

        def line5():
            y0,y1 = np.random.choice(height,2)
            return (0,y0),(width-1,y1)

        for i in range(num_lines):
            p = np.array([1/4,1/4,1/4,1/4,1,1])
            func = np.random.choice([line0,line1,line2,line3,line4,line5],p=p/p.sum())
            (x0,y0),(x1,y1) = func()

            color     = np.random.uniform(0,1)
            thickness = np.random.randint(1,5)
            line_type = np.random.choice([cv2.LINE_AA,cv2.LINE_4,cv2.LINE_8])

            cv2.line(image,(x0,y0),(x1,y1), color, thickness, line_type)

        return image

    def do_random_custom_distortion1(self, image):
        magnitude=0.5

        distort=magnitude*0.3

        height,width = image.shape
        s_x = np.array([0.0, 0.5, 1.0,  0.0, 0.5, 1.0,  0.0, 0.5, 1.0])
        s_y = np.array([0.0, 0.0, 0.0,  0.5, 0.5, 0.5,  1.0, 1.0, 1.0])
        d_x = s_x.copy()
        d_y = s_y.copy()
        d_x[[1,4,7]] += np.random.uniform(-distort,distort, 3)
        d_y[[3,4,5]] += np.random.uniform(-distort,distort, 3)

        s_x = (s_x*width )
        s_y = (s_y*height)
        d_x = (d_x*width )
        d_y = (d_y*height)

        #---
        distort = np.zeros((height,width),np.float32)
        for index in ([4,1,3],[4,1,5],[4,7,3],[4,7,5]):
            point = np.stack([s_x[index],s_y[index]]).T
            qoint = np.stack([d_x[index],d_y[index]]).T

            src  = np.array(point, np.float32)
            dst  = np.array(qoint, np.float32)
            mat  = cv2.getAffineTransform(src, dst)

            point = np.round(point).astype(np.int32)
            x0 = np.min(point[:,0])
            x1 = np.max(point[:,0])
            y0 = np.min(point[:,1])
            y1 = np.max(point[:,1])
            mask = np.zeros((height,width),np.float32)
            mask[y0:y1,x0:x1] = 1

            mask = mask*image
            warp = cv2.warpAffine(mask, mat, (width, height),borderMode=cv2.BORDER_REPLICATE)
            distort = np.maximum(distort,warp)
            #distort = distort+warp

        return distort

    def do_random_grid_distortion(self, image):
        magnitude=0.4

        num_step = 5
        distort  = magnitude

        # http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
        distort_x = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]
        distort_y = [1 + random.uniform(-distort,distort) for i in range(num_step + 1)]

        #---
        height, width = image.shape[:2]
        xx = np.zeros(width, np.float32)
        step_x = width // num_step

        prev = 0
        for i, x in enumerate(range(0, width, step_x)):
            start = x
            end   = x + step_x
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + step_x * distort_x[i]

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        yy = np.zeros(height, np.float32)
        step_y = height // num_step
        prev = 0
        for idx, y in enumerate(range(0, height, step_y)):
            start = y
            end = y + step_y
            if end > height:
                end = height
                cur = height
            else:
                cur = prev + step_y * distort_y[idx]

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image


# class Bengaliai_DS(Dataset):

#     def __init__(self, img_arr, lbl_arr, transform=None, norm=True, split_label=False, RGB=True, raw=False, dtype='float32'):
#         self.img_arr            = img_arr
#         self.labels             = lbl_arr.astype(int)
#         self.transform          = transform
#         self.norm               = norm
#         self.split_label        = split_label
#         self.RGB                = RGB
#         self.raw                = raw
#         self.dtype              = dtype
#         self.k                  = np.ones((2, 2), np.uint8)

#     def __getitem__(self, index):
#         """Returns an example or a sequence of examples from each population."""
        
#         img = self.img_arr[index]
        
#         # some boundry cleanup
#         if self.raw:
#             img[:3, :] = 0
#             img[-3:, :] = 0
#             img[:, :3] = 0
#             img[:, -3:] = 0
        
#         if self.transform:
            
#             morph = random.randint(0, 2)
#             msk = None
#             # quater chance dilate
#             #if morph == 1:
#             #    img = cv2.dilate(img, self.k, iterations=1)
#             # quater chance erode
#             #elif morph == 2:
#             #    img = cv2.erode(img, self.k, iterations=1)
#             # 1/3 chance grid mask
#             if morph == 1:
#                 msk = torch.from_numpy(self.grid_mask(img))
                
#             img = self.transform(img)
#             if msk is not None:
#                 img[0, msk == 1] = 0.
#         else:
#             img = img / 255.
#             img = torch.from_numpy(img[np.newaxis, :, :].astype(np.float32))
        
#         if self.RGB:
#             img = img.repeat(3, 1, 1)
        
#         if self.norm:
#             img = (img - 0.09790837519093555) / 0.23233699741280162
            
#         lbl = self.labels[index].astype(int)
#         if self.split_label:
#             lbl = lbl.tolist()
            
#         return img, lbl

#     def __len__(self):
#         """Returns the number of data points."""
#         return self.img_arr.shape[0]
    
#     @property
#     def is_empty(self):
#         return self.__len__() == 0
    
#     def grid_mask(self, image):
#         assert len(image.shape) == 2
#         h, w = image.shape

#         mask = np.zeros((h*2, w*2))

#         density_h = random.randint(h//8, h//4)
#         density_w = random.randint(w//8, w//4)
#         factor_h = random.randint(2, 3)
#         factor_w = random.randint(2, 3)

#         mask[::density_h, ::density_w] = 1

#         mask = cv2.dilate(mask, np.ones((density_h//factor_h, density_w//factor_w), np.uint8), iterations=1)

#         mask = np.round(ndimage.rotate(mask, random.randint(-10, 10), reshape=False))

#         mask = mask[h//2:h+h//2, w//2:w+w//2]

#         return np.round(mask)


# class Bengaliai_DS_AL(Dataset):
#     """ Advanced Loss (AL)
#     """

#     def __init__(self, img_arr, lbl_arr, double_classes=None, transform=None, scale=True, norm=True, split_label=False, dtype='float32'):
#         self.img_arr            = img_arr
#         self.labels             = lbl_arr.astype(int)
#         self.double_classes     = double_classes
#         self.transform          = transform
#         self.scale              = scale
#         self.norm               = norm
#         self.split_label        = split_label
#         self.dtype              = dtype

#     def __getitem__(self, index):
#         """Returns an example or a sequence of examples from each population."""
        
#         img = self.img_arr[index]
#         lbl = self.labels[index].astype(int)
        
#         if isinstance(self.double_classes, int) & (random.randint(0, 1)):
#             img = img[:, ::-1]
#             lbl += self.double_classes
        
#         if self.transform:
            
#             k = np.ones((random.randint(2, 3), random.randint(2, 3)), np.uint8)
#             if random.randint(0, 1):
#                 img = cv2.dilate(img, k, iterations=1)
#             else:
#                 img = cv2.erode(img, k, iterations=1)
                
#             img = np.array(self.transform(image=img))
            
#         img = img[np.newaxis, :].astype(float)
        
#         if self.scale:
#             img = img / 255.
        
#         if self.norm:
#             img = (img - 0.0692) / 0.2051
#             #img = (img - img.mean()) / img.std()
            
#         if self.split_label:
#             lbl = lbl.tolist()
            
#         return (img.astype(self.dtype), lbl), lbl

#     def __len__(self):
#         """Returns the number of data points."""
#         return self.img_arr.shape[0]
    
#     @property
#     def is_empty(self):
#         return self.__len__() == 0
    

# class Bengaliai_DS_LIT(Dataset):
#     """ Load in time (LIT)
#     """

#     def __init__(self, pdf, img_dir='../features/grapheme-imgs-128x128/', transform=None, scale=True, norm=True, split_label=False):
#         self.dir                = img_dir
#         assert all([col in pdf for col in ['image_id', 'grapheme_root',	'vowel_diacritic', 'consonant_diacritic', 'grapheme']])
#         self.pdf                = pdf
#         self.transform          = transform
#         self.scale              = scale
#         self.norm               = norm
#         self.split_label        = split_label

#     def __getitem__(self, index):
#         """Returns an example or a sequence of examples from each population."""
#         img = Image.open(os.path.join('../features/grapheme-imgs-128x128/', self.pdf.iloc[index, 0] + '.png')).convert('L')
#         img = np.array(img)
        
#         if self.transform:
#             img = self.transform(image=img)[np.newaxis, :]
#         else:
#             img = img[np.newaxis, :]
            
#         img = img.astype(float)
        
#         if self.scale:
#             img = img / 255.
        
#         if self.norm:
#             img = (img - 0.0692) / 0.2051
#             #img = (img - img.mean()) / img.std()
            
#         lbl = self.pdf.iloc[index, 1:4].values.astype(int)
#         if self.split_label:
#             lbl = lbl.tolist()
            
#         return img.astype('float32'), lbl

#     def __len__(self):
#         """Returns the number of data points."""
#         return self.pdf.shape[0]
    
#     @property
#     def is_empty(self):
#         return self.__len__() == 0
    
    
# # -----------------------------------------------------------------
# class Balanced_Sampler(Sampler):
    
#     def __init__(self, pdf, count_column, primary_group, secondary_group, size):
        
#         self.gb = pdf.groupby(primary_group)
        
#         self.size = size
        
#         self._g_indices = []
#         self._g_weights = []
        
#         # equal opportunity at this level
#         for g in self.gb:
#             indices = []
#             weights = []
#             sub_gb = g[1].groupby(secondary_group)
#             # weighted drawing between sub-groups
#             for sub_g in sub_gb:
#                 indices.append(sub_g[1].index.tolist())
#                 weights.append(sub_g[1].shape[0])
                
#             # post process weights for this group
#             weights = np.array(weights)
#             weights = weights.sum() / weights
            
#             self._g_weights.append(
#                 list(
#                     itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
#                 )
#             )
#             self._g_indices.append(list(itertools.chain(*indices)))
            
#     def __len__(self):
#         return self.size
    
#     def __iter__(self):
#         samples_per_group = np.round(self.size / len(self.gb)).astype(int)
#         samples = [random.choices(population=p, weights=w, k=samples_per_group) for p, w in zip(self._g_indices, self._g_weights)]
#         random.shuffle(samples)
#         return iter([val for tup in zip(*samples) for val in tup])
    

# class Balanced_Sampler_v2(Sampler):
    
#     def __init__(self, pdf, size, block_size, group_draw_weights=[1, 1, 1, 1]):
        
#         # grapheme_root, vowel_diacritic, consonant_diacritic
#         self.gb0 = pdf.groupby(['grapheme_root', 'vowel_diacritic'])
#         self.gb0_draw_ndces = np.arange(len(self.gb0)).tolist()
#         self.gb1 = pdf.groupby(['vowel_diacritic', 'consonant_diacritic'])
#         self.gb1_draw_ndces = np.arange(len(self.gb1)).tolist()
#         self.gb2 = pdf.groupby(['consonant_diacritic', 'grapheme_root'])
#         self.gb2_draw_ndces = np.arange(len(self.gb2)).tolist()
        
#         self.size = size
#         self.block_size = block_size
#         self.gb_draw_weights = group_draw_weights
        
#         self.gb0_subg_indices = []
#         self.gb0_subg_weights = []
#         self.gb1_subg_indices = []
#         self.gb1_subg_weights = []
#         self.gb2_subg_indices = []
#         self.gb2_subg_weights = []
        
#         self.gb_len = [0, 0, 0, 1]
        
#         indices_super = []
#         weights_super = []
        
#         # equal opportunity at this level
#         for g in self.gb0:
#             indices = []
#             weights = []
#             sub_gb = g[1].groupby('consonant_diacritic')
            
#             if len(sub_gb) == 1:
#                 for sub_g in sub_gb:
#                     indices_super.append(sub_g[1].index.tolist())
#                     weights_super.append(sub_g[1].shape[0])
#             else:
#                 # weighted drawing between sub-groups
#                 for sub_g in sub_gb:
#                     indices.append(sub_g[1].index.tolist())
#                     weights.append(sub_g[1].shape[0])

#                 # post process weights for this group
#                 weights = np.array(weights)
#                 weights = weights.sum() / weights

#                 self.gb0_subg_weights.append(
#                     list(
#                         itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
#                     )
#                 )
#                 self.gb0_subg_indices.append(list(itertools.chain(*indices)))
            
#                 self.gb_len[0] += 1
            
#         for g in self.gb1:
#             indices = []
#             weights = []
#             sub_gb = g[1].groupby('grapheme_root')
            
#             if len(sub_gb) == 1:
#                 for sub_g in sub_gb:
#                     indices_super.append(sub_g[1].index.tolist())
#                     weights_super.append(sub_g[1].shape[0])
#             else:
#                 # weighted drawing between sub-groups
#                 for sub_g in sub_gb:
#                     indices.append(sub_g[1].index.tolist())
#                     weights.append(sub_g[1].shape[0])

#                 # post process weights for this group
#                 weights = np.array(weights)
#                 weights = weights.sum() / weights

#                 self.gb1_subg_weights.append(
#                     list(
#                         itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
#                     )
#                 )
#                 self.gb1_subg_indices.append(list(itertools.chain(*indices)))
                
#                 self.gb_len[1] += 1
            
#         self.gb_len.append(len(self.gb1))
        
#         for g in self.gb2:
#             indices = []
#             weights = []
#             sub_gb = g[1].groupby('vowel_diacritic')
            
#             if len(sub_gb) == 1:
#                 for sub_g in sub_gb:
#                     indices_super.append(sub_g[1].index.tolist())
#                     weights_super.append(sub_g[1].shape[0])
#             else:
#                 # weighted drawing between sub-groups
#                 for sub_g in sub_gb:
#                     indices.append(sub_g[1].index.tolist())
#                     weights.append(sub_g[1].shape[0])

#                 # post process weights for this group
#                 weights = np.array(weights)
#                 weights = weights.sum() / weights

#                 self.gb2_subg_weights.append(
#                     list(
#                         itertools.chain(*[[w] * len(i) for w, i in zip(weights.tolist(), indices)])
#                     )
#                 )
#                 self.gb2_subg_indices.append(list(itertools.chain(*indices)))
                
#                 self.gb_len[2] += 1
            
#         self.gb_len.append(len(self.gb2))
        
#         weights_super = np.array(weights_super)
#         weights_super = weights_super.sum() / weights_super
#         self.gbsuper_subg_weights = [list(
#             itertools.chain(*[[w] * len(i) for w, i in zip(weights_super.tolist(), indices_super)])
#         )]
#         self.gbsuper_subg_indices = [list(itertools.chain(*indices_super))]
#         self.gb_len.append(1)
        
#         self.gb_draw_ndcs = [0, 1, 2, 3]
            
#         self.gb_sampling_pw = (
#             (self.gb0_subg_indices, self.gb0_subg_weights),
#             (self.gb1_subg_indices, self.gb1_subg_weights),
#             (self.gb2_subg_indices, self.gb2_subg_weights),
#             (self.gbsuper_subg_indices, self.gbsuper_subg_weights)
#         )
        
        
            
#     def __len__(self):
#         return self.size
    
#     def __iter__(self):
        
#         n_iters = self.size // self.block_size + (1 if self.size % self.block_size > 0 else 0)
#         gb_draws = random.choices(population=[0,1,2,3], weights=self.gb_draw_weights, k=n_iters)
        
#         samples = []
#         for g in gb_draws:
#             gb_pick = random.randint(0, self.gb_len[g]-1)
#             samples += random.choices(
#                 population=self.gb_sampling_pw[g][0][gb_pick],
#                 weights=self.gb_sampling_pw[g][1][gb_pick],
#                 k=self.block_size
#             )
            
#         samples = samples[:self.size]
#         return iter(samples)