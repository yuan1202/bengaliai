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

import imgaug.augmenters as iaa
from heng.augmentation import *


# raw size             mean 0.055373063765223995, std 0.17266245915644673
# 128Pad8MaxNoclean    mean 0.10809827140865982   std 0.23662995113750404
# 98168                mean 0.053062911426318776  std 0.16613851882368738
# 180x180              mean 0.08596207363893509   std 0.21542730913818242
# 96x96                mean 0.11604626310025852   std 0.24329951808465136
# half                 mean 0.05213458652450004   std 0.16489511099093002
# 168                  mean 0.10095022765352311   std 0.23023208883090254
# 128x220              mean 0.10095022765352311   std 0.16278256833350963
# 224x224              mean 0.0528851918275502    std 0.16336041874511376


class Bengaliai_DS(Dataset):

    def __init__(self, img_arr, lbl_arr, transform=None, split_label=False, RGB=False, dtype='float32'):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.split_label        = split_label
        self.RGB                = RGB
        self.dtype              = dtype
        self.k                  = np.ones((2, 2), np.uint8)

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        
        img = self.img_arr[index]
        
        if 0:
            img[:2, :] = 0
            img[-2:, :] = 0
            img[:, :2] = 0
            img[:, -2:] = 0
        
        # some boundry cleanup
        
        if self.transform:
            img = self.transform(image=img)
            
            #if random.randint(0, 1):
            #    if random.randint(0, 1):
            #        img = cv2.dilate(img, self.k, iterations=1)
            #    else:
            #        img = cv2.erode(img, self.k, iterations=1)
            
            opt = random.randint(0, 1)
            if opt == 0:
                img = self.do_random_line(img)
            #elif opt == 1:
            #    msk = self.grid_mask(img)
            #    img[msk==1] = 0
                
        # scale
        img = img / 255.
        # norm
        img = (img - 0.05213458652450004) / 0.16489511099093002
        
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

        num_lines = int(round(1 + np.random.randint(8) * magnitude))

        # ---
        height, width = image.shape
        image = image.copy()

        def line0():
            return (0, 0),(width-1, 0)

        def line1():
            return (0, height-1),(width-1, height-1)

        def line2():
            return (0, 0),(0, height-1)

        def line3():
            return (width-1, 0), (width-1, height-1)

        def line4():
            x0, x1 = np.random.choice(width, 2)
            return (x0, 0), (x1, height-1)

        def line5():
            y0, y1 = np.random.choice(height, 2)
            return (0, y0), (width-1, y1)

        for i in range(num_lines):
            p = np.array([1/4, 1/4, 1/4, 1/4, 1, 1])
            func = np.random.choice([line0, line1, line2, line3, line4, line5], p=p/p.sum())
            (x0, y0), (x1, y1) = func()

            color     = np.random.randint(0, 128)
            thickness = np.random.randint(1, 2)
            line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])

            cv2.line(image, (x0, y0), (x1, y1), color, thickness, line_type)

        return image
    
    def grid_mask(self, image):
        assert len(image.shape) == 2
        h, w = image.shape

        mask = np.zeros((h*2, w*2))

        density_h = random.randint(h//8, h//4)
        density_w = random.randint(w//8, w//4)
        factor_h = random.randint(2, 3)
        factor_w = random.randint(2, 3)

        mask[::density_h, ::density_w] = 1

        mask = cv2.dilate(mask, np.ones((density_h//factor_h, density_w//factor_w), np.uint8), iterations=1)
        
        random_h = random.randint(0, h//2)
        random_w = random.randint(0, w//2)

        mask = mask[random_h:h+random_h, random_w:w+random_w]

        return np.round(mask).astype('uint8')


class Bengaliai_DS_heng1(Dataset):

    def __init__(self, img_arr, lbl_arr, transform=False, split_label=False, RGB=False, raw=False, dtype='float32'):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.split_label        = split_label
        self.RGB                = RGB
        self.raw                = raw
        self.dtype              = dtype
        self.k                  = np.ones((2, 2), np.uint8)
        
        self.perspective        = iaa.PerspectiveTransform(scale=.1, keep_size=True)
        self.scale              = iaa.CropAndPad(percent=(-.15, .15), pad_mode='constant', pad_cval=0, sample_independently=False)
        self.rotate             = iaa.Affine(rotate=(-20, 20))
        self.shearx             = iaa.Affine(shear={"x": (-20, 20)})
        self.sheary             = iaa.Affine(shear={"y": (-15, 15)})
        self.stretchx           = iaa.Affine(scale={"x": (0.80, 1.20)})
        self.stretchy           = iaa.Affine(scale={"y": (0.80, 1.20)})
        self.piecewise          = iaa.PiecewiseAffine(scale=.03)
        self.edgedetect         = iaa.DirectedEdgeDetect(alpha=(.01, .99), direction=(0.0, 1.0))
        self.contrast           = iaa.SigmoidContrast(gain=5, cutoff=(0.4, 0.6))

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        
        img = self.img_arr[index]
        
        if self.transform:
            
            opt = random.randint(0, 9)
            if opt == 1: img = self.perspective(image=img)
            elif opt == 3: img = self.scale(image=img)
            elif opt == 4: img = self.rotate(image=img)
            elif opt == 5: img = self.shearx(image=img)
            elif opt == 6: img = self.sheary(image=img)
            elif opt == 7: img = self.stretchx(image=img)
            elif opt == 8: img = self.stretchy(image=img)
            elif opt == 9: img = self.piecewise(image=img)
                
            opt = random.randint(0, 5)
            if opt == 1: img = cv2.erode(img, self.k, iterations=1)
            elif opt == 2: img = cv2.dilate(img, self.k, iterations=1)
            elif opt == 3: img = self.do_random_line(image=img)
            elif opt == 4: img = self.edgedetect(image=img)
            elif opt == 5: img = self.contrast(image=img)
                
            opt = random.randint(0, 1)
            if opt == 1: img = self.do_random_block_fade(image=img)
                
        if self.raw == False:
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

        num_lines = int(round(1 + np.random.randint(8) * magnitude))

        # ---
        height, width = image.shape
        image = image.copy()

        def line0():
            return (0, 0),(width-1, 0)

        def line1():
            return (0, height-1),(width-1, height-1)

        def line2():
            return (0, 0),(0, height-1)

        def line3():
            return (width-1, 0), (width-1, height-1)

        def line4():
            x0, x1 = np.random.choice(width, 2)
            return (x0, 0), (x1, height-1)

        def line5():
            y0, y1 = np.random.choice(height, 2)
            return (0, y0), (width-1, y1)

        for i in range(num_lines):
            p = np.array([1/4, 1/4, 1/4, 1/4, 1, 1])
            func = np.random.choice([line0, line1, line2, line3, line4, line5], p=p/p.sum())
            (x0, y0), (x1, y1) = func()

            color     = np.random.randint(0, 128)
            thickness = np.random.randint(1, 5)
            line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])

            cv2.line(image, (x0, y0), (x1, y1), color, thickness, line_type)

        return image
    
    def do_random_block_fade(self, image):
        H, W = image.shape
        wip = image.copy()
        
        base_size = min(H, W)
        cut = np.round(np.sqrt(np.random.uniform(.1, .25)) * base_size).astype(np.int)
        
        x = np.random.randint(W)
        x = np.clip(x, cut // 2, W - cut // 2)
        y = np.random.randint(H)
        y = np.clip(y, cut // 2, H - cut // 2)
        
        bbx0s = x - cut // 2
        bby0s = y - cut // 2
        bbx1s = x + cut // 2
        bby1s = y + cut // 2
        
        wip[bby0s:bby1s, bbx0s:bbx1s] = np.round(image[bby0s:bby1s, bbx0s:bbx1s].astype(np.float32) * np.random.uniform(0.2, 0.7)).astype('uint8')
        
        return wip


class Bengaliai_DS_heng0(Dataset):

    def __init__(self, img_arr, lbl_arr, transform=None, split_label=False, RGB=False, target_size=False, dtype='float32'):
        self.img_arr            = img_arr
        self.labels             = lbl_arr.astype(int)
        self.transform          = transform
        self.split_label        = split_label
        self.RGB                = RGB
        self.dtype              = dtype
        self.targetSize         = target_size
        self.k                  = np.ones((2, 2), np.uint8)
        self.cNp                = iaa.CropAndPad(percent=(-0.15, 0.15), pad_mode='constant', pad_cval=0, sample_independently=False)

    def __getitem__(self, index):
        """Returns an example or a sequence of examples from each population."""
        
        img = self.img_arr[index]
        
        if self.targetSize:
            img = self.to_64x112(img)
        
        if self.transform == 'heng':
            img = img / 255.
            
            opt = random.randint(0, 10)
            if opt == 1: img = self.do_random_projective(img)
            elif opt == 2: img = self.do_random_perspective(img)
            elif opt == 3: img = self.do_random_scale(img)
            elif opt == 4: img = self.do_random_rotate(img)
            elif opt == 5: img = self.do_random_shear_x(img)
            elif opt == 6: img = self.do_random_shear_y(img)
            elif opt == 7: img = self.do_random_stretch_x(img)
            elif opt == 8: img = self.do_random_stretch_y(img)
            elif opt == 9: img = self.do_random_grid_distortion(img)
            elif opt == 10: img = self.do_random_custom_distortion1(img)
                
            opt = random.randint(0, 3)
            if opt == 1: img = self.do_random_erode(img)
            elif opt == 2: img = self.do_random_dilate(img)
            elif opt == 3: img = self.do_random_line(img)
                
            opt = random.randint(0, 2)
            if opt == 1: img = self.do_random_contast(img)
            elif opt == 2: img = self.do_random_block_fade(img)
            
            img = self.cNp(image=img)
            
        elif self.transform is not None:
            img = self.transform(image=img)
            op = random.randint(0, 2)
            if op == 0:
                img = self.do_random_line(img)
            elif op == 1:
                if random.randint(0, 1):
                    img = cv2.dilate(img, self.k, iterations=1)
                else:
                    img = cv2.erode(img, self.k, iterations=1)
            # scale
            img = img / 255.
        else:
            # scale
            img = img / 255.
            
        # norm
        img = (img - 0.05212665592479939) / 0.1578502826654819
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
    
    def to_64x112(self, image):
        image = cv2.resize(image,dsize=None, fx=64/137, fy=64/137, interpolation=cv2.INTER_AREA)
        image = cv2.copyMakeBorder(image, 0, 0, 1, 1, cv2.BORDER_CONSTANT, 0)
        return image
    
    def do_random_projective(self, image):
        magnitude=0.4

        mag = np.random.uniform(-1, 1) * 0.5 * magnitude

        height, width = image.shape[:2]
        x0, y0 = 0, 0
        x1, y1 = 1, 0
        x2, y2 = 1, 1
        x3, y3 = 0, 1

        mode = np.random.choice(['top','bottom','left','right'])
        if mode =='top':
            x0 += mag;   x1 -= mag
        if mode =='bottom':
            x3 += mag;   x2 -= mag
        if mode =='left':
            y0 += mag;   y3 -= mag
        if mode =='right':
            y1 += mag;   y2 -= mag

        s = np.array([[ 0, 0],[ 1, 0],[ 1, 1],[ 0, 1],])*[[width, height]]
        d = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3],])*[[width, height]]
        
        transform = cv2.getPerspectiveTransform(s.astype(np.float32),d.astype(np.float32))

        image = cv2.warpPerspective(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        return image
    
    def do_random_perspective(self, image):
        magnitude=0.4

        mag = np.random.uniform(-1, 1, (4, 2)) * 0.25 * magnitude

        height, width = image.shape[:2]
        s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        d = s + mag
        s *= [[width, height]]
        d *= [[width, height]]
        
        transform = cv2.getPerspectiveTransform(s.astype(np.float32),d.astype(np.float32))

        image = cv2.warpPerspective(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        return image
    
    def do_random_scale(self, image):
        magnitude=0.4

        s = 1 + np.random.uniform(-1, 1) * magnitude * 0.5

        height, width = image.shape[:2]
        
        transform = np.array(
            [
                [s, 0, 0],
                [0, s, 0],
            ], 
            np.float32
        )
        
        image = cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_shear_x(self, image):
        magnitude=0.5

        sx = np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array(
            [
                [1, sx, 0],
                [0, 1, 0],
            ],
            np.float32
        )
        
        image = cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_shear_y(self, image):
        magnitude=0.4

        sy = np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array(
            [
                [1, 0, 0],
                [sy, 1, 0],
            ],
            np.float32
        )
        
        image = cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_stretch_x(self, image):
        magnitude=0.5

        sx = 1 + np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array(
            [
                [sx, 0, 0],
                [0, 1, 0],
            ],
            np.float32
        )
        image = cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_stretch_y(self, image):
        magnitude=0.5

        sy = 1 + np.random.uniform(-1, 1) * magnitude

        height, width = image.shape[:2]
        transform = np.array(
            [
                [1, 0, 0],
                [0, sy, 0],
            ],
            np.float32
        )
        image = cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_rotate(self, image):
        magnitude=0.4

        angle = 1 + np.random.uniform(-1, 1) * 30 * magnitude

        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2

        transform = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        image = cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_line(self, image):
        magnitude=0.5

        num_lines = int(round(1 + np.random.randint(8) * magnitude))

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
            p = np.array([1/4, 1/4, 1/4, 1/4, 1, 1])
            func = np.random.choice([line0, line1, line2, line3, line4, line5], p=p/p.sum())
            (x0, y0), (x1, y1) = func()

            color     = np.random.uniform(0,1)
            thickness = np.random.randint(1,5)
            line_type = np.random.choice([cv2.LINE_AA, cv2.LINE_4, cv2.LINE_8])

            cv2.line(image, (x0, y0), (x1, y1), color, thickness, line_type)

        return image

    def do_random_custom_distortion1(self, image):
        magnitude=0.5

        distort=magnitude*0.3

        height,width = image.shape
        s_x = np.array([0.0, 0.5, 1.0,  0.0, 0.5, 1.0,  0.0, 0.5, 1.0])
        s_y = np.array([0.0, 0.0, 0.0,  0.5, 0.5, 0.5,  1.0, 1.0, 1.0])
        d_x = s_x.copy()
        d_y = s_y.copy()
        d_x[[1, 4, 7]] += np.random.uniform(-distort, distort, 3)
        d_y[[3, 4, 5]] += np.random.uniform(-distort, distort, 3)

        s_x = (s_x*width )
        s_y = (s_y*height)
        d_x = (d_x*width )
        d_y = (d_y*height)

        #---
        distort = np.zeros((height, width), np.float32)
        for index in ([4, 1, 3],[4, 1, 5],[4, 7, 3],[4, 7, 5]):
            point = np.stack([s_x[index], s_y[index]]).T
            qoint = np.stack([d_x[index], d_y[index]]).T

            src  = np.array(point, np.float32)
            dst  = np.array(qoint, np.float32)
            mat  = cv2.getAffineTransform(src, dst)

            point = np.round(point).astype(np.int32)
            x0 = np.min(point[:, 0])
            x1 = np.max(point[:, 0])
            y0 = np.min(point[:, 1])
            y1 = np.max(point[:, 1])
            mask = np.zeros((height, width), np.float32)
            mask[y0:y1, x0:x1] = 1

            mask = mask*image
            warp = cv2.warpAffine(mask, mat, (width, height),borderMode=cv2.BORDER_REPLICATE)
            distort = np.maximum(distort, warp)

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
        
        image = cv2.remap(
            image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return image
    
    def do_random_contast(self, image):
        magnitude=0.5

        alpha = 1 + random.uniform(-1, 1) * magnitude
        image = image.astype(np.float32) * alpha
        image = np.clip(image, 0, 1)
        return image

    def do_random_block_fade(self, image):
        magnitude=0.5

        size = [0.1, magnitude]

        height,width = image.shape

        #get bounding box
        m = image.copy()
        cv2.rectangle(m, (0, 0), (height, width), 1, 5)
        m = image < 0.5
        
        if m.sum()==0: return image

        m = np.where(m)
        y0, y1, x0, x1 = np.min(m[0]), np.max(m[0]), np.min(m[1]), np.max(m[1])
        w = x1 - x0
        h = y1 - y0
        
        if w * h < 10: return image

        ew, eh = np.random.uniform(*size, 2)
        ew = int(ew * w)
        eh = int(eh * h)

        ex = np.random.randint(0, w - ew) + x0
        ey = np.random.randint(0, h - eh) + y0
        
        image[ey:ey+eh, ex:ex+ew] *= np.random.uniform(0.2, 0.5)
        image = np.clip(image, 0.05, 1)
        
        return image
    
    def do_random_erode(self, image):
        magnitude=0.4

        s = int(round(1 + np.random.uniform(0, 1) * magnitude * 6))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
        image  = cv2.erode(image, kernel, iterations=1)
        return image

    def do_random_dilate(self, image):
        magnitude=0.4
        s = int(round(1 + np.random.uniform(0, 1) * magnitude * 6))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple((s, s)))
        image  = cv2.dilate(image, kernel, iterations=1)
        return image

    def do_random_sprinkle(self, image):
        magnitude=0.5

        size = 16
        num_sprinkle = int(round( 1 + np.random.randint(10) * magnitude))

        height, width = image.shape
        image = image.copy()
        image_small = cv2.resize(image, dsize=None, fx=0.25, fy=0.25)
        m   = np.where(image_small > 0.25)
        num = len(m[0])
        
        if num==0: return image

        s = size//2
        i = np.random.choice(num, num_sprinkle)
        
        for y,x in zip(m[0][i], m[1][i]):
            y = y * 4 + 2
            x = x * 4 + 2
            image[y-s:y+s, x-s:x+s] = 0
            
        return image

    def do_random_noise(self, image):
        magnitude=0.5

        height,width = image.shape
        noise = np.random.uniform(-1, 1, (height, width)) * magnitude * 0.7
        image = image + noise
        image = np.clip(image, 0, 1)
        return image
