import os, gc, sys
from datetime import datetime
import numpy as np
import pandas as pd
import cv2

from config import *


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, pad=PADDING, keep_aspect=KEEP_ASPECT):
    #crop a box around pixels large than the threshold 
    #some images contain line at the sides
    img0[:3, :] = 0
    img0[-3:, :] = 0
    img0[:, :3] = 0
    img0[:, -3:] = 0
    ymin, ymax, xmin, xmax = bbox(img0 > 50)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 5 if (xmin > 5) else 0
    ymin = ymin - 5 if (ymin > 5) else 0
    xmax = xmax + 5 if (xmax < WIDTH - 5) else WIDTH
    ymax = ymax + 5 if (ymax < HEIGHT - 5) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    #img[img < THRESHOLD] = 0
    if keep_aspect:
        lx, ly = xmax-xmin, ymax-ymin
        l = max(lx,ly) + pad
        #make sure that the aspect ratio is kept in rescaling
        img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    else:
        img = np.pad(img, [(pad, pad)])
    return cv2.resize(img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA )


def to_64x112(image):
    image = cv2.resize(image, dsize=None, fx=64/137,fy=64/137, interpolation=cv2.INTER_AREA )
    image = cv2.copyMakeBorder(image,0,0,1,1,cv2.BORDER_CONSTANT,0)
    return image


def to_128x128(image):
    image = cv2.resize(image, dsize=(RESIZE, RESIZE), interpolation=cv2.INTER_AREA )
    return image


def to_98x168(image):
    image = cv2.resize(image, dsize=(168, 98), interpolation=cv2.INTER_AREA )
    return image


def to_128x220(image):
    image = cv2.resize(image, dsize=(220, 128), interpolation=cv2.INTER_AREA )
    return image


def to_224x224(image):
    image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA )
    return image


def prepare_image(datadir, data_type='train', submission=False, indices=[0, 1, 2, 3], preprocess=True):
    assert data_type in ['train', 'test']
    
    image_df_list = [pd.read_parquet(os.path.join(datadir, '{}_image_data_{}.parquet'.format(data_type, i))) for i in indices]

    print('image_df_list', len(image_df_list))
    
    # restore image shape and change to numpy array
    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    label_trace = [df.iloc[:, 0:1].values for df in image_df_list]
    
    # clean up
    del image_df_list
    gc.collect()
    
    # correct value, enhance and crop/center image
    images = 255 - np.concatenate(images, axis=0)
    
    images_enhance = []
    for img in images:
        if preprocess:
            #images_enhance.append(crop_resize(np.round(img * (255. / img.max())).astype(np.uint8)))
            images_enhance.append(crop_resize(img.astype(np.uint8)))
        else:
            images_enhance.append(to_64x112(img.astype(np.uint8)))
        
    return np.stack(images_enhance), np.concatenate(label_trace, axis=0)


class csv_logger():
    def __init__(self, items):
        assert isinstance(items, (list, tuple)), 'Please initiliase logger with a list of columns (items to log).'
        self.log = pd.DataFrame(columns=items)
        self.log.index.name = 'Epoch'
        self.__pointer = -1
        
    def reset(self, items):
        self.log = pd.DataFrame(columns=items)
        
    def new_epoch(self):
        self.__pointer += 1
    
    def enter(self, entry):
        assert self.__pointer >= 0, 'Please call new_epoch() after initialisation.'
        assert isinstance(entry, dict), 'Input information has to be in the form of python dictionary.'
        for kv in entry.items():
            if kv[0] not in self.log:
                print('{} is not setup to be tracked. Please reset logger with correct items.'.format(kv[0]))
                continue
            else:
                self.log.loc[self.__pointer, kv[0]] = kv[1]
            
    def save(self, path):
        assert isinstance(path, str), 'Please specify a path.'
        if not path.endswith('.csv'):
            path = path + '.csv'
        self.log.to_csv(path)
        
        
def display_progress(total, current, key_value_pairs={}, postfix='batches', persist=False):
    time = '; time: {};'.format(datetime.now().strftime("%d%b%Y-%HH%MM%SS"))
    message = '{:d}/{:d} {}: '.format(current, total, postfix) + '; '.join([k + ': {:.4f}'.format(v) for k, v in key_value_pairs.items()]) + time
    if persist:
        print('\n')
        print(message)
    else:
        sys.stdout.write('\r')
        sys.stdout.write(message)
        sys.stdout.flush()
        
        
def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn