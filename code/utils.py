import os, gc, sys
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


def crop_resize(img0, pad=PADDING):
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


def prepare_image(datadir, featherdir, data_type='train', submission=False, indices=[0, 1, 2, 3]):
    assert data_type in ['train', 'test']
    
    if submission:
        image_df_list = [pd.read_parquet(os.path.join(datadir, '{}_image_data_{}.parquet'.format(data_type, i))) for i in indices]
    else:
        image_df_list = [pd.read_feather(os.path.join(featherdir,  '{}_image_data_{}.feather'.format(data_type, i))) for i in indices]

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
        images_enhance.append(crop_resize(np.round(img * (255. / img.max())).astype(np.uint8)))
        
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
        
        
def display_progress(total, current, key_value_pairs, postfix='batches', persist=False):
    message = '{:d}/{:d} {}: '.format(current, total, postfix) + '; '.join([k + ': {:.3f}'.format(v) for k, v in key_value_pairs.items()])
    if persist:
        print('\n')
        print(message)
    else:
        sys.stdout.write('\r')
        sys.stdout.write(message)
        sys.stdout.flush()