import os
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
# from keras.utils.np_utils import to_categorical
from pydicom import dcmread

from utils.general_utils import rle2mask


def kitti_road_loader(image_path):
    img = imageio.imread(image_path)
    seg_path = image_path.replace('image_2', 'semantic')
    seg = imageio.imread(seg_path)
    seg = (seg == 7).astype(int)  # on vkitti, 7 is the the road
    # seg = to_categorical(seg, num_classes=2)
    return img, seg


def siim_acr_loader(image_path):
    dcm = dcmread(image_path)
    img = dcm.pixel_array
    img = img[..., None]
    parent = Path(image_path).parent.parent.absolute()
    csvs = [os.path.join(parent, f) for f in os.listdir(parent) if f.endswith('csv')]
    # df = pd.read_csv(csvs[0]).append(pd.read_csv(csvs[1]))
    df = pd.read_csv(csvs[0])
    df = df.rename(columns={' EncodedPixels': 'EncodedPixels'})
    encoded_pixels = str(df[df.ImageId == dcm.SOPInstanceUID].EncodedPixels.iloc[0])
    if not encoded_pixels == ' -1':
        rle = [int(p) for p in encoded_pixels.split(' ')]
        seg = rle2mask(rle, img.shape[1], img.shape[0])
    else:
        seg = np.zeros((img.shape[0], img.shape[1]))
    # seg = to_categorical(seg, num_classes=2)
    return img, seg.astype(np.uint8)


data_loaders = {'kitti_road': kitti_road_loader,
                'siim': siim_acr_loader}
