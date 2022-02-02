import os
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from pydicom import dcmread
from functools import partial

from utils.general_utils import rle2mask


def trim_array_by_q(arr, q):
    if q is not None:
        target_rows, target_cols = q * (arr.shape[0] // q), q * (arr.shape[1] // q)
        arr = arr[:target_rows, :target_cols]
    return arr


def kitti_class_loader(image_path, class_num, q_factor=None):
    img = imageio.imread(image_path)
    seg_path = image_path.replace('image_2', 'semantic')
    seg = imageio.imread(seg_path)
    seg = (seg == class_num).astype(int)  # on vkitti, 26 is the the car
    seg = to_categorical(seg, num_classes=2)
    img = trim_array_by_q(img, q_factor)
    seg = trim_array_by_q(seg, q_factor)
    return img, seg, True


def carla_class_loader(image_path, class_num, q_factor=None):
    '''
    https://www.kaggle.com/nbuhagiar/carla-semantic-segmentation/data
    ["Unlabeled",
    "Building",
    "Fence",
    "Other",
    "Pedestrian",
    "Pole",
    "Road line",
    "Road",
    "Sidewalk",
    "Vegetation",
    "Car",
    "Wall",
    "Traffic sign",]
    '''
    img = imageio.imread(image_path)
    seg_path = image_path.replace('CameraRGB', 'CameraSeg')
    seg = imageio.imread(seg_path)
    # crop the own car
    img = img[:500, :, :]
    seg = seg[:500, :, 0]
    seg = (seg == class_num).astype(int)
    seg = to_categorical(seg, num_classes=2)
    img = trim_array_by_q(img, q_factor)
    seg = trim_array_by_q(seg, q_factor)
    return img, seg, True


def siim_acr_loader(image_path, q_factor=None):
    dcm = dcmread(image_path)
    img = dcm.pixel_array
    img = img[..., None]

    parent = Path(image_path).parent.parent.absolute()
    df = pd.read_csv(os.path.join(parent, 'train-rle.csv'))
    column_names = df.columns
    for column_name in column_names:
        df.rename(columns={column_name: column_name.replace(' ', '')}, inplace=True)
    try:
        encoded_pixels = str(df[df.ImageId == dcm.SOPInstanceUID].EncodedPixels.iloc[0])[1:]
    except:
        return None, None, False
    if not encoded_pixels == '-1':
        rle = [int(p) for p in encoded_pixels.split(' ')]
        seg = rle2mask(rle, img.shape[1], img.shape[0])
    else:
        seg = np.zeros((img.shape[0], img.shape[1]))
    seg = to_categorical(seg, num_classes=2)
    img = trim_array_by_q(img, q_factor)
    seg = trim_array_by_q(seg, q_factor)
    return img, seg, True


def siim_acr_true_loader(image_path, q_factor=None):
    dcm = dcmread(image_path)
    img = dcm.pixel_array
    img = img[..., None]

    parent = Path(image_path).parent.parent.absolute()
    df = pd.read_csv(os.path.join(parent, 'train-rle.csv'))
    column_names = df.columns
    for column_name in column_names:
        df.rename(columns={column_name: column_name.replace(' ', '')}, inplace=True)
    try:
        encoded_pixels = str(df[df.ImageId == dcm.SOPInstanceUID].EncodedPixels.iloc[0])[1:]
    except:
        return None, None, False
    if not encoded_pixels == '-1':
        rle = [int(p) for p in encoded_pixels.split(' ')]
        seg = rle2mask(rle, img.shape[1], img.shape[0])
        success = True
    else:
        seg = np.zeros((img.shape[0], img.shape[1]))
        success = False

    seg = to_categorical(seg, num_classes=2)
    img = trim_array_by_q(img, q_factor)
    seg = trim_array_by_q(seg, q_factor)
    return img, seg, True


def drive_loader(image_path, q_factor=None):
    img = imageio.imread(image_path)
    img_dirname = os.path.dirname(image_path)
    img_basename = os.path.basename(image_path)
    seg_dirname = img_dirname.replace('images', '1st_manual')
    seg_basename = img_basename.split('_')[0] + '_manual1.gif'
    seg_path = os.path.join(seg_dirname, seg_basename)
    seg = imageio.imread(seg_path)
    seg = np.clip(seg, 0, 1)
    seg = to_categorical(seg, num_classes=2)
    img = trim_array_by_q(img, q_factor)
    seg = trim_array_by_q(seg, q_factor)
    return img, seg, True


data_loaders = {'kitti_road': partial(kitti_class_loader, class_num=7),
                'kitti_car': partial(kitti_class_loader, class_num=26),
                'carla_car': partial(carla_class_loader, class_num=10),
                'siim': siim_acr_loader,
                'siim_true': siim_acr_true_loader,
                'drive': drive_loader}
