import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tqdm import tqdm

import constants
from models.architectures import unet2d_8
from utils.general_utils import folder_picker
from utils.image_utils import normalize_image

pool_folder = folder_picker(initialdir=constants.DATA_DIR)

import platform
if 'macOS' in platform.platform():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

def priority_metric(y_pred):
    p = np.sqrt(np.sum(y_pred[1:, :, :] - y_pred[:-1, :, :]) ** 2 + np.sum(
        y_pred[:, 1:, :] - y_pred[:, :-1, :]) ** 2) // np.prod(y_pred.shape)
    return p


def data_generator(pool_folder, batch_size=1):
    # image_paths = [pathlib.Path(pool_folder, file) for file in os.listdir(pool_folder) if 'image_' in file]
    df = pd.read_csv(constants.PRIORITY_DF)
    image_paths = list(df['paths'])
    sample_image = np.load(image_paths[0])
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((batch_size, n_rows, n_cols, n_input_channels))
    y = np.zeros((batch_size, n_rows, n_cols, constants.n_classes))
    while True:
        df = pd.read_csv(constants.PRIORITY_DF)
        image_paths = list(df['paths'])
        priorities = list(df['p'])
        priorities = np.array(priorities) / np.sum(priorities)
        for i in range(batch_size):
            found_non_empty_scribble = False
            while not found_non_empty_scribble:
                image_path = np.random.choice(image_paths, p=priorities)
                image_path = pathlib.Path(image_path)
                scribble_path = pathlib.Path(image_path.parent, image_path.name.replace('image_', 'scribble_'))
                scribble = np.load(scribble_path)
                if np.any(scribble):
                    img = np.load(image_path)
                    x[i] = normalize_image(img)
                    y[i] = scribble.astype(int)
                    found_non_empty_scribble = True
        yield x, y


def weighted_cce(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -K.sum(tf.reduce_sum(y_true * K.log(y_pred), axis=-1, keepdims=True) * weights, -1)
    return loss


training_generator = data_generator(pool_folder, batch_size=1)
x, y = next(training_generator)
n_input_channels = x.shape[-1]
model = unet2d_8(n_input_channels)
model.compile(loss=[weighted_cce], optimizer=optimizers.Adam(1e-3))

image_paths = [pathlib.Path(pool_folder, file) for file in os.listdir(pool_folder) if 'image_' in file]
pred_paths = [pathlib.Path(image_path.parent, image_path.name.replace('image_', 'pred_')) for image_path in
              image_paths]

while True:
    model.fit(training_generator, steps_per_epoch=100, epochs=10)
    df = []
    for image_path, pred_path in tqdm(zip(image_paths, pred_paths)):
        image = np.load(image_path)
        pred = model.predict(image[None, ...])[0]
        p = priority_metric(pred)
        df.append([image_path, p])
        df = pd.DataFrame.from_records(df, columns=['paths', 'p'])
        df.to_csv(constants.PRIORITY_DF)
        np.save(arr=pred, file=pred_path)
