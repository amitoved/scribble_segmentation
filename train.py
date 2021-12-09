import os

import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import tensorflow as tf
from tqdm import tqdm
import constants
from models.architectures import unet2d
from utils.utils import get_paths

pool_folder_name = 'kitti'
n_input_channels = 3
pool_folder = os.path.join(constants.DATA_DIR, pool_folder_name)


def data_generator(pool_folder, batch_size=1):
    image_filenames = [filename for filename in os.listdir(pool_folder) if 'image' in filename]
    indices = [filename.split('image_')[1].split('.npy')[0] for filename in image_filenames]
    sample_image_path = get_paths(pool_folder, indices[0])[0]
    sample_image = np.load(sample_image_path)
    if n_input_channels == 3:
        n_rows, n_cols, _ = sample_image.shape
    else:
        n_rows, n_cols = sample_image.shape
    x = np.zeros((batch_size, n_rows, n_cols, n_input_channels))
    y = np.zeros((batch_size, n_rows, n_cols, constants.n_classes))
    while True:
        for i in range(batch_size):
            found_non_empty_scibble = False
            while found_non_empty_scibble == False:
                index = np.random.choice(indices)
                image_path, _, scribble_path = get_paths(pool_folder, index)
                scribble = np.load(scribble_path)
                if np.any(scribble):
                    img = np.load(image_path)
                    if n_input_channels == 1:
                        x[i, :, :, 0] = img
                    else:
                        x[i] = img
                    y[i] = scribble.astype(int)
                    found_non_empty_scibble = True

        target_rows = max(128, 2 ** np.ceil(np.log2(n_rows)))
        target_cols = max(128, 2 ** np.ceil(np.log2(n_cols)))
        rows_to_add = int(target_rows - n_rows)
        cols_to_add = int(target_cols - n_cols)
        x_ = np.pad(x, ((0, 0), (0, rows_to_add), (0, cols_to_add), (0, 0)), mode='constant', constant_values=-1024)
        y_ = np.pad(y, ((0, 0), (0, rows_to_add), (0, cols_to_add), (0, 0)), mode='constant', constant_values=0)
        yield x_, y_




def weighted_cce(y_true, y_pred):

    weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)
    # weights = weights / K.sum(weights)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = tf.reduce_sum(y_true * K.log(y_pred), axis=-1, keepdims=True) * weights
    loss = -K.sum(loss, -1)
    return loss

model = unet2d(n_input_channels)
model.compile(loss=[weighted_cce], optimizer=optimizers.Adam(1e-4))
training_generator = data_generator(pool_folder, batch_size=1)

while True:
    model.fit_generator(training_generator, steps_per_epoch=4, epochs=2)
    image_paths = [os.path.join(pool_folder, file) for file in os.listdir(pool_folder) if 'image' in file]
    pred_paths = [image_path.replace('image', 'pred') for image_path in image_paths]
    for image_path, pred_path in tqdm(zip(image_paths, pred_paths)):
        image = np.load(image_path).squeeze()
        if image.ndim == 3:
            n_rows, n_cols, _ = image.shape
        else:
            n_rows, n_cols = image.shape
        target_rows = max(128, 2 ** np.ceil(np.log2(n_rows)))
        target_cols = max(128, 2 ** np.ceil(np.log2(n_cols)))
        rows_to_add = int(target_rows - n_rows)
        cols_to_add = int(target_cols - n_cols)
        if image.ndim == 3:
            image = np.pad(image, ((0, rows_to_add), (0, cols_to_add), (0, 0)), mode='constant', constant_values=0)
            image = image[None, ...]
        else:
            image = np.pad(image, ((0, rows_to_add), (0, cols_to_add)), mode='constant', constant_values=-1024)
            image = image[None, :, :, None]
        image = image.squeeze()
        pred = model.predict(image)[0]
        pred = pred[:n_rows, :n_cols, :]
        np.save(arr=pred, file=pred_path)
