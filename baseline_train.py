import os
import pathlib

import configargparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, callbacks
import pandas as pd

import constants
from models.architectures import model_types, q_factor
from utils.data_loaders_utils import data_loaders
from utils.general_utils import folder_picker
from utils.image_utils import normalize_image

pool_folder = folder_picker(initialdir=constants.DATA_DIR)

import platform

if 'macOS' in platform.platform():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")

def load_data(image_paths, args):
    n = len(image_paths)
    img, gt = data_loaders[args.data_loader](image_paths[0])
    target_rows, target_cols = q_factor[args.model] * (img.shape[0] // q_factor[args.model]), q_factor[args.model] * (img.shape[1] // q_factor[args.model])

    if img.ndim == 2:
        img = img[..., None]
    sample_image = img[:target_rows, :target_cols, :]
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((n, n_rows, n_cols, n_input_channels))
    y = np.zeros((n, n_rows, n_cols, constants.n_classes))
    for i in range(n):
        image_path = pathlib.Path(image_paths[i])
        img, gt = data_loaders[args.data_loader](image_path)
        x[i] = normalize_image(img[:target_rows, :target_cols])
        y[i] = gt[:target_rows, :target_cols]
    return x, y


def data_generator(image_paths, args):
    img, gt = data_loaders[args.data_loader](image_paths[0])
    target_rows, target_cols = q_factor[args.model] * (img.shape[0] // q_factor[args.model]), q_factor[args.model] * (img.shape[1] // q_factor[args.model])

    sample_image = img[:target_rows, :target_cols, :]
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((args.batch_size, n_rows, n_cols, n_input_channels))
    y = np.zeros((args.batch_size, n_rows, n_cols, constants.n_classes))
    while True:
        for i in range(args.batch_size):
            image_path = pathlib.Path(np.random.choice(image_paths))
            img, gt = data_loaders[args.data_loader](image_path)
            if img.ndim == 2:
                img = img[..., None]
            x[i] = normalize_image(img[:target_rows, :target_cols])
            y[i] = gt[:target_rows, :target_cols]
        yield x, y


def weighted_cce(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = -K.sum(tf.reduce_sum(y_true * K.log(y_pred), axis=-1, keepdims=True) * weights, -1)
    return loss


def config_parser():
    parser = configargparse.ArgumentParser(ignore_unknown_config_file_keys=True)
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--epochs', type=int, help='number of epochs')
    parser.add_argument('--spe', type=int, help='steps per epoch')
    parser.add_argument('--batch', type=int, help='batchsize')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--model', type=str, help='model arch')
    parser.add_argument('--annotate_gt', action='store_true')
    parser.add_argument('--baseline_training_sizes', action='append')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    training_pool = os.path.join(pool_folder, 'train')
    val_pool = os.path.join(pool_folder, 'val')
    training_image_paths = list(pd.read_csv(os.path.join(training_pool, 'priorities.csv')).paths)
    val_image_paths = list(pd.read_csv(os.path.join(val_pool, 'priorities.csv')).paths)

    val_x, val_y = load_data(val_image_paths, args)
    n_input_channels = val_x.shape[-1]

    model = model_types[args.model](n_input_channels)
    model.compile(loss=[weighted_cce], optimizer=optimizers.Adam(args.lr))
    for relative_training_set_size in args.baseline_training_sizes:
        training_set_size = int(float(relative_training_set_size) * len(training_image_paths))
        print('##########')
        print(f'strating training_set_size = {training_set_size}')
        model_path = os.path.join(pool_folder, f'training_log_size_{training_set_size}.h5')
        log_path = os.path.join(pool_folder, f'training_log_size_{training_set_size}.npy')

        checkpoint_callback = callbacks.ModelCheckpoint(filepath=model_path, verbose=1, monitor='val_loss',
                                                        save_best_only=True, save_weights_only=False)
        training_generator = data_generator(training_image_paths[:training_set_size], args)
        training_log = model.fit(training_generator, validation_data=(val_x, val_y), steps_per_epoch=args.spe,
                                 epochs=args.epochs, callbacks=[checkpoint_callback])
        np.save(log_path, training_log.history)
        np.load(log_path, allow_pickle=True)
