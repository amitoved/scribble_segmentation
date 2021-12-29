import os
import pathlib

import configargparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, callbacks

import constants
from models.architectures import models_types
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

def load_data(image_paths):
    n = len(image_paths)
    sample_image = np.load(image_paths[0])
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((n, n_rows, n_cols, n_input_channels))
    y = np.zeros((n, n_rows, n_cols, constants.n_classes))
    for i in range(n):
        image_path = pathlib.Path(image_paths[i])
        gt_path = pathlib.Path(image_path.parent, image_path.name.replace('image_', 'gt_'))
        img = np.load(image_path)
        x[i] = normalize_image(img)
        y[i] = np.load(gt_path)
    return x, y


def data_generator(image_paths, batch_size):
    sample_image = np.load(image_paths[0])
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((batch_size, n_rows, n_cols, n_input_channels))
    y = np.zeros((batch_size, n_rows, n_cols, constants.n_classes))
    while True:
        for i in range(batch_size):
            image_path = pathlib.Path(np.random.choice(image_paths))
            gt_path = pathlib.Path(image_path.parent, image_path.name.replace('image_', 'gt_'))
            img = np.load(image_path)
            x[i] = normalize_image(img)
            y[i] = np.load(gt_path)
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
    parser.add_argument('--model_arch', type=str, help='model arch')
    parser.add_argument('--annotate_gt', action='store_true')
    parser.add_argument('--baseline_training_sizes', action='append')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    training_pool = os.path.join(pool_folder, 'train')
    val_pool = os.path.join(pool_folder, 'val')
    training_image_paths = [os.path.join(training_pool, file) for file in os.listdir(training_pool) if 'image_' in file]
    val_image_paths = [os.path.join(val_pool, file) for file in os.listdir(val_pool) if 'image_' in file]

    val_x, val_y = load_data(val_image_paths)
    n_input_channels = val_x.shape[-1]

    model = models_types[args.model_arch](n_input_channels)
    model.compile(loss=[weighted_cce], optimizer=optimizers.Adam(args.lr))
    for relative_training_set_size in args.baseline_training_sizes:
        training_set_size = int(float(relative_training_set_size) * len(training_image_paths))
        print('##########')
        print(f'strating training_set_size = {training_set_size}')
        model_path = os.path.join(pool_folder, f'training_log_size_{training_set_size}.h5')
        log_path = os.path.join(pool_folder, f'training_log_size_{training_set_size}.npz')

        checkpoint_callback = callbacks.ModelCheckpoint(filepath=model_path, verbose=1, monitor='val_loss',
                                                        save_best_only=True, save_weights_only=False)
        training_generator = data_generator(training_image_paths[:training_set_size], batch_size=args.batch)
        training_log = model.fit(training_generator, validation_data=(val_x, val_y), steps_per_epoch=args.spe,
                                 epochs=args.epochs, callbacks=[checkpoint_callback])
        np.save(log_path, training_log)
