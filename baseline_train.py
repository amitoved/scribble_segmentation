import os
import pathlib

import configargparse
import numpy as np
import pandas as pd
import skimage
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import optimizers
from tqdm import tqdm

import constants
from models.architectures import model_types
from utils.general_utils import folder_picker
from utils.image_utils import normalize_image
import platform

np.random.seed(constants.SEED)
pool_folder = folder_picker(initialdir=constants.DATA_DIR)


if 'macOS' in platform.platform():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")




def data_generator(batch_size, image_paths):

    sample_image = np.load(image_paths[0])
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((batch_size, n_rows, n_cols, n_input_channels))
    y = np.zeros((batch_size, n_rows, n_cols, constants.n_classes))
    while True:
        for i in range(batch_size):
            image_path = np.random.choice(image_paths)
            image_path = pathlib.Path(image_path)

            gt_path = pathlib.Path(image_path.parent, image_path.name.replace('image_', 'gt_'))
            true = np.load(gt_path)

            if np.any(true):
                img = np.load(image_path)
                x[i] = normalize_image(img)
                y[i] = true.astype(int)
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
    parser.add_argument('--annotate_gt', action='store_true')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--train_p', type=float, help='training data proportion')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    log_file = os.path.join(pool_folder, 'baseline_log.csv')
    image_paths = [pathlib.Path(pool_folder, file) for file in os.listdir(pool_folder) if 'image_' in file]
    np.random.shuffle(image_paths)
    train_image_paths = image_paths[:int(args.train_p * len(image_paths))]
    val_image_paths = image_paths[int(args.train_p * len(image_paths)) + 1 :]
    n_train_full = len(train_image_paths)
    n_val_full = len(val_image_paths)


    training_generator = data_generator(batch_size=args.batch, image_paths=train_image_paths)
    validation_generator = data_generator(batch_size=args.batch, image_paths=val_image_paths)
    x, y = next(training_generator)
    n_input_channels = x.shape[-1]
    model = model_types[args.model](n_input_channels)
    model.compile(loss=[weighted_cce], optimizer=optimizers.Adam(args.lr))

    early_stopping_cbk = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

    for data_size in [0.1, 0.3, 0.5, 0.7, 0.9]:
        training_generator = data_generator(batch_size=args.batch,
                                            image_paths=train_image_paths[:int(data_size * n_train_full)])
        validation_generator = data_generator(batch_size=args.batch, image_paths=val_image_paths)
        model.fit(training_generator, validation_data=validation_generator, steps_per_epoch=args.spe,
                  epochs=args.epochs, callbacks=[early_stopping_cbk])

        loss_per_data_size = model.evaluate(validation_generator)
        with open(log_file, 'a+') as log:
            log.write(f'{int(data_size * n_train_full)},{loss_per_data_size}\n')
