import os
import pathlib

import configargparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tqdm import tqdm

import constants
from models.architectures import model_types, q_factor
from utils.data_loaders_utils import data_loaders
from utils.general_utils import folder_picker, generate_pool_paths
from utils.image_utils import normalize_image, img_tv

pool_folder = folder_picker(initialdir=constants.DATA_DIR)

import platform

if 'macOS' in platform.platform():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")


def data_generator(args):
    df = pd.read_csv(os.path.join(pool_folder, 'train', 'priorities.csv'))
    image_paths = list(df['paths'])

    sample_image, _, _ = data_loaders[args.data_loader](image_paths[0])
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((args.batch, n_rows, n_cols, n_input_channels))
    y = np.zeros((args.batch, n_rows, n_cols, constants.n_classes))
    while True:
        df = pd.read_csv(os.path.join(pool_folder, 'train', 'priorities.csv'))
        image_paths = list(df['paths'])
        priorities = list(df['score'])
        priorities = np.array(priorities) / np.sum(priorities)

        for i in range(args.batch):
            found_non_empty_scribble = False
            while not found_non_empty_scribble:
                image_path = np.random.choice(image_paths, p=priorities)
                image_path = pathlib.Path(image_path)
                basename, _ = os.path.splitext(os.path.basename(image_path))
                _, gt_path, _, scribble_path = generate_pool_paths(os.path.join(pool_folder, 'train'), basename)
                true = np.load(scribble_path)['arr_0']

                if np.any(true):
                    img, _, _ = data_loaders[args.data_loader](image_path)
                    x[i] = normalize_image(img)
                    y[i] = true.astype(int)
                    found_non_empty_scribble = True
        yield x, y


def load_data(image_paths, args):
    n = len(image_paths)
    img, gt, _ = data_loaders[args.data_loader](image_paths[0])
    target_rows, target_cols = q_factor[args.model] * (img.shape[0] // q_factor[args.model]), q_factor[args.model] * (img.shape[1] // q_factor[args.model])

    if img.ndim == 2:
        img = img[..., None]
    sample_image = img[:target_rows, :target_cols, :]
    n_rows, n_cols, n_input_channels = sample_image.shape
    x = np.zeros((n, n_rows, n_cols, n_input_channels))
    y = np.zeros((n, n_rows, n_cols, constants.n_classes))
    for i in range(n):
        image_path = pathlib.Path(image_paths[i])
        img, gt, _ = data_loaders[args.data_loader](image_path)
        x[i] = normalize_image(img[:target_rows, :target_cols])
        y[i] = gt[:target_rows, :target_cols]
    return x, y


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
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--annotate_gt', action='store_true')
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--data_loader', type=str, help='the name of the data loading function')

    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    training_generator = data_generator(args)
    x_sample, _ = next(training_generator)

    val_pool = os.path.join(pool_folder, 'val')
    val_image_paths = list(pd.read_csv(os.path.join(val_pool, 'priorities.csv')).paths)
    val_x, val_y = load_data(val_image_paths, args)

    n_input_channels = x_sample.shape[-1]
    model = model_types[args.model](n_input_channels)
    model.compile(loss=[weighted_cce], optimizer=optimizers.Adam(args.lr))

    image_paths = [pathlib.Path(pool_folder, file) for file in os.listdir(pool_folder) if 'image_' in file]
    pred_paths = [pathlib.Path(image_path.parent, image_path.name.replace('image_', 'pred_')) for image_path in
                  image_paths]
    gt_paths = [pathlib.Path(image_path.parent, image_path.name.replace('image_', 'gt_')) for image_path in
                image_paths]

    while True:
        model.fit(training_generator, validation_data=(val_x, val_y), steps_per_epoch=args.spe,
                  epochs=args.epochs)
        df = []
        for image_path, pred_path in tqdm(zip(image_paths, pred_paths)):
            image = np.load(image_path)
            pred = model.predict(image[None, ...])[0]
            score = img_tv(pred)
            df.append([image_path, score])
            np.save(arr=(pred * 255).astype(np.uint8), file=pred_path)
        df = pd.DataFrame.from_records(df, columns=['paths', 'score'])
        df.to_csv(constants.PRIORITY_DF)
