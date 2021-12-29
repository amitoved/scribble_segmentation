import os
import pathlib

import configargparse
import imageio
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from pydicom import dcmread
from tqdm import tqdm

from constants import DATA_DIR, n_classes, SEED
from utils.general_utils import generate_pool_paths, folder_picker


def get_files(folder, extensions):
    from pathlib import Path
    all_files = []
    for ext in extensions:
        all_files.extend(Path(folder).glob(ext))
    return all_files


def config_parser():
    parser = configargparse.ArgumentParser(ignore_unknown_config_file_keys=True)
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--train_p', type=float, help='training data proportion')
    return parser


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    q = 32
    source_folder = folder_picker(title='Choose a folder containing images')
    source_files = get_files(folder=source_folder, extensions=['*.png', '*.jpg', '*.dcm'])
    pool_name = os.path.basename(source_folder)
    pool_folder = os.path.join(DATA_DIR, pool_name)

    if not os.path.exists(pool_folder):
        os.mkdir(pool_folder)
        os.mkdir(os.path.join(pool_folder, 'train'))
        os.mkdir(os.path.join(pool_folder, 'val'))

    df = []
    np.random.seed(SEED)
    np.random.shuffle(source_files)
    n_files = len(source_files)
    training_source_files = source_files[:int(args.train_p * n_files)]
    val_source_files = source_files[int(args.train_p * n_files):]
    for data_type in ['train', 'val']:
        source_files_subset = training_source_files if data_type == 'train' else val_source_files
        for source_file in tqdm(source_files_subset):
            source_file = str(pathlib.Path(source_file))
            base = os.path.basename(source_file)
            basename, ext = os.path.splitext(base)
            image_path, gt_path, pred_path, scribble_path = generate_pool_paths(os.path.join(pool_folder, data_type),
                                                                                basename)
            if ext == '.dcm':
                dcm = dcmread(source_file)
                img = dcm.pixel_array
            elif ext in ['.png', 'jpg']:
                img = imageio.imread(source_file)
                gt = imageio.imread(source_file.replace('image_2', 'semantic'))

            target_rows, target_cols = q * (img.shape[0] // q), q * (img.shape[1] // q)

            if img.ndim == 2:
                img = img[..., None]
            img = img[:target_rows, :target_cols, :]
            gt = gt[:target_rows, :target_cols]
            gt = (gt == 7).astype(int)  # on vkitti, 7 is the the road
            gt = to_categorical(gt, num_classes=2)
            pred = np.zeros([target_rows, target_cols, n_classes])
            scribble = np.zeros([target_rows, target_cols, n_classes], dtype=bool)
            if data_type == 'train':
                df.append([image_path, 1.])
            np.save(image_path, img)
            np.save(gt_path, gt)
            np.save(pred_path, pred)
            np.save(scribble_path, scribble)

    df = pd.DataFrame.from_records(df, columns=['paths', 'score'])
    df_path = os.path.join(os.path.join(pool_folder, 'train'), 'priorities.csv')
    df.to_csv(df_path)

    timer_path = os.path.join(pool_folder, 'train', 'timer.txt')
    with open(timer_path, 'w') as f:
        f.write('')
    print('Done')
