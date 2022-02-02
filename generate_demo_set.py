import os
import pathlib

import configargparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from constants import DATA_DIR, SEED
from models.architectures import q_factor
from utils.data_loaders_utils import data_loaders
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
    parser.add_argument('--data_loader', type=str, help='the name of the data loading function')
    parser.add_argument('--max_data', default=None, help='the maximal length of data')
    parser.add_argument('--model', type=str, help='model arch')
    parser.add_argument('--source_folder', type=str, default=None, required=False,
                        help='source folder for pool generation')

    return parser


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    if args.source_folder is None:
        source_folder = folder_picker(title='Choose a folder containing images')
    else:
        source_folder = args.source_folder
    source_files = get_files(folder=source_folder, extensions=['*.png', '*.jpg', '*.dcm', '*.tif'])
    pool_folder = os.path.join(DATA_DIR, args.data_loader)

    if not os.path.exists(pool_folder):
        os.mkdir(pool_folder)
        os.mkdir(os.path.join(pool_folder, 'train'))
        os.mkdir(os.path.join(pool_folder, 'val'))

    df = []
    val_paths = []
    np.random.seed(SEED)
    np.random.shuffle(source_files)
    if args.max_data is not None:
        source_files = source_files[:min(len(source_files), args.max_data)]
    train_val_split = int(args.train_p * len(source_files))
    training_source_files = source_files[:train_val_split]
    val_source_files = source_files[train_val_split:]
    for data_type in ['train', 'val']:
        source_files_subset = training_source_files if data_type == 'train' else val_source_files
        for source_file in tqdm(source_files_subset):
            source_file = str(pathlib.Path(source_file))
            base = os.path.basename(source_file)
            basename, ext = os.path.splitext(base)
            image_path, gt_path, pred_path, scribble_path = generate_pool_paths(os.path.join(pool_folder, data_type),
                                                                                basename)
            data_loader = data_loaders[args.data_loader]
            img, gt, success = data_loader(source_file, q_factor=q_factor[args.model])
            if not success:
                continue

            pred = np.zeros(gt.shape)
            scribble = np.zeros(gt.shape, dtype=bool)
            if data_type == 'train':
                df.append([source_file, 1.])
            else:
                val_paths.append([source_file])
            try:
                np.save(pred_path, (255 * pred).astype(np.uint8))
                np.savez_compressed(scribble_path, scribble)
            except Exception as e:
                print(e)

    df = pd.DataFrame.from_records(df, columns=['paths', 'score'])
    df_path = os.path.join(os.path.join(pool_folder, 'train'), 'priorities.csv')
    df.to_csv(df_path)
    df = pd.DataFrame.from_records(val_paths, columns=['paths'])
    df_path = os.path.join(os.path.join(pool_folder, 'val'), 'priorities.csv')
    df.to_csv(df_path)

    timer_path = os.path.join(pool_folder, 'train', 'timer.txt')
    with open(timer_path, 'w+') as f:
        f.write('')
    print('Done')
