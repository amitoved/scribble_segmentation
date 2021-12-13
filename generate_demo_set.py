import os
import pathlib

import imageio
import numpy as np
from tqdm import tqdm

from constants import DATA_DIR, n_classes
from utils.general_utils import generate_pool_paths, folder_picker, file_picker
from utils.image_utils import normalize_image


def get_files(folder, extensions):
    from pathlib import Path
    all_files = []
    for ext in extensions:
        all_files.extend(Path(folder).glob(ext))
    return all_files


if __name__ == "__main__":
    source_file = file_picker(title='Choose an *.npy file')
    if source_file == '':
        source_folder = folder_picker(title='Choose a folder containing images')
        source_files = get_files(folder=source_folder, extensions=['*.png', '*.jpg'])
        pool_name = os.path.basename(source_folder)
        is_folder = True
    else:
        pool_name, ext = os.path.splitext(pathlib.Path(source_file).name)

        if ext == '.npy':
            vol = np.load(source_file)
        is_folder = False
    pool_folder = os.path.join(DATA_DIR, pool_name)

    if not os.path.exists(pool_folder):
        os.mkdir(pool_folder)
    q = 32
    if is_folder:
        for source_file in tqdm(source_files):
            base = os.path.basename(source_file)
            basename, ext = os.path.splitext(base)
            image_path, pred_path, scribble_path = generate_pool_paths(pool_folder, basename)
            img = imageio.imread(source_file)
            target_rows, target_cols = q * (img.shape[0] // q), q * (img.shape[1] // q)

            if img.ndim == 2:
                img = img[..., None]
            img = img[:target_rows, :target_cols, :]
            pred = np.zeros([target_rows, target_cols, n_classes])
            scribble = np.zeros([target_rows, target_cols, n_classes], dtype=bool)

            np.save(image_path, img)
            np.save(pred_path, pred)
            np.save(scribble_path, scribble)
    else:
        for idx, img in tqdm(enumerate(vol[::2])):
            target_rows, target_cols = q * (img.shape[0] // q), q * (img.shape[1] // q)
            img = img[:target_rows, :target_cols]
            img = np.clip(img, -200, 1000)
            img = normalize_image(img)
            img = img[..., None]
            pred = np.zeros([target_rows, target_cols, n_classes])
            scribble = np.zeros([target_rows, target_cols, n_classes], dtype=bool)
            image_path, pred_path, scribble_path = generate_pool_paths(pool_folder, idx)

            np.save(image_path, img)
            np.save(pred_path, pred)
            np.save(scribble_path, scribble)

        print('Done')
