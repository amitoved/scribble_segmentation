import os
from tkinter import *
from tkinter import filedialog

import imageio
import numpy as np
from tqdm import tqdm

from constants import DATA_DIR, n_classes
from utils.utils import generate_pool_paths


def get_files(folder, extensions):
    from pathlib import Path
    all_files = []
    for ext in extensions:
        all_files.extend(Path(folder).glob(ext))
    return all_files


if __name__ == "__main__":

    root = Tk()
    root.withdraw()
    source_folder = filedialog.askdirectory(initialdir=DATA_DIR)
    pool_name = os.path.basename(source_folder)
    pool_folder = os.path.join(DATA_DIR, pool_name)
    if not os.path.exists(pool_folder):
        os.mkdir(pool_folder)
    source_files = get_files(folder=source_folder, extensions=['*.png', '*.jpg'])
    q = 32

    for source_file in tqdm(source_files):
        base = os.path.basename(source_file)
        basename, ext = os.path.splitext(base)
        image_path, pred_path, scribble_path = generate_pool_paths(pool_folder, basename)
        img = imageio.imread(source_file)
        target_rows, target_cols = q * (img.shape[0] // q), q * (img.shape[1] // q)

        if img.ndim == 3:
            img = img[:target_rows, :target_cols, :]
        else:
            img = img[:target_rows, :target_cols]

        pred = np.zeros([target_rows, target_cols, n_classes])
        scribble = np.zeros([target_rows, target_cols, n_classes], dtype=np.bool)

        np.save(image_path, img)
        np.save(pred_path, pred)
        np.save(scribble_path, scribble)
    print('Done')
