import os

import numpy as np

import constants
from constants import DATA_DIR
from utils.utils import get_paths

if __name__ == "__main__":
    pool_folder = os.path.join(DATA_DIR, 'pool')
    vol = np.load(os.path.join(DATA_DIR, 'head_volume.npy'))

    for idx, slice in enumerate(vol[:100]):
        slice_path, pred_path, scribble_path = get_paths(pool_folder, idx)
        n_rows, n_cols = slice.shape
        pred = np.zeros([n_rows, n_cols, constants.n_classes])
        scribble = np.zeros([n_rows, n_cols, constants.n_classes], dtype=np.bool)
        np.save(slice_path, slice)
        np.save(pred_path, pred)
        np.save(scribble_path, scribble)
    print('Done')

# def _sample_slice(pool_folder):
#     files = os.listdir(pool_folder)
#     full_paths = [os.path.join(pool_folder, file) for file in files if file.endswith('.npy')]
#     chosen_file = np.random.choice(full_paths)
#     img = np.load(chosen_file)
#     img = img - np.min(img)
#     img = img / np.max(img)
#     img = img * 255
#     return img
