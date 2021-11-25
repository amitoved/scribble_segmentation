import os

import numpy as np

from constants import DATA_DIR
from utils import get_paths

if __name__ == "__main__":
    pool_folder = os.path.join(DATA_DIR, 'pool')
    vol = np.load(os.path.join(DATA_DIR, 'head_volume.npy'))

    for idx, slice in enumerate(vol):
        slice_path, pred_path, scribble_path = get_paths(pool_folder, idx)
        pred = np.zeros_like(slice)
        scribble = 255 * np.ones_like(slice)
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
