import os


def get_paths(folder, idx):
    slice_path = os.path.join(folder, f'image_{idx}.npy')
    pred_path = os.path.join(folder, f'pred_{idx}.npy')
    scribble_path = os.path.join(folder, f'scribble_{idx}.npy')
    return slice_path, pred_path, scribble_path
