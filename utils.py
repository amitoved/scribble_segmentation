import os


def get_paths(folder, idx):
    slice_path = os.path.join(folder, 'image_' + str(idx) + '.npy')
    pred_path = os.path.join(folder, 'pred_' + str(idx) + '.npy')
    scribble_path = os.path.join(folder, 'scribble_' + str(idx) +'.npy')
    return slice_path, pred_path, scribble_path
