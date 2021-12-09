import os


def generate_pool_paths(folder, suffix):
    image_path = os.path.join(folder, 'image_' + str(suffix) + '.npy')
    pred_path = os.path.join(folder, 'pred_' + str(suffix) + '.npy')
    scribble_path = os.path.join(folder, 'scribble_' + str(suffix) + '.npy')
    return image_path, pred_path, scribble_path


def rgb2tk(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"