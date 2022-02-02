import numpy as np
from skimage import filters


def normalize_image(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)


def img_tv(y_pred):
    # https://en.wikipedia.org/wiki/Total_variation_denoising
    dx = y_pred[1:, 1:] - y_pred[1:, :-1]
    dy = y_pred[1:, 1:] - y_pred[:-1, 1:]
    return np.mean((dx ** 2 + dy ** 2) ** 0.5)


def multichannel2rgb(x):
    # This function accepts an array of shape [rows, cols, channels]
    # and returns and RGB array of shape [rows, cols, 3]
    assert x.ndim == 3, 'input array shape must be [rows, cols, channels]'
    if type(x[0, 0, 0]) is np.bool_:
        x = x * 1.0
    n_channels = x.shape[-1]
    x = x - np.min(x)

    red = np.linspace(1, 0, n_channels)
    green = 1 - np.abs(np.linspace(-1, 1, n_channels))
    blue = np.linspace(0, 1, n_channels)
    filters = [f / np.sum(f + 1e-9) for f in [red, green, blue]]
    rgb = np.stack([np.sum(x * f, axis=-1) for f in filters], axis=-1)
    rgb = rgb / (np.max(rgb) + 1e-9)
    return rgb


def generate_colormap(n_classes, h, w):
    colormap = np.linspace(0, n_classes, h, dtype=int, endpoint=False)
    v = np.zeros((len(colormap), n_classes))
    for i in range(len(colormap)):
        v[i, colormap[i]] = 1

    return multichannel2rgb(np.stack([v] * w, axis=1))
