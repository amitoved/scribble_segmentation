import os
import platform
from tkinter import filedialog


def folder_picker(initialdir=None, title=None):
    if 'macOS' in platform.platform():
        print('Type folder path')
        folder = str(input())
    else:
        folder = filedialog.askdirectory(initialdir=initialdir, title=title)
    return folder


def file_picker(initialdir=None, title=None):
    if 'macOS' in platform.platform():
        print('Type file path')
        file_path = str(input())
    else:
        file_path = filedialog.askopenfilename(initialdir=initialdir, title=title)
    return file_path


def generate_pool_paths(folder, suffix):
    image_path = os.path.join(folder, 'image_' + str(suffix) + '.npy')
    gt_path = os.path.join(folder, 'gt_' + str(suffix) + '.npy')
    pred_path = os.path.join(folder, 'pred_' + str(suffix) + '.npy')
    scribble_path = os.path.join(folder, 'scribble_' + str(suffix) + '.npy')
    return image_path, gt_path, pred_path, scribble_path


def rgb2tk(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"
