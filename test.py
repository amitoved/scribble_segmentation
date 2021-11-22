import os

import numpy as np

pool_folder = r"C:\Users\Amit\PycharmProjects\scribble_segmentation\pool"
vol = np.load('head_volume.npy')

def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y

def draw_smth(event):
    global lasx, lasy
    canvas.create_line((lasx, lasy, event.x, event.y),
                      fill='red',
                      width=2)
    lasx, lasy = event.x, event.y

def get_paths(folder, idx):
    slice_path = os.path.join(pool_folder, f'slice_{idx}.npy')
    pred_path = os.path.join(pool_folder, f'pred_{idx}.npy')
    scribble_path = os.path.join(pool_folder, f'scribble_{idx}.npy')
    return slice_path, pred_path, scribble_path


for idx, slice in enumerate(vol):
    slice_path, pred_path, scribble_path = get_paths(pool_folder, idx)
    np.save(slice_path, vol[idx])


def _sample_slice(pool_folder):
    files = os.listdir(pool_folder)
    full_paths = [os.path.join(pool_folder, file) for file in files if file.endswith('.npy')]
    chosen_file = np.random.choice(full_paths)
    img = np.load(chosen_file)
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img


from tkinter import *
from PIL import ImageTk, Image
root = Tk()
canvas = Canvas(root, width=300, height=300)
canvas.pack()
img = _sample_slice(pool_folder)
img = ImageTk.PhotoImage(Image.fromarray(_sample_slice(pool_folder)))
canvas.create_image(20, 20, anchor=NW, image=img)

canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw_smth)

root.mainloop()
