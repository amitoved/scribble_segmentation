import os
import pathlib
import tkinter as tk
from time import time
from tkinter import colorchooser, StringVar

import configargparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk, Image, ImageDraw
from skimage.transform import resize

import constants
from utils.data_loaders_utils import data_loaders
from utils.general_utils import rgb2tk, folder_picker, generate_pool_paths
from utils.image_utils import multichannel2rgb, generate_colormap


class App:

    def __init__(self, args):
        self.color = '#FF0000'
        self.figure = 'rectangle'
        self.size = 5
        self.class_val = None
        self.pil_colors = constants.class_colors
        self.annotations = {val: [] for val in constants.classes.values()}
        self.window = tk.Tk()
        self.annotate_gt = args.annotate_gt

        screen_w, screen_h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.geometry(f'{screen_w}x{screen_h}')
        self.selecting_file()
        self.window.mainloop()

    def selecting_file(self, update=False):
        if not update:
            print('Select the training folder')
            self.pool_folder = folder_picker(initialdir=constants.DATA_DIR, title='pick training folder')
            self.prob_df = pd.read_csv(os.path.join(os.path.join(self.pool_folder, 'priorities.csv')))
            self.timer_path = os.path.join(self.pool_folder, 'timer.txt')

        self.image_path = self.prob_df.iloc[np.random.randint(0, len(self.prob_df))].paths
        basename, _ = os.path.splitext(os.path.basename(self.image_path))
        _, _, self.pred_path, self.scribble_path = generate_pool_paths(self.pool_folder, basename)
        image, gt = data_loaders[args.data_loader](self.image_path)

        if self.annotate_gt:
            image = multichannel2rgb(gt)

        pred = np.load(self.pred_path) / 255.
        pred = multichannel2rgb(pred)
        image = image / np.max(image)
        self.height_o, self.width_o, n_input_chanels = image.shape

        if n_input_chanels == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image = image * (1 - constants.alpha) + constants.alpha * pred
        image = (image * 255).astype(np.uint8)
        self.image = Image.fromarray(image)
        self.colormap_np = generate_colormap(constants.n_classes, pred.shape[0], int(pred.shape[0] / 10))
        fig = plt.figure()
        plt.imshow(self.colormap_np)
        plt.yticks(
            self.colormap_np.shape[0] / len(constants.classes_order) * np.arange(1, 1 + len(constants.classes_order)),
            constants.classes_order, rotation=-90, va="bottom")
        fig.canvas.draw()
        self.colormap_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        self.colormap_np = self.colormap_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.colormap_np = self.colormap_np[30:-30, 200:-200, :]
        # self.colormap_np = 255 * resize(self.colormap_np, (image.shape[0], image.shape[1] // 10))

        screen_w, screen_h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        dim = np.argmin(np.array([screen_h, screen_w]) - 1.1 * (
                np.array(image.shape[:2]) + np.array(self.colormap_np.shape[:2]) * np.array([0, 1])))
        self.factor = np.array([screen_h, screen_w])[dim] / (
                1.1 * (np.array(image.shape[:2]) + np.array(self.colormap_np.shape[:2]) * np.array([0, 1]))[dim])
        self.resized_image = self.image.resize((int(image.shape[1] * self.factor), int(image.shape[0] * self.factor)))
        self.photo = ImageTk.PhotoImage(image=self.resized_image)

        self.height = int(self.height_o * self.factor)
        self.width = int(self.width_o * self.factor)
        self.annotation_img = Image.new('RGB', (self.width, self.height))

        self.colormap = ImageTk.PhotoImage(Image.fromarray((self.colormap_np).astype('uint8')))

        self.draw = ImageDraw.Draw(self.image)
        self.scribble = Image.fromarray(255 * np.ones([self.height, self.width]).astype(np.uint8))
        self.start_time = time()
        if not update:
            self.frame_tools = tk.Frame(self.window)
            self.frame_tools.pack()
            self.selected_class = StringVar()
            self.option_menu = tk.OptionMenu(self.frame_tools, self.selected_class, *constants.classes_order)
            self.option_menu.pack(side='left')
            self.brush_size = StringVar()
            self.brush_size.set(constants.BRUSH_SIZES[2])
            self.option_menu = tk.OptionMenu(self.frame_tools, self.brush_size, *constants.BRUSH_SIZES)
            self.option_menu.pack(side='left')

            self.save_button = tk.Button(self.frame_tools, text='save', command=self.save)
            self.save_button.pack(side='left')

            self.clear_button = tk.Button(self.frame_tools, text='clear', command=self.clear)
            self.clear_button.pack(side='left')

            self.size_entry = tk.Entry(self.frame_tools)
            self.size_entry.bind('<Return>', self.change_size)
            self.size_entry.pack(side='left')

            # self.color_button = tk.Button(self.frame_tools, text='Color: #FF0000', command=self.change_color)
            # self.color_button.pack(side='left')

            self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
            self.canvas.pack(side='left')
            self.colormap_canvas = tk.Canvas(self.window, width=self.colormap_np.shape[1],
                                             height=self.colormap_np.shape[0])
            self.colormap_canvas.pack(side='right')

            self.canvas.bind("<Button-1>", self.get_x_and_y)
            self.canvas.bind("<B1-Motion>", self.draw_smth)

        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.colormap_canvas.create_image(0, 0, image=self.colormap, anchor=tk.NW)

        self.window.geometry(f'{screen_w}x{screen_h}')

    def get_x_and_y(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_smth(self, event):
        self.class_val = constants.classes_order.index(self.selected_class.get())
        pil_rgb = rgb2tk(tuple((255 * np.array(list(self.pil_colors[self.class_val]))).astype(int)[:-1]))

        brush_size = int(self.brush_size.get())
        self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), fill=pil_rgb, width=brush_size)
        self.annotations[self.class_val].append([event.x, event.y])

        img1 = ImageDraw.Draw(self.scribble)
        img1.line((self.last_x, self.last_y, event.x, event.y), fill=self.class_val, width=brush_size)
        self.last_x, self.last_y = event.x, event.y

    def update_scribble(self):
        scribble = np.load(self.scribble_path)['arr_0']
        current_scribble = np.array(self.scribble)
        current_scribble = resize(current_scribble, (self.height_o, self.width_o), order=0, anti_aliasing=False)
        r, c = np.where(current_scribble != 255)
        val = current_scribble[r, c]
        scribble[r, c, val.astype(int)] = scribble[r, c, val.astype(int)] == False
        self.scribble = scribble

    def clear(self):
        self.last_x, self.last_y = None, None
        self.annotations = {val: [] for val in constants.classes.values()}
        self.class_val = None
        self.scribble = np.zeros_like(self.scribble)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def save(self):
        if np.any(np.array(self.scribble)):
            self.update_scribble()
            np.savez_compressed(self.scribble_path, self.scribble)
            print(self.scribble_path)

            with open(self.timer_path, 'a+') as f:
                f.write(f'{self.scribble_path},{time() - self.start_time}\n')

        self.last_x, self.last_y = None, None
        self.annotations = {val: [] for val in constants.classes.values()}
        self.class_val = None
        self.selecting_file(update=True)

    def change_color(self):
        rgb, color_string = colorchooser.askcolor(initialcolor=self.color)
        if color_string:
            self.color = color_string
            self.color_button['text'] = 'Color: ' + color_string

    def change_size(self, event=None):
        try:
            self.size = int(self.size_entry.get())
        except Exceptions as ex:
            print(ex)


def config_parser():
    parser = configargparse.ArgumentParser(ignore_unknown_config_file_keys=True)
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # parser.add_argument('--epochs', type=int, help='number of epochs')
    # parser.add_argument('--spe', type=int, help='steps per epoch')
    # parser.add_argument('--batch', type=int, help='batchsize')
    # parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--annotate_gt', action='store_true')
    parser.add_argument('--data_loader', type=str, help='the name of the data loading function')
    parser.add_argument('--q', type=int, help='the image size should be a multiplier of this number')


    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    print(args)

    App(args)
