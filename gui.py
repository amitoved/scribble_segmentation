import os
import pathlib
import platform
import tkinter as tk
from tkinter import colorchooser, StringVar, filedialog

import numpy as np
from PIL import ImageTk, Image, ImageDraw

import constants
from utils.general_utils import rgb2tk
from utils.image_utils import normalize_image, multichannel2rgb


class App:

    def __init__(self):
        self.color = '#FF0000'
        self.figure = 'rectangle'
        self.size = 5
        self.class_val = None
        # self.colors = constants.TK_COLORS[0:: len(constants.TK_COLORS) // len(constants.classes_order)]
        self.pil_colors = constants.class_colors
        self.annotations = {val: [] for val in constants.classes.values()}
        self.window = tk.Tk()
        self.pool_folder = os.path.join(constants.DATA_DIR, 'pool')
        self.selecting_file()
        # self.window.after(100, self.selecting_file)
        self.window.mainloop()

    def selecting_file(self, update=False):
        if not update:
            if 'macOS' in platform.platform():
                print('Pick pool folder')
                file_path_string = str(input())
            else:
                file_path_string = filedialog.askdirectory(initialdir=constants.DATA_DIR)
            self.scribble_paths = [pathlib.Path(file_path_string, file) for file in os.listdir(file_path_string) if
                                   'scribble_' in file]
        self.scribble_path = np.random.choice(self.scribble_paths)
        self.image_path = pathlib.Path(self.scribble_path.parent,
                                       self.scribble_path.name.replace('scribble_', 'image_'))
        self.pred_path = pathlib.Path(self.scribble_path.parent, self.scribble_path.name.replace('scribble_', 'pred_'))
        image = np.load(self.image_path)
        self.height, self.width, n_input_chanels = image.shape
        self.annotation_img = Image.new('RGB', (self.width, self.height))

        pred = np.load(self.pred_path)
        pred = multichannel2rgb(pred)
        image = image / np.max(image)
        if n_input_chanels == 1:
            image = np.concatenate([image] * 3, axis=-1)
        image = image * (1 - constants.alpha) + constants.alpha * pred
        image = (image * 255).astype(np.uint8)
        self.image = Image.fromarray(image)
        self.draw = ImageDraw.Draw(self.image)
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.scribble = Image.fromarray(255 * np.ones(list(pred.shape[:2])).astype(np.uint8))

        if not update:
            self.frame_tools = tk.Frame(self.window)
            self.frame_tools.pack()
            self.selected_class = StringVar()
            self.option_menu = tk.OptionMenu(self.frame_tools, self.selected_class, *constants.classes_order)
            self.option_menu.pack(side='left')
            self.save_button = tk.Button(self.frame_tools, text='save', command=self.save)
            self.save_button.pack(side='left')

            self.size_entry = tk.Entry(self.frame_tools)
            self.size_entry.bind('<Return>', self.change_size)
            self.size_entry.pack(side='left')

            # self.color_button = tk.Button(self.frame_tools, text='Color: #FF0000', command=self.change_color)
            # self.color_button.pack(side='left')

            self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
            self.canvas.pack()
            self.canvas.bind("<Button-1>", self.get_x_and_y)
            self.canvas.bind("<B1-Motion>", self.draw_smth)

        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def get_x_and_y(self, event):
        self.last_x, self.last_y = event.x, event.y

    def draw_smth(self, event):
        self.class_val = constants.classes_order.index(self.selected_class.get())
        # color = self.colors[self.class_val]
        pil_color = tuple((255 * np.array(list(self.pil_colors[self.class_val]))).astype(int)[:-1])
        self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), fill=rgb2tk(pil_color), width=2)
        self.annotations[self.class_val].append([event.x, event.y])
        img1 = ImageDraw.Draw(self.annotation_img)

        # img1.line((self.last_x, self.last_y, event.x, event.y), fill=pil_color, width=0)
        img1.line((self.last_x, self.last_y, event.x, event.y), fill=self.class_val, width=5)
        img1 = ImageDraw.Draw(self.scribble)
        self.last_x, self.last_y = event.x, event.y

    def update_scribble(self):
        scribble = np.load(self.scribble_path)
        current_scribble = np.array(self.scribble)
        r, c = np.where(current_scribble != 255)
        val = current_scribble[r, c]
        scribble[r, c, val] = scribble[r, c, val] == False
        self.scribble = scribble

    def save(self):
        self.update_scribble()
        np.save(self.scribble_path, self.scribble)
        print('saved scribble to' + self.scribble_path)
        self.last_x, self.last_y = None, None
        self.annotations = {val: [] for val in constants.classes.values()}
        self.class_val = None
        self.selecting_file(update=True)

    def change_color(self):
        rgb, color_string = colorchooser.askcolor(initialcolor=self.color)
        # print(rgb, color_string)
        if color_string:
            self.color = color_string
            self.color_button['text'] = 'Color: ' + color_string

    def change_size(self, event=None):
        try:
            self.size = int(self.size_entry.get())
        except Exceptions as ex:
            print(ex)


App()
