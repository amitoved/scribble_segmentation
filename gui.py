import os
import tkinter as tk
from tkinter import colorchooser

import numpy as np
from matplotlib import pyplot as plt

cmap = plt.get_cmap('jet')

from PIL import ImageTk, Image, ImageDraw

import constants
from utils.utils import normalize, multichannel2rgb


class App:

    def __init__(self):
        self.color = '#FF0000'
        self.figure = 'rectangle'
        self.size = 5
        self.window = tk.Tk()
        self.pool_folder = os.path.join(constants.DATA_DIR, 'pool')
        self.selecting_file()
        # self.window.after(100, self.selecting_file)
        self.window.mainloop()

    def selecting_file(self, update=False):

        self.class_val = None
        self.last_x, self.last_y = None, None
        self.annotations = {val: [] for val in constants.classes.values()}
        image_paths = [os.path.join(self.pool_folder, file) for file in os.listdir(self.pool_folder) if 'image' in file]
        self.image_path = np.random.choice(image_paths)
        self.scribble_path = self.image_path.replace('image', 'scribble')
        self.pred_path = self.image_path.replace('image', 'pred')

        # self.image_path = filedialog.askopenfilename()
        # self.file_path = filedialog.askopenfilename(initialdir='images/', initialfile='example.jpg')

        # pred = Image.fromarray(np.load(self.pred_path))

        image = np.load(self.image_path)
        image = normalize(image)
        self.height, self.width = image.shape

        pred = np.load(self.pred_path)
        pred[100:200, 30:150, 0] = 0.9
        pred[50:80, 30:150, 1] = 0.7
        pred = multichannel2rgb(pred)
        image = np.stack([image] * 3, axis=-1) * (1 - constants.alpha) + constants.alpha * pred
        image = (image * 255).astype(np.uint8)
        self.image = Image.fromarray(image)
        self.draw = ImageDraw.Draw(self.image)
        self.photo = ImageTk.PhotoImage(image=self.image)
        self.scribble = Image.fromarray(255 * np.ones(list(pred.shape[:2])).astype(np.uint8))

        if not update:
            self.frame_tools = tk.Frame(self.window)
            self.frame_tools.pack()

            self.background_button = tk.Button(self.frame_tools, text=constants.BACKGROUND,
                                               command=lambda: self.change_class(
                                                   constants.classes[constants.BACKGROUND]))
            self.background_button.pack(side='left')
            self.foreground_button = tk.Button(self.frame_tools, text=constants.FOREGROUND,
                                               command=lambda: self.change_class(
                                                   constants.classes[constants.FOREGROUND]))
            self.foreground_button.pack(side='left')

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
        if self.class_val == constants.classes[constants.FOREGROUND]:
            color = 'red'
        elif self.class_val == constants.classes[constants.BACKGROUND]:
            color = 'blue'
        else:
            return
        self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), fill=color, width=2)
        self.last_x, self.last_y = event.x, event.y
        self.annotations[self.class_val].append([event.x, event.y])
        img1 = ImageDraw.Draw(self.scribble)
        img1.line((self.last_x, self.last_y, event.x, event.y), fill=self.class_val, width=5)
        # TODO: change line width (currently - only draws the vertices)
        # self.annotation_img[event.y, event.x, self.class_val] = 1

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
        self.selecting_file(update=True)

    def change_color(self):
        rgb, color_string = colorchooser.askcolor(initialcolor=self.color)
        # print(rgb, color_string)
        if color_string:
            self.color = color_string
            self.color_button['text'] = 'Color: ' + color_string

    def change_class(self, val):
        self.class_val = val

    def change_size(self, event=None):
        try:
            self.size = int(self.size_entry.get())
        except Exceptions as ex:
            print(ex)


App()
