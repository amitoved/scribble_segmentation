import os
import tkinter as tk
from tkinter import colorchooser

import numpy as np
from PIL import ImageTk, Image, ImageDraw

import constants


class App:

    def __init__(self):
        self.color = '#FF0000'
        self.figure = 'rectangle'
        self.size = 5
        self.class_val = None
        self.annotations = {val: [] for val in constants.classes.values()}
        self.window = tk.Tk()
        self.pool_folder = os.path.join(constants.DATA_DIR, 'pool')
        self.selecting_file()
        # self.window.after(100, self.selecting_file)
        self.window.mainloop()

    def selecting_file(self, update=False):

        image_paths = [os.path.join(self.pool_folder, file) for file in os.listdir(self.pool_folder) if 'image' in file]
        self.image_path = np.random.choice(image_paths)
        self.scribble_path = self.image_path.replace('image', 'scribble')
        self.pred_path = self.image_path.replace('image', 'pred')

        # self.image_path = filedialog.askopenfilename()
        # self.file_path = filedialog.askopenfilename(initialdir='images/', initialfile='example.jpg')

        self.image = Image.fromarray(np.load(self.image_path))
        self.width, self.height = self.image.size
        self.annotation_img = Image.new('RGB', (self.width, self.height))

        self.draw = ImageDraw.Draw(self.image)
        self.photo = ImageTk.PhotoImage(image=self.image)

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
        # self.last_x.append(event.x)
        # self.last_y.append(event.y)

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
        img1 = ImageDraw.Draw(self.annotation_img)
        img1.line((self.last_x, self.last_y, event.x, event.y), fill=color, width=0)
        # self.annotation_img[event.y, event.x, self.class_val] = 1

    def update_scribble(self):
        scribble = np.load(self.scribble_path)
        for key, val in self.annotations.items():
            val = np.array(val, dtype=np.uint8)
            scribble[val[:, 1], val[:, 0]] = key
        return scribble

    def save(self):
        scribble = self.update_scribble()
        np.save(self.scribble_path, scribble)
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

    def change_class(self, val):
        self.class_val = val

    def change_size(self, event=None):
        try:
            self.size = int(self.size_entry.get())
        except Exceptions as ex:
            print(ex)


App()
