import tkinter as tk
from tkinter import filedialog, colorchooser

from PIL import ImageTk, Image, ImageDraw


class App:

    def __init__(self):
        self.color = '#FF0000'
        self.figure = 'rectangle'
        self.size = 5
        self.bg_last_x = None
        self.bg_last_y = None
        self.fg_last_x = None
        self.fg_last_y = None

        self.window = tk.Tk()
        self.window.after(100, self.selecting_file)
        self.window.mainloop()

    def selecting_file(self):
        self.file_path = filedialog.askopenfilename()
        # self.file_path = filedialog.askopenfilename(initialdir='images/', initialfile='example.jpg')

        self.image = Image.open(self.file_path)
        self.width, self.height = self.image.size

        self.draw = ImageDraw.Draw(self.image)
        self.photo = ImageTk.PhotoImage(image=self.image)

        self.frame_tools = tk.Frame(self.window)
        self.frame_tools.pack()

        self.background_button = tk.Button(self.frame_tools, text='background',
                                          command=lambda: self.change_figure('background'))
        self.background_button.pack(side='left')
        self.foreground_button = tk.Button(self.frame_tools, text='foreground', command=lambda: self.change_figure('foreground'))
        self.foreground_button.pack(side='left')
        self.size_entry = tk.Entry(self.frame_tools)
        self.size_entry.bind('<Return>', self.change_size)
        self.size_entry.pack(side='left')

        self.color_button = tk.Button(self.frame_tools, text='Color: #FF0000', command=self.change_color)
        self.color_button.pack(side='left')

        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()
        # self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind("<Button-1>", self.get_x_and_y)
        self.canvas.bind("<B1-Motion>", self.draw_smth)

        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.save_button = tk.Button(self.window, text='Save', command=self.save)
        self.save_button.pack()

    def on_click(self, event):
        if self.figure == 'background':
            self.draw_background(event)
        elif self.figure == 'foreground':
            self.draw_foreground(event)

    def draw_foreground(self, event):
        if self.fg_last_x:
            self.canvas.create_line((self.fg_last_x, self.fg_last_y, event.x, event.y), fill='red', width=10)
        self.fg_last_x = event.x
        self.fg_last_y = event.y

    def get_x_and_y(self, event):
        self.fg_lastx, self.fg_lasty = event.x, event.y

    def draw_smth(self, event):
        self.canvas.create_line((self.fg_lastx, self.fg_lasty, event.x, event.y),
                           fill='red',
                           width=2)
        self.fg_lastx, self.fg_lasty = event.x, event.y

    def draw_background(self, event):
        if self.bg_last_x:
            self.canvas.create_line((self.bg_last_x, self.bg_last_y, event.x, event.y), fill='blue', width=10)
        self.bg_last_x = event.x
        self.bg_last_y = event.y

    def save(self):
        self.image.save('output.jpg')

    def change_color(self):
        rgb, color_string = colorchooser.askcolor(initialcolor=self.color)
        # print(rgb, color_string)
        if color_string:
            self.color = color_string
            self.color_button['text'] = 'Color: ' + color_string

    def change_figure(self, figure):
        self.figure = figure

    def change_size(self, event=None):
        try:
            self.size = int(self.size_entry.get())
        except Exceptions as ex:
            print(ex)

App()