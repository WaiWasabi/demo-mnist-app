from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
import torch
from torchvision.transforms import *
from torch.nn import functional

from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


model = torch.jit.load('model_weights0.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train(False)


def predict_digit(img):
    transform = transforms.Compose([transforms.ToTensor()])

    #resize image to 28x28 pixels
    img = img.resize((28, 28))

    #convert rgb to grayscale
    img = np.invert(img.convert('L'))
    img = np.array(img)

    img = transform(Image.fromarray(img))

    imshow(img[0])
    plt.show()

    img = img.view(1, 1, 28, 28)

    #predicting the class
    res = functional.softmax(model(torch.FloatTensor(img).to(device))[0], dim=-1)
    return np.argmax(res.detach().cpu().numpy()), torch.max(res)


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit)+', ' + str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
