import sentimentAnalysis as sa

import tkinter as tk
from tkinter import scrolledtext
from tkinter.ttk import Frame, Label, Entry

from PIL import Image, ImageTk

count_pos = 0
count_neg = 0
total = 0

LARGE_FONT = ("Verdana", 12)

class MovieReview(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)
        self.geometry("800x600")

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, InputReview):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Home", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        self.showImg()

        frame1 = Frame(self, width=150, height=175)
        frame1.pack()

        button1 = tk.Button(frame1, text="Input Review", command=lambda: controller.show_frame(InputReview))
        button1.pack(anchor='n', side='left', padx=5, pady=5)

    def showImg(self):
        canvas = tk.Canvas(self, width=300, height=350)
        canvas.pack()

        load = Image.open("Avengers.jpg")
        img = ImageTk.PhotoImage(load)
        canvas.create_image(150, 175, anchor='center', image=img)
        canvas.image = img


class InputReview(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Input Review", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        input_name = tk.StringVar()

        frame1 = Frame(self)
        frame1.pack(anchor='nw')

        lbl1 = Label(frame1, text="Name", width=8)
        lbl1.pack(side='left', padx=5, pady=5)

        entry1 = Entry(frame1, textvariable=input_name, width=50)
        entry1.pack(side='left', padx=5, expand=True)

        frame2 = Frame(self)
        frame2.pack(anchor='nw', expand=True)

        lbl2 = Label(frame2, text="Review", width=8)
        lbl2.pack(side='left', anchor='nw', padx=5, pady=5)

        txt = tk.Text(frame2, width=50, height=20)
        txt.pack(side='left', pady=5, padx=5, expand=True)

        button1 = tk.Button(frame2, text="Confirm", command=lambda: self.predictReview(input_name, txt.get(1.0, "end-1c"), entry1, txt, frame3, text))
        button1.pack(anchor='n', side='top', padx=5, pady=5)

        button2 = tk.Button(frame2, text="Home", command=lambda: controller.show_frame(StartPage))
        button2.pack(anchor='s', side='bottom', padx=5, pady=5)

        frame3 = Frame(self)
        frame3.pack(anchor='nw', expand=True)

        text = tk.scrolledtext.ScrolledText(frame3, height=10, width=70, wrap='word')

    def predictReview(self, _name, _review, entry1, txt, frame3, text):
        result = ''

        input_name = _name.get()
        input_review = _review

        id, conf = sa.sentiment(input_review)

        if id == 'pos':
            result = id

        elif id == 'neg':
            result = id

        name = input_name
        review = input_review

        entry1.delete(0, 'end')
        txt.delete(1.0, 'end-1c')

        lbl3 = Label(frame3, text='', width=8)
        lbl3.pack(side='left', anchor='nw', padx=5, pady=5)

        n = '\n'
        if result == 'pos':
            txt = name, ':', review, "Positive Review", n
            text.insert(1.0, txt)
        if result == 'neg':
            txt = name, ':', review, "Negative Review", n
            text.insert(1.0, txt)

        text.pack(side='left', pady=5, padx=5, expand=True)

        tk.messagebox.showinfo('Message', 'Review Added')

app = MovieReview()
app.mainloop()
