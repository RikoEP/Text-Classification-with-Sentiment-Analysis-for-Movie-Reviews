# import  sentimentAnalysis as sa
#
# str = input("input review: ")
#
# res, conf = sa.sentiment(str)
#
# print(res, conf)
#
# if res == 'pos':
#     print('good review')
# elif res == 'neg':
#     print('bad review')
# import tkinter
#
# def get_text():
#     content = body.get(1.0, "end-1c")
#     entry_content.set(content)
#     print(content)
# master = tkinter.Tk()
#
# body = tkinter.Text(master)
# body.pack()
#
# entry_content = tkinter.StringVar()
# entry = tkinter.Entry(master, textvariable=entry_content)
# entry.pack()
#
# button = tkinter.Button(master, text="Get tkinter.Text content", command=get_text)
# button.pack()
#
#
#
# master.mainloop()

x = []

class A:
    def func_1(self):
        #populate the x variable
        global x
        x.append(1)
class B:
    def func_2(self):
        global x
        print (x)

a = A()
a.func_1()
b = B()
b.func_2()