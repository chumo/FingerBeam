#  https://gist.github.com/TravisJoe/5576258
from Tkinter import *

root = Tk()

#w = Label(root, text="Hello, world!")
root.overrideredirect(True)
root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
root.focus_set()  # <-- move focus to this widget
root.bind("q", lambda e: e.widget.quit())
#w.pack()

root.mainloop()
