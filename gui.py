import tkinter as tk
from tkinter import ttk
from pathlib import Path


home_path = str(Path.home())

class Window:
    "Class to create windows and widgets for the application"
    def __init__(self):
        "Initialize the window with default properties"
        self.title = "My Application"
        self.width = 800
        self.height = 600
        self.window=tk.Tk()
        # Let the window adapt to the content instead of a fixed size
        self.window.update_idletasks()
        self.window.minsize(self.window.winfo_reqwidth(), self.window.winfo_reqheight())
        self.window.resizable(True, True)
        # Make root grid cells expandable
        for r in (1, 2):
            self.window.grid_rowconfigure(r, weight=1)
        for c in (1, 2):
            self.window.grid_columnconfigure(c, weight=1)

    def insert_frame(self, row, column, text=None):
        "Create a frame and place it in the specified grid location"
        frame = tk.Frame(self.window)
        frame.grid(row=row, column=column, sticky="nsew")
        if text:
            lbf=ttk.Label(frame,text=text)
            lbf.grid(row=row,column=column)
        return frame
    
    def insert_subframe(self, parent, row, column, text=None, pady=0):
        "Create a subframe inside a parent frame and optionally add a title label"
        frame = tk.Frame(parent)
        frame.grid(row=row, column=column, pady=pady)
        if text:
            lbf = ttk.Label(frame, text=text)
            lbf.grid(row=0, column=1)
        return frame
    
    def insert_button(self,frame,row,column,text,command):
        "Create a button and place it in the specified grid location within the given frame"
        btn=tk.Button(frame,text=text,padx=10,pady=5,fg="white",bg="#263D42", command=command)
        btn.grid(row=row,column=column)
        return btn

    def insert_combobox(self, frame, row, column, values, width=29, state='readonly', default=None):
        "Create a combobox with provided values"
        cb = ttk.Combobox(frame, values=values, width=width, state=state)
        cb.grid(row=row, column=column)
        if default is not None:
            cb.set(default)
        return cb

    def insert_checkbutton(self, frame, row, column, text, command=None):
        "Create a checkbutton; returns the widget and its IntVar"
        ch_rec=tk.IntVar()
        ch1 = tk.Checkbutton(frame, command=command,variable=ch_rec,text=text)
        ch1.grid(row=row,column=column)
        return ch1, ch_rec
        

    def insert_entry(self, frame, row, column, w=5, state='disabled', text=None, command=None):
        "Create an entry widget with optional default text"
        if text and command:
            btn=tk.Button(frame,text=text,padx=5,pady=0,fg="white",bg="#263D42", command=command)
            btn.grid(row=row+1,column=column)
            entry = tk.Entry(frame,bd=5,width=w,state=state)
            entry.grid(row=row,column=column)
        else:
            lbl=tk.Label(frame,text=text)
            lbl.grid(row=row,column=column)
            entry = tk.Entry(frame,bd=5,width=w,state=state)
            entry.grid(row=row,column=column+1)
        return entry 

