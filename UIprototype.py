import tkinter as tk
from tkinter import ttk

class App:
    def __init__(self, root):
        self.root = root
        self.root.geometry('880x540')
        self.root.configure(background='#F0F8FF')
        self.root.title('Main Window')

        self.container = tk.Frame(self.root)
        self.container.grid(row=1, column=0, columnspan=4, sticky='nsew')

        self.root.grid_rowconfigure(1, weight=1)
        for i in range(4):
            self.root.grid_columnconfigure(i, weight=1)

        self.frames = {}
        for F in (HomePage, RosterPage, MatchupPage, PlayersPage, LeaguePage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.show_frame("HomePage")

        buttons = ['Roster', 'Matchup', 'Players', 'League']
        for i, text in enumerate(buttons):
            button = tk.Button(self.root, text=text, bg='#FAEBD7', font=('arial', 12, 'normal'),
                               command=lambda t=text: self.show_frame(t + "Page"), height=1)
            button.grid(row=0, column=i, sticky='nsew', padx=5, pady=5)

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#ADD8E6')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        label = tk.Label(self, text="Home Page", font=('arial', 16, 'bold'), bg='#ADD8E6')
        label.grid(row=0, column=0, pady=10, sticky='nsew')

class RosterPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#90EE90')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        label = tk.Label(self, text="Roster Page", font=('arial', 16, 'bold'), bg='#90EE90')
        label.grid(row=0, column=0, pady=10, sticky='nsew')

        back_button = tk.Button(self, text="Back to Home", bg='#FAEBD7', font=('arial', 12, 'normal'), height=1,
                                command=lambda: controller.show_frame("HomePage"))
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)
        back_button.grid(row=2, column=1, pady=10, padx=10, sticky='se')

class MatchupPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#FFB6C1')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        label = tk.Label(self, text="Matchup Page", font=('arial', 16, 'bold'), bg='#FFB6C1')
        label.grid(row=0, column=0, pady=10, sticky='nsew')

        back_button = tk.Button(self, text="Back to Home", bg='#FAEBD7', font=('arial', 12, 'normal'), height=1,
                                command=lambda: controller.show_frame("HomePage"))
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)
        back_button.grid(row=2, column=1, pady=10, padx=10, sticky='se')

class PlayersPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#FFFACD')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        label = tk.Label(self, text="Players Page", font=('arial', 16, 'bold'), bg='#FFFACD')
        label.grid(row=0, column=0, pady=10, sticky='nsew')

        back_button = tk.Button(self, text="Back to Home", bg='#FAEBD7', font=('arial', 12, 'normal'), height=1,
                                command=lambda: controller.show_frame("HomePage"))
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)
        back_button.grid(row=2, column=1, pady=10, padx=10, sticky='se')

class LeaguePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg='#D3D3D3')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        label = tk.Label(self, text="League Page", font=('arial', 16, 'bold'), bg='#D3D3D3')
        label.grid(row=0, column=0, pady=10, sticky='nsew')

        back_button = tk.Button(self, text="Back to Home", bg='#FAEBD7', font=('arial', 12, 'normal'), height=1,
                                command=lambda: controller.show_frame("HomePage"))
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(1, weight=1)
        back_button.grid(row=2, column=1, pady=10, padx=10, sticky='se')

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
