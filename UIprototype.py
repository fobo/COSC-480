import tkinter as tk
from tkinter import ttk

# Function called when the button is clicked
def btnClickFunction():
    print('clicked')

root = tk.Tk()

# Configure the main window
root.geometry('880x540')
root.configure(background='#F0F8FF')
root.title('Hello, I\'m the main window')

# Configure the rows and columns for grid layout
root.grid_rowconfigure(0, minsize=40)  # Set row height for the label
root.grid_rowconfigure(1, minsize=40)  # Set row height to ensure buttons don't stretch
root.grid_rowconfigure(2, weight=1)  # Added empty row for spacing
for i in range(4):
    root.grid_columnconfigure(i, weight=1)

# Create the label and add it to the grid centered in the middle two columns, with a height of 30 pixels
label = tk.Label(root, text='My Team', bg='#F0F8FF', font=('arial', 12, 'normal'), height=2)
label.grid(row=0, column=1, columnspan=2, pady=5, sticky='n')

# Create the buttons and add them to the grid with fixed height
buttons = ['Roster', 'Matchup', 'Players', 'League']
for i, text in enumerate(buttons):
    button = tk.Button(root, text=text, bg='#FAEBD7', font=('arial', 12, 'normal'), command=btnClickFunction, height=1)
    button.grid(row=1, column=i, sticky='nsew', padx=5, pady=5)

root.mainloop()
