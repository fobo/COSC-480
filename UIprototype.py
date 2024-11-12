import tkinter as tk
from tkinter import ttk
import mlbstatsapi

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
        label.grid(row=0, column=0, pady=10, padx=20, sticky='w')

        # Frame to contain the entry box and its label
        entry_frame = tk.Frame(self, bg='#FFFACD')
        entry_frame.grid(row=1, column=0, pady=10, sticky='w')

        # Label for entry box title
        entry_label = tk.Label(entry_frame, text="Enter Season Year:", font=('arial', 12), bg='#FFFACD')
        entry_label.pack(anchor='w', padx=20)

        # Entry box for user input
        self.entry_box = tk.Entry(entry_frame, font=('arial', 12))
        self.entry_box.pack(padx=20, pady=5)

        # Button to save entry box value to a variable
        save_button = tk.Button(self, text="Load Players", bg='#FAEBD7', font=('arial', 12),
                                command=self.save_entry_value)
        save_button.grid(row=2, column=0, pady=10, sticky='w', padx=20)

        # Back button
        back_button = tk.Button(self, text="Back to Home", bg='#FAEBD7', font=('arial', 12), height=1,
                                command=lambda: controller.show_frame("HomePage"))
        back_button.grid(row=3, column=1, pady=10, padx=10, sticky='se')

        # Configure grid columns for layout consistency
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def save_entry_value(self):
        
        season_year = self.entry_box.get()
        print("Player Name:", season_year)
        teams = [
            "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox", 
            "Chicago White Sox", "Chicago Cubs", "Cincinnati Reds", "Cleveland Guardians", 
            "Colorado Rockies", "Detroit Tigers", "Houston Astros", "Kansas City Royals", 
            "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins", "Milwaukee Brewers", 
            "Minnesota Twins", "New York Yankees", "New York Mets", "Oakland Athletics", 
            "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants", 
            "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays", "Texas Rangers", 
            "Toronto Blue Jays", "Washington Nationals"
        ]
        for z in range(len(teams)):
            mlb  = mlbstatsapi.Mlb()
            lions_id = mlb.get_team_id(teams[z]) #select team name
            lions = mlb.get_team_roster(lions_id)       #grabs all player
            returnString = ""
            batAVG = ""
            homeRuns = ""
            RBIs = ""
            score = 0
            stats = ['season']
            section = ['hitting']
            params = {'season':2022}
            for x in range(len(lions)):
               returnString += lions[x].fullname + ", "
               try:
                    stats = mlb.get_player_stats(lions[x].id,['season'], ['hitting'],**{'season':season_year})
                    expectedstats = stats['hitting']['season']
                    for split in expectedstats.splits:
                        for k, v in split.stat.__dict__.items():
             #               print(k,v)

                            if(k == "doubles"):
                                score += 2 * v
                            elif(k == "triples"):
                                 score += 3  * v
                            elif(k == "homeruns"):
                                score += 4 * v
                                homeruns = str(v)
                            elif(k == "runs"):
                                score += 1 * v
                            elif(k == "rbi"):
                                score += 1 * v
                                RBIs = str(v)
                            elif(k == "baseonballs"):
                                score += 1 * v
                            elif(k == "hitbypitch"):
                                score += 1 * v
                            elif(k == "stolenbases"):
                                score += 2 * v
                            elif(k == "caugtstealing"):
                                score -= 1 * v
                            elif(k == "avg"):
                                batAVG = str(v)
                    returnString += batAVG + ", "
                    returnString += homeruns + ", "
                    returnString += RBIs + ", "
                    returnString += str(score)
                    if(RBIs == "0" or batAVG == ".000" or score == 0):
                        break
                    else:
                        print(returnString)
                    returnString = ""
                    score = 0
               except:
                    returnString = ""
                    score = 0
        #        print("Stats not found\n")




        
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
