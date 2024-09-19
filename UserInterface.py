import pickle
import DraftSystem as draftSystem
import tkinter as tk
class UserInterface:
    def __init__(self):
        self.fantasy_draft = draftSystem.DraftSystem()
        m = tk.Tk()
        m.mainloop()
        #self.show_prompt()

    def show_prompt(self):
        while True:
            user_input = input("Enter A Command: ").strip().lower()
            if user_input == "exit":
                self.exit()
            elif user_input == "help":
                self.show_commands()
            elif user_input == "idraft":
                last_name = input("Enter last name: ").strip()
                player_first_initial = input("Enter a character: ").strip()
                if player_first_initial:
                    self.fantasy_draft.i_draft(last_name, player_first_initial[0])
                else:
                    print("Initial was not provided.")
            elif user_input == "odraft":
                last_name2 = input("Enter last name: ").strip()
                initial_input = input("Enter the first initial: ").strip()
                if not initial_input:
                    print("First initial was not provided.")
                    continue
                first_initial = initial_input[0]
                team_name_input = input("Enter team name A, B, C, D: ").strip().upper()
                if team_name_input and team_name_input[0] in ['A', 'B', 'C', 'D']:
                    self.fantasy_draft.o_draft(last_name2, first_initial, team_name_input[0])
                else:
                    print("Invalid team name provided. Must be A, B, C, D.")
            elif user_input == "evalfun":
                print("Variables for non-pitchers: bA, hR, r, h, sO --- No Division")
                eval_expression = input("Enter for non-pitchers: ").strip()
                if eval_expression:
                    self.fantasy_draft.eval_fun(eval_expression)
                    print("Non-pitchers sorted successfully using the given expression.")
                else:
                    print("Expression was not provided.")
            elif user_input == "pevalfun":
                print("Variables for pitchers: sO, iP, h, r, hR --- No Division")
                eval_expression_for_pitchers = input("Enter for pitchers: ").strip()
                if eval_expression_for_pitchers:
                    self.fantasy_draft.p_eval_fun(eval_expression_for_pitchers)
                    print("Pitchers sorted successfully")
                else:
                    print("Expression was not provided.")
            elif user_input == "overall":
                pos_input = input("Enter player position or leave blank: ").strip()
                self.fantasy_draft.overall(pos_input)
            elif user_input == "poverall":
                self.fantasy_draft.p_overall()
            elif user_input == "stars":
                team_char1 = input("Enter a team letter: ").strip()[0]
                self.fantasy_draft.stars(team_char1)
            elif user_input == "team":
                team_char2 = input("Enter a team letter: ").strip()[0]
                self.fantasy_draft.team(team_char2)
            elif user_input == "save":
                self.save()
            elif user_input == "restore":
                self.restore()
            else:
                print('Invalid Command: Type "help" to see possible commands')

    def save(self):
        file_name = input("Enter name of file to save Draft to: ").strip()
        try:
            with open(file_name, 'wb') as file:
                pickle.dump(self.fantasy_draft, file)
            print(f"Draft has been saved to {file_name}")
        except Exception as e:
            print("Error while saving Draft:", str(e))

    def restore(self):
        file_name = input("Enter name of file collection is saved at: ").strip()
        try:
            with open(file_name, 'rb') as file:
                self.fantasy_draft = pickle.load(file)
            print(f"fantasyDraft has been restored from {file_name}")
        except Exception as e:
            print("Error while restoring Draft:", str(e))

    def exit(self):
        # Ask user if they want to save first
        exit()

    def show_commands(self):
        print("Available Commands: exit, help, overall, poverall, stars, team, idraft, odraft, evalfun, pevalfun, save, restore")
