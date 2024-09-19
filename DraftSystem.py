import pickle
import PlayerTeam as playerTeam
import Pitcher as pitcher
import BaseBallPlayer as baseBallPlayer
from operator import attrgetter


class DraftSystem:
    def __init__(self):
        # initialize empty lists for teams
        self.teamA = playerTeam.PlayerTeam('A')
        self.teamB = playerTeam.PlayerTeam('B')
        self.teamC = playerTeam.PlayerTeam('C')
        self.teamD = playerTeam.PlayerTeam('D')

        self.current_team_picking = self.teamA

        self.pitchers = []
        self.baseball_players = []

        # Load player data from files
        try:
            with open("PlayerList.txt", "r") as file:
                full_player_list = file.read()
            player_strings = full_player_list.split("(?<=[a-zA-Z]{3}\\d{2})")
            self.baseball_players = [baseBallPlayer.BaseBallPlayer(info) for info in player_strings]

            with open("PitcherList.txt", "r") as file:
                full_player_list = file.read()
            pitcher_strings = full_player_list.split("(?<=[a-zA-Z]{3}\\d{2})")
            self.pitchers = [pitcher.Pitcher(info) for info in pitcher_strings]

            # Sort players by 'bA'
            self.eval_fun("bA")
        except FileNotFoundError as e:
            print("Error creating Draft")
            print(e)

    def i_draft(self, last_name, first_initial):
        player_found = False

        for player in self.baseball_players:
            if (last_name.lower() == player.last_name.lower()
                    and first_initial.upper() == player.first_name[0].upper()):
                if player.is_drafted:
                    print("Player already drafted")
                else:
                    player.drafted = True
                    self.teamA.add_to_team(player)
                    print(f"{player.last_name} has been drafted to team A")
                player_found = True
                break

        if not player_found:
            for player in self.pitchers:
                if (last_name.lower() == player.last_name.lower()
                        and first_initial.upper() == player.first_name[0].upper()):
                    if player.is_drafted:
                        print("Player already drafted")
                    else:
                        self.teamA.add_to_team(player)
                        print(f"{player.last_name} has been drafted to team A")
                    player_found = True
                    break

        if self.teamA.get_pitcher_amount() == 5:
            print("Team A has max pitchers")

        if not player_found:
            print("Could not find player")

    def o_draft(self, last_name, first_initial, team_name):
        player_found = False
        player_drafted = False
        temp_team_variable = None

        if team_name.upper() == 'A':
            temp_team_variable = self.teamA
        elif team_name.upper() == 'B':
            temp_team_variable = self.teamB
        elif team_name.upper() == 'C':
            temp_team_variable = self.teamC
        elif team_name.upper() == 'D':
            temp_team_variable = self.teamD

        if temp_team_variable != self.current_team_picking:
            print(f"It is team {self.current_team_picking.team_name}'s turn to pick")
        else:
            for player in self.baseball_players:
                if (last_name.lower() == player.last_name.lower()
                        and first_initial.upper() == player.first_name[0].upper()):
                    if player.is_drafted:
                        print("Player already drafted")
                    elif temp_team_variable.is_position_in_team(player.position):
                        print("Position already in team -- Not drafted")
                    else:
                        player_drafted = True
                        temp_team_variable.add_to_team(player)
                        print(f"{player.last_name} has been drafted to team {temp_team_variable.team_name}")
                    player_found = True
                    break

            if not player_found and temp_team_variable.get_pitcher_amount() < 5:
                for player in self.pitchers:
                    if (last_name.lower() == player.last_name.lower()
                            and first_initial.upper() == player.first_name[0].upper()):
                        if player.is_drafted:
                            print("Player already drafted")
                        else:
                            player_drafted = True
                            temp_team_variable.add_to_team(player)
                            print(f"{player.last_name} has been drafted to team {temp_team_variable.team_name}")
                        player_found = True
                        break

            if temp_team_variable.get_pitcher_amount() == 5:
                print(f"Team {temp_team_variable.team_name} has max pitchers")

            if not player_found:
                print("Couldn't find player")

            if player_drafted:
                self.current_team_picking = {
                    'A': self.teamB,
                    'B': self.teamC,
                    'C': self.teamD,
                    'D': self.teamA
                }[temp_team_variable.team_name]

    def team(self, team_name):
        if team_name.upper() == 'A':
            self.teamA.print_team()
        elif team_name.upper() == 'B':
            self.teamB.print_team()
        elif team_name.upper() == 'C':
            self.teamC.print_team()
        elif team_name.upper() == 'D':
            self.teamD.print_team()

    def stars(self, team_name):
        if team_name.upper() == 'A':
            self.teamA.print_stars()
        elif team_name.upper() == 'B':
            self.teamB.print_stars()
        elif team_name.upper() == 'C':
            self.teamC.print_stars()
        elif team_name.upper() == 'D':
            self.teamD.print_stars()

    def overall(self, position):
        if not position:
            for player in self.baseball_players:
                if not player.is_drafted and not self.current_team_picking.is_position_in_team(player.position):
                    print(player)
        else:
            for player in self.baseball_players:
                if position.lower() == player.position.lower() and not player.is_drafted:
                    print(player)

    def p_overall(self):
        if self.current_team_picking.get_pitcher_amount() < 5:
            count = 0
            for pitcher in sorted(self.pitchers, key=attrgetter('SO'), reverse=True):
                if count == 50:
                    break
                if not pitcher.is_drafted:
                    print(pitcher)
                count += 1
        else:
            print(f"Team {self.current_team_picking.team_name} has max pitchers")

    def eval_fun(self, eval_expression):
        try:
            self.baseball_players.sort(key=lambda p: p.evaluate(eval_expression), reverse=True)
        except Exception as e:
            print("Invalid expression -- setting to bA")
            self.baseball_players.sort(key=attrgetter('bA'), reverse=True)
            print(e)

    def p_eval_fun(self, eval_expression):
        try:
            self.pitchers.sort(key=lambda p: p.evaluate(eval_expression), reverse=True)
        except Exception as e:
            print("Invalid expression -- setting to SO")
            self.pitchers.sort(key=attrgetter('SO'), reverse=True)
            print(e)
