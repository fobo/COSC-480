class PlayerTeam:
    def __init__(self, team_name):
        self.team_name = team_name
        self.player_pick_order = []
        self.pitchers = []
        self.players = {}
        self.amount_of_pitchers = 0
        self.is_pos_in_team = {
            "C": False, "1B": False, "2B": False, "3B": False,
            "SS": False, "LF": False, "RF": False, "CF": False
        }

    def add_to_team(self, player):
        if isinstance(player, BaseBallPlayer):
            if not self.is_position_in_team(player.get_position()):
                self.players[player.get_position()] = player
                self.player_pick_order.append(player)
                player.set_is_drafted()
                self.is_pos_in_team[player.get_position()] = True
        elif isinstance(player, Pitcher):
            if self.amount_of_pitchers == 5:
                print(f"Player {self.team_name} has five pitchers already, pitcher not added")
            else:
                self.pitchers.append(player)
                self.player_pick_order.append(player)
                player.set_is_drafted()
                self.amount_of_pitchers += 1

    def is_position_in_team(self, pos):
        if "D" in pos:
            return True
        elif pos == "P":
            return self.amount_of_pitchers == 5
        return self.is_pos_in_team.get(pos, False)

    def print_team(self):
        print("----------------------------------")
        print(f"Team {self.team_name}")
        print()
        
        positions = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        for pos in positions:
            temp_player = self.players.get(pos)
            if temp_player:
                print(f"{temp_player.get_position()}   {temp_player.get_first_name()} {temp_player.get_last_name()}")

        # Print pitchers
        for pitcher in self.pitchers:
            print(f"{pitcher.get_position()}   {pitcher.get_first_name()} {pitcher.get_last_name()}")
        print("----------------------------------")
        print()

    def print_stars(self):
        print("------------------------------")
        print(f"Team {self.team_name} STARS")
        print()
        for player in self.player_pick_order:
            if isinstance(player, BaseBallPlayer):
                print(f"{player.get_position()}   {player.get_first_name()} {player.get_last_name()}")
            elif isinstance(player, Pitcher):
                print(f"{player.get_position()}   {player.get_first_name()} {player.get_last_name()}")
        print("------------------------------")
        print()

    def get_team_name(self):
        return self.team_name

    def get_pitcher_amount(self):
        return self.amount_of_pitchers
