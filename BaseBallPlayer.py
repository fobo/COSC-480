import re
from pyjexl import JEXL

class BaseBallPlayer:
    def __init__(self, player_info):
        self.first_name = ''
        self.last_name = ''
        self.is_drafted = False
        self.player_position = ''
        self.team_name = ''
        self.bA = 0.0
        self.hR = 0.0
        self.r = 0.0
        self.h = 0.0
        self.sO = 0.0
        self.eval_result = 0.0
        
        # Split player_info and handle empty stats
        info = player_info.split(",")
        info = [i if i.strip() != '' else '0' for i in info]

        # Process the name
        name_split = re.split(r'(?=(?<!^)[A-Z])', info[1])
        self.first_name = name_split[0] + name_split[1] if len(name_split) == 3 else name_split[0]
        self.last_name = name_split[-1]

        # Remove any non-letter from the last name (e.g., # or *)
        self.last_name = re.sub(r'[^a-zA-Z]', '', self.last_name)

        # Assign player stats
        self.r = float(info[7])
        self.h = float(info[8])
        self.hR = float(info[11])
        self.bA = float(info[17])
        self.eval_result = self.bA
        self.sO = float(info[16])
        self.team_name = info[3]
        
        # Find player position and correct if needed
        self.player_position = info[28]
        for char in self.player_position:
            if char.isdigit():
                self.find_position(char)
                break

    def __str__(self):
        return f"{self.first_name} {self.last_name} {self.team_name} {self.player_position} {self.eval_result}"

    def find_position(self, pos_num):
        positions = {
            '1': 'P', '2': 'C', '3': '1B', '4': '2B',
            '5': '3B', '6': 'SS', '7': 'LF', '8': 'CF', '9': 'RF'
        }
        self.player_position = positions.get(pos_num, '')

    def evaluate(self, expression):
        try:
            jexl = JEXL()
            context = {
                "bA": self.bA,
                "hR": self.hR,
                "r": self.r,
                "h": self.h,
                "sO": self.sO
            }
            result = jexl.evaluate(expression, context)
            self.eval_result = result
            return result
        except Exception as e:
            raise e

    def set_is_drafted(self):
        self.is_drafted = True

    def get_ba(self):
        return self.bA

    def get_h(self):
        return self.h

    def get_r(self):
        return self.r

    def get_hr(self):
        return self.hR

    def get_so(self):
        return self.sO

    def get_position(self):
        return self.player_position

    def temp_player(self):
        return self.first_name

    def get_last_name(self):
        return self.last_name

    def get_is_drafted(self):
        return self.is_drafted

    def get_first_name(self):
        return self.first_name
