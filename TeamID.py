class teamId():
    def __init__(self):
        self.teams = {
    "los angeles angels": 108,
    "arizona diamondbacks": 109,
    "baltimore orioles": 110,
    "boston red sox": 111,
    "chicago cubs": 112,
    "cincinnati reds": 113,
    "cleveland indians": 114,
    "colorado rockies": 115,
    "detroit tigers": 116,
    "houston astros": 117,
    "kansas city royals": 118,
    "los angeles dodgers": 119,
    "washington nationals": 120,
    "new york mets": 121,
    "oakland athletics": 133,
    "pittsburgh pirates": 134,
    "san diego padres": 135,
    "seattle mariners": 136,
    "san francisco giants": 137,
    "st. louis cardinals": 138,
    "tampa bay rays": 139,
    "texas rangers": 140,
    "toronto blue jays": 141,
    "minnesota twins": 142,
    "philadelphia phillies": 143,
    "atlanta braves": 144,
    "chicago white sox": 145,
    "miami marlins": 146,
    "new york yankees": 147,
    "milwaukee brewers": 158
}
    def printTeams(self):
        for team in self.teams:
            print(f"",{team})

    def getTeamId(self,name):
        # Input Error Checking
        if name in self.teams:
            return self.teams[name]
        else:
            print("Team Not Found. Please Try Again")
        return "Not Found"
        
