import mlbstatsapi
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
for x in range(len(teams)):
    mlb  = mlbstatsapi.Mlb()
    lions_id = mlb.get_team_id(teams[x])[0] #select team name
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
            stats = mlb.get_player_stats(lions[x].id,['season'], ['hitting'],**{'season':2023})
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
                returnString = ""
                score = 0
            else:       
                print(returnString)
                returnString = ""
                score = 0
       except:
            returnString = ""
            score = 0
    #        print("Stats not found\n")

