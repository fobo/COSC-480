import statsapi
import TeamID

id = TeamID.teamId()

# In future implementation this should be a selectable date year
# Given range **should** include all games in season entered
def getDates(year):
    start_day,end_day = "", ""
    start_day = "03/12/" + year
    end_day = "11/7/" + year
    return start_day, end_day

# Collects team name/opponent from user, will keep trying until a valid name is entered
# Not case sensitive but spelling must be correct
# In the future we could use enums/preset strings for team selection in our GUI so typing wont be necessary
def getTeams():
    teamID = 0
    while True:
        team = input("Enter a name for the first team: ")
        teamID = id.getTeamId(team.lower())
        if(teamID == "Not Found"):
            continue
        break
    return team, teamID

def analyzeGames(games, teamID, opponentID):
    teamWins, oppWins = 0,0
    for game in games:
        try:
            if(id.teams[game['winning_team'].lower()] == teamID):
                teamWins+=1
            elif(id.teams[game['winning_team'].lower()] == opponentID):
                oppWins+=1
        except KeyError as e:
            continue
    return [teamWins, oppWins]

def CreateOpponentDict(teamID, seasonYear):
    opponent_dict = {}
    start_day, end_day = getDates(seasonYear)
    for opponent in id.teams:
        opponentID = id.teams[opponent]
        if(id.teams[team] == teamID):
            games = statsapi.schedule(start_date=start_day,end_date=end_day,team=teamID,opponent=opponentID)
            opponent_dict.update({opponentID: analyzeGames(games, teamID, opponentID)})
    return opponent_dict

def CreateRecordsDict(teamID, seasonYear):
    records_dict = {}
    print(teamID)
    records_dict.update({seasonYear: {teamID: CreateOpponentDict(teamID, seasonYear)}})
    return records_dict

# ex: {2008: {      
#   132: {108: [5,1] },  teamID: {opponentID: [teamWins, opponentWins]}
#        {109: [3,2] }...

#   133: {108: [3,5] },
#        {109: [2,1] }...
# ...
#
#   Note: If teamWins and opponentWins are 0,0 then there are no games played that season (Or no data for some reason)
#   Also if more data is needed games data has a lot more available


# Refer to teamId.py for team key/value pairs 
team, teamID = getTeams()


dict = CreateRecordsDict(teamID, '2019')

print(dict)
