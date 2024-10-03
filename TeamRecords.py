import statsapi
import TeamID
import json

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
    opponent_dict = {}                          #create empty dict to update
    start_day, end_day = getDates(seasonYear)   #create date strings from season year
    for opponent in id.teams:                   #iterating through all teams in teams list
        opponentID = id.teams[opponent]         #getting ID of opponent for this iteration
        if(id.teams[opponent] != teamID):           #ignoring 
            games = statsapi.schedule(start_date=start_day,end_date=end_day,team=teamID,opponent=opponentID)
            scoreArray = analyzeGames(games, teamID, opponentID)
            if (scoreArray == [0,0]):   #if scores for both teams is 0 then no need to store record data
                continue
            opponent_dict.update({opponentID: scoreArray})
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
#team, teamID = getTeams()

# generates dict for every team in season
# maybe we write these to a big json file to grab from as this is quite slow
seasonYear = '2015'
for team in id.teams:
    dict = CreateRecordsDict(id.teams[team], seasonYear)
    json_object=json.dumps(dict, indent=3)
    with open(f"{seasonYear}_{id.teams[team]}.json", "w") as outfile:
        outfile.write(json_object)
    print(dict)

