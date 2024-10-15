import statsapi
import csv
# Open the CSV file
# that contains the player IDs
# found online https://razzball.com/mlbamids/

playerIds = []

with open('razzball.csv', 'r') as file:
    reader = csv.DictReader(file)  # Use DictReader to access columns by name
    playerIds = [row['MLBAMID'] for row in reader]


class Player():    
    def __init__(self, data):
        self.name = data['first_name'] + " " + data['last_name']
        self.batting_avg = self.get_batting_avg(data['stats'])
        self.homeruns = self.get_home_runs(data['stats'])
        self.RBI = self.get_rbis(data['stats'])
        self.fantasy_points = self.fantasy_points(data['stats'])
        self.dict = {"player_name": self.name, "batting_avg": self.batting_avg,
                     "home_runs": self.homeruns, "RBIs": self.RBI, "fantasy_points": self.fantasy_points}
    
    def fantasy_points(self, stats):
        total_points = 0
        if not stats:
            return total_points  # Return 0 if no stats are available
        # Batting Points
        batting_stats = stats[0]['stats']  # Assuming first entry is batting
        total_points += batting_stats.get('hits', 0) * 1  # Singles
        total_points += batting_stats.get('doubles', 0) * 2  # Doubles
        total_points += batting_stats.get('triples', 0) * 3  # Triples
        total_points += batting_stats.get('homeRuns', 0) * 4  # Home Runs
        total_points += batting_stats.get('runs', 0) * 1  # Runs
        total_points += batting_stats.get('rbi', 0) * 1  # RBIs
        total_points += batting_stats.get('baseOnBalls', 0) * 1  # Walks
        total_points += batting_stats.get('hitByPitch', 0) * 1  # Hit By Pitch
        total_points += batting_stats.get('stolenBases', 0) * 2  # Stolen Bases
        total_points -= batting_stats.get('caughtStealing', 0) * -1  # Caught Stealing

        # Pitching Points
        pitching_stats = stats[0]  # Assuming first entry is pitching
        if pitching_stats['type'] == 'season' and pitching_stats['group'] == 'pitching':
            pitching_data = pitching_stats['stats']
            total_points += pitching_data.get('wins', 0) * 4  # Wins
            total_points += pitching_data.get('saves', 0) * 2  # Saves
            total_points += float(pitching_data.get('inningsPitched', 0)) * 1  # Innings Pitched
            total_points -= pitching_data.get('earnedRuns', 0) * -1  # Earned Runs Allowed

        return total_points

    def get_batting_avg(self, stats):
        # Check if there's at least one stats entry and return the 'avg'
        if stats and 'stats' in stats[0] and 'avg' in stats[0]['stats']:
            return stats[0]['stats']['avg']
        else:
            return None  # Return None or a default value if 'avg' is not found
        
    def get_home_runs(self, stats):
        # Check if there's at least one stats entry and return the 'homeRuns'
        if stats and 'stats' in stats[0] and 'homeRuns' in stats[0]['stats']:
            return stats[0]['stats']['homeRuns']
        else:
            return None  # Return None or a default value if 'homeRuns' is not found
            
    def get_rbis(self, stats):
    # Check if there's at least one stats entry and return the 'rbi'
        if stats and 'stats' in stats[0] and 'rbi' in stats[0]['stats']:
            return stats[0]['stats']['rbi']
        else:
            return None  # Return None or a default value if 'rbi' is not found

#   Writes to csv file all of the non zero records of players.
#   In the future could alter to this and the Player class to
#   differentiate between pitching and batting independently?
for player in playerIds:
    player_data = (statsapi.player_stat_data(player, group="[hitting,pitching,fielding]", type="season", sportId=1))
    #   Create player object
    formated = Player(player_data)
    if(formated.batting_avg != None and formated.RBI != None
       and formated.batting_avg != '0.000' and formated.RBI != 0):  
        player_data = formated.dict 
        with open('out.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=player_data.keys())
            # Write the single row (dictionary)
            writer.writerow(player_data)

