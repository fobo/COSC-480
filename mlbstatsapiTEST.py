import mlbstatsapi
mlb  = mlbstatsapi.Mlb()
lions_id = mlb.get_team_id("New York Yankees")[0]
lions = mlb.get_team_roster(lions_id)

stats = ['expectedstatistics']
section = ['hitting']
params = {'season':2023}
for x in range(len(lions)):
    print("Player ID: ", lions[x].id)
    try:
        stats = mlb.get_player_stats(lions[x].id,['expectedstatistics'], ['hitting'],**{'season':2019})
        expectedstats = stats['hitting']['expectedstatistics']
        for split in expectedstats.splits:
            for k, v in split.stat.__dict__.items():
                print(k,v)
        print()
    except:
        print("Stats not found\n")
