import unittest

class DraftSystem:
    def overall(self, position):
        print(f"Overall for position: {position}")
        print("----------------------------------")

    def iDraft(self, player, code):
        print(f"iDraft called for player {player} with code {code}")

    def oDraft(self, player, code1, code2):
        print(f"oDraft called for player {player} with codes {code1} and {code2}")

    def stars(self, team_code):
        print(f"stars called for team {team_code}")

    def team(self, team_code):
        print(f"team called for {team_code}")

    def evalFun(self, expression):
        print(f"Evaluating expression: {expression}")

    def pOverall(self):
        print("pOverall called")

    def pEvalFun(self, expression):
        print(f"Evaluating pitcher expression: {expression}")

# Unit test cases
class DraftMethodTesting(unittest.TestCase):

    def test_overall_methods(self):
        dS = DraftSystem()
        
        positions = ["P", "C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
        for position in positions:
            dS.overall(position)
            print()
    
    def test_draft_methods(self):
        dS = DraftSystem()
        
        draft_players = [("Hamilton", 'C'), ("Bannon", 'R'), ("Gallo", 'J'), ("Gordon", 'N'),
                         ("Gausman", 'K'), ("Eflin", 'Z'), ("Gray", 'S'), ("Bibee", 'T'),
                         ("Abreu", 'B'), ("Gausman", 'K'), ("Brito", 'J'), ("Bannon", 'R')]
        
        for player, code in draft_players:
            dS.iDraft(player, code)

        o_draft_players = [("Cowser", 'C', 'A'), ("Bride", 'J', 'B'), ("Hamilton", 'D', 'B'),
                           ("Hummel", 'C', 'B'), ("Blanco", 'D', 'C'), ("Capel", 'C', 'C'),
                           ("Capel", 'C', 'D'), ("Dalbec", 'B', 'D')]
        
        for player, code1, code2 in o_draft_players:
            dS.oDraft(player, code1, code2)
        
        dS.stars('A')
        dS.stars('B')
        
        for team_code in ['A', 'B', 'C', 'D']:
            dS.team(team_code)

    def test_eval_methods(self):
        dS = DraftSystem()
        
        dS.overall("")
        dS.evalFun("bA * r")
        dS.overall("")
        
        dS.pOverall()
        dS.pEvalFun("iP * sO")
        dS.pOverall()

if __name__ == "__main__":
    unittest.main()
