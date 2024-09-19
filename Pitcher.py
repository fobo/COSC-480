import ast
import operator as op

class Pitcher:
    def __init__(self, player_info):
        info = player_info.split(",")
        
        # Fill blank stats with 0's
        info = [i if i.strip() else "0" for i in info]
        
        name_split = list(filter(None, info[1]))
        
        if len(name_split) == 3:
            self.first_name = name_split[0] + name_split[1]
        else:
            self.first_name = name_split[0]
        
        self.last_name = name_split[-1]
        
        # Check if end of last name has a non-letter, if so, remove it.
        if self.last_name[-1] in "#*":
            self.last_name = self.last_name[:-1]
        
        # Assign stats
        self.sO = float(info[21])
        self.eval_result = self.sO  # Evaluation result starts with strikeouts
        
        self.iP = float(info[14])   # Innings pitched
        self.h = float(info[15])    # Hits allowed
        self.r = float(info[16])    # Runs allowed
        self.hR = float(info[18])   # Home runs allowed
        self.team_name = info[3]    # Team name
        self.player_position = "P"
        self.is_drafted = False
        
    def __str__(self):
        return f"{self.first_name} {self.last_name} {self.team_name} {self.player_position} {self.eval_result}"

    def set_is_drafted(self):
        self.is_drafted = True

    def get_position(self):
        return self.player_position

    def get_first_name(self):
        return self.first_name

    def get_last_name(self):
        return self.last_name

    def get_is_drafted(self):
        return self.is_drafted

    def get_SO(self):
        return self.sO

    def get_IP(self):
        return self.iP

    def get_H(self):
        return self.h

    def get_R(self):
        return self.r

    def get_HR(self):
        return self.hR

    def evaluate(self, expression):
        # Define safe operators for the eval expression
        allowed_operators = {
            ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
            ast.USub: op.neg
        }
        
        # Recursively evaluate the expression safely
        def eval_expr(node):
            if isinstance(node, ast.Num):  # <number>
                return node.n
            elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                return allowed_operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
                return allowed_operators[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Name):  # Variables (sO, iP, h, r, hR)
                return getattr(self, node.id)
            else:
                raise TypeError(f"Unsupported expression: {node}")
        
        try:
            # Parse the expression and evaluate it safely
            parsed_expr = ast.parse(expression, mode='eval').body
            result = eval_expr(parsed_expr)
            self.eval_result = result
            return result
        except Exception as e:
            raise ValueError(f"Invalid expression: {expression}") from e
