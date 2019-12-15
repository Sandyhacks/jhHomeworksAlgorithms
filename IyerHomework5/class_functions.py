import random

class TicTacToe:

    def __init__(self):
        # initialize with 3 x 3 empty grid at first
        self.layout = [" ", " ", " ", 
                      " ", " ", " ", 
                      " ", " ", " "]

    def display(self):
        print("""
          -------------
          | {} | {} | {} |
          -------------
          | {} | {} | {} |
          -------------
          | {} | {} | {} |
          -------------
        """.format(*self.layout))        

    def availability(self):
        moves = []
        for i in range(0, len(self.layout)):
            if self.layout[i] == " ":
                moves.append(i)
        return moves

    def make_move(self, pos, player):
        self.layout[pos] = player

    def win_check(self):
        combinations = ([0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6])
        for player in ("X", "O"):
            pos = []
            for i in range(0, len(self.layout)):
                if self.layout[i] == player:
                    pos.append(i)
            for combo in combinations:
                w = True
                for position in combo:
                    if position not in pos:
                        w = False
                if w:
                    return player

    def play_over(self):
        if self.win_check() != None:
            return True
        for index in self.layout:
            if index == " ":
                return False
        return True

    def minimax(self, node, depth, player):
        """
        recursive DFS for each possible game state and choose new move
        """
        if depth == 0 or node.play_over():
            if node.win_check() == "X":
                return 0
            elif node.win_check() == "O":
                return 100
            else:
                return 50

        if player == "O":
            b_value = 0
            for move in node.availability():
                node.make_move(move, player)
                value = self.minimax(node, depth-1, new_player(player))
                node.make_move(move, " ")
                b_value = max(b_value, value)
            return b_value
        
        if player == "X":
            b_value = 100
            for move in node.availability():
                node.make_move(move, player)
                value = self.minimax(node, depth-1, new_player(player))
                node.make_move(move, " ")
                b_value = min(b_value, value)
            return b_value

def new_player(player):
    if player == "X":
        return "O"
    else:
        return "X"

def AI(layout, depth, player):
    choice = []
    for move in layout.availability():
        layout.make_move(move, player)
        value = layout.minimax(layout, depth-1, new_player(player))
        layout.make_move(move, " ")
        if value > 50:
            choice = [move]
            break
        elif value == 50:
            choice.append(move)
    if len(choice) > 0:
        return random.choice(choice)
    else:
        return random.choice(layout.availability())
