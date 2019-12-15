import random
from class_functions import new_player
from class_functions import AI
from class_functions import TicTacToe


if __name__ == '__main__':
    play = TicTacToe()
    play.display()

    while play.play_over() == False:
        print('For this game, you play as X')
        turn = int(input("Enter a number from 1 to 9: "))
        play.make_move(turn-1, "X")
        play.display()
        if play.play_over() == True:
            break
        computer_move = AI(play, -1, "O")
        play.make_move(computer_move, "O")
        play.display()

    if play.win_check() == "X":
        print('X wins')
    elif play.win_check() == "O":
        print('O wins')
    elif play.play_over() == True:
        print('Draw')