from __future__ import print_function
from dlgo.agent import naive
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo.utils import print_board, print_move, point_from_coords
from six.moves import input
from dlgo import scoring


def main():
    while True:
        print("The board size(5-19)")
        board_size = int(input())
        if 19 >= board_size >= 5:
            break
        else:
            print("Wrong size,please input 5-19")
    game = goboard.GameState.new_game(board_size)
    bot = naive.RandomBot()
    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == gotypes.Player.black:
            human_move = input('-- ').upper()
            point = point_from_coords(human_move.strip())
            move = goboard.Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)
    print(scoring.compute_game_result(game))


if __name__ == '__main__':
    main()
