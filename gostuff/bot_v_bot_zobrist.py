from dlgo.agent import naive
from dlgo import goboard
from dlgo import gotypes
from dlgo import scoring
from dlgo.utils import print_board, print_move
import time
import timeit


def main():
    while True:
        print("The board size(5-19)")
        board_size = int(input())
        if 19 >= board_size >= 5:
            break
        else:
            print("Wrong size,please input 5-19")
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: naive.RandomBot(),
        gotypes.Player.white: naive.RandomBot(),
    }
    start = timeit.default_timer()
    while not game.is_over():
        time.sleep(0.3)
        print(chr(27) + "[2J")
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
    stop = timeit.default_timer()
    print(scoring.compute_game_result(game))
    print('Runtime:', stop - start, 'seconds')


if __name__ == '__main__':
    main()
