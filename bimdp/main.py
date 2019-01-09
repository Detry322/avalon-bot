from proposal_game import ProposalGame, HiddenState, Round, create_playthrough
from solver import Solver
from play import play_game
from observer import observe_playthrough

import numpy as np


def play():
    game = ProposalGame
    k = 2
    hidden_state = HiddenState(evil=1)
    play_game(game, hidden_state, k)


def observe():
    game = ProposalGame
    rounds = [
        Round(1, [1, 2], 'fail'),
        Round(2, [2, 3], 'success'),
        Round(3, [2, 3], 'success'),
    ]
    playthrough = create_playthrough(rounds)
    k = 4
    solver = Solver(game)
    observe_playthrough(solver, playthrough, k)


if __name__ == "__main__":
    observe()
