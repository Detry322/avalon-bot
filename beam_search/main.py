from proposal_game import ProposalGame, HiddenState, Round, create_playthrough
from agent import Agent
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
        Round(1, [2, 3], 'success'),
        Round(2, [2, 3], 'fail'),
        Round(3, [2, 3], 'fail'),
    ]
    playthrough = create_playthrough(rounds)
    k = 2
    # hacky initial distribution
    solver = Agent(game, k, [1./3, 1./3, 1./3], False)
    observe_playthrough(solver, playthrough, k)


if __name__ == "__main__":
    observe()
