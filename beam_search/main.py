from proposal_game import ProposalGame, HiddenState, Round, create_playthrough, load_games
from agent import Agent
from play import play_game
from observer import observe_playthrough

import numpy as np
import pandas as pd


ALL_GAMES = load_games()
VALID_GAMES = [25, 139, 68, 60, 98, 36, 32, 33, 74, 132, 106, 77, 103, 112, 108, 107, 100, 62, 61, 131, 44, 42, 94, 39, 130, 109, 38, 102, 96, 79, 114]

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


MCTS_ITER = [1, 5, 10, 50, 100, 500, 1000]

def run_game((game_id, iterations)):
    game = ProposalGame
    result = []
    print game_id, iterations
    rounds = ALL_GAMES[game_id]
    playthrough = create_playthrough(rounds)
    k = 1
    solver = Agent(game, k, np.ones(3)/3.0, False, iterations)
    results = observe_playthrough(solver, playthrough, k)
    for stage, belief in enumerate(results):
        result.append({
            'game': game_id,
            'stage': stage,
            'k': 1,
            'gamma': 1.0,
            'mcts_iter': iterations,
            'p1': belief[0],
            'p2': belief[1],
            'p3': belief[2],
        })
    return result

def collect_data():
    import multiprocessing
    import itertools
    pool = multiprocessing.Pool()
    result = []

    args = list(itertools.product(VALID_GAMES, MCTS_ITER))

    results = pool.map(run_game, args)

    for r in results:
        result.extend(r)

    df = pd.DataFrame(result)
    return df

if __name__ == "__main__":
    df = collect_data()
    df.to_msgpack('mcts_data.msg')
    print df
