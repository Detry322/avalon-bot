import numpy as np

from moves import *

from collections import defaultdict




def get_zero_move_set(state, player, is_bad):
    scores = []
    moves = state.get_moves(player, is_bad)
    for potential_move in moves:
        if state.is_proposing(player):
            _, proposal = potential_move
            scores.append(0)
        elif state.is_final():
            _, pick = potential_move
            if pick is None:
                scores.append(10)
            else:
                scores.append(0)
        else:
            if state.is_on_mission(player) and (is_bad ^ (potential_move[0] == MISSION_SUCCEED)):
                scores.append(10)
            else:
                scores.append(0)
    scores = np.exp(np.array(scores))
    scores = scores / np.sum(scores)
    return { move: probability for move, probability in zip(moves, scores) }
