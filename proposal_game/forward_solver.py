import operator
from moves import *
from game import *
from policy import get_zero_move_set

from collections import defaultdict

from functools import wraps

import numpy as np

BETA = 1.0

def memoized(f):
    backing_cache = {}
    @wraps(f)
    def cache_func(*args):
        if args not in backing_cache:
            result = f(*args)
            backing_cache[args] = result
        return backing_cache[args]
    return cache_func


def product(iterable):
    return reduce(operator.mul, iterable, 1.0)


@memoized
def get_move_probabilities(state, my_belief, k_beliefs):
    move_probabilities = []
    for bad_guy, probability in enumerate(my_belief):
        moves_if_bad_guy = []
        for player in range(NUM_PLAYERS):
            player_is_bad = player == bad_guy
            player_belief = None if len(k_beliefs) == 0 else k_beliefs[0][player][int(player_is_bad)]
            player_k_beliefs = k_beliefs[1:]
            _, move = get_value_and_move(state, player, player_is_bad, player_belief, player_k_beliefs)
            moves_if_bad_guy.append(move)
        move_probabilities.append((probability, tuple(moves_if_bad_guy)))
    return move_probabilities


@memoized
def single_belief_update(state, new_state, my_belief, k_beliefs):
    move_probabilities = get_move_probabilities(state, my_belief, k_beliefs)
    
    bad_probabilities = np.array([0.0]*NUM_PLAYERS) # this would be changed to support more belief states than just num_players
    for bad_guy, (bad_guy_prob, moves_if_bad) in enumerate(move_probabilities):
        for move_list in itertools.product(*moves_if_bad):
            if state.move(move_list) != new_state:
                continue # ignore moves that wouldn't have produced this state
            move_prob = product(moves_if_bad[player][m] for player, m in enumerate(move_list))
            bad_probabilities[bad_guy] += bad_guy_prob * move_prob

    new_belief = bad_probabilities / np.sum(bad_probabilities)
    return tuple(new_belief)


@memoized
def k_belief_update(state, new_state, k_beliefs):
    if len(k_beliefs) == 0:
        return k_beliefs

    lower_k_beliefs = k_belief_update(state, new_state, k_beliefs[1:])
    new_k_belief = tuple([
        tuple([
            single_belief_update(state, new_state, k_beliefs[0][player][is_bad], k_beliefs[1:]) # This would be changed to support N types.
            for is_bad in range(2)
        ])
        for player in range(NUM_PLAYERS)
    ])
    return (new_k_belief, ) + lower_k_beliefs


@memoized
def get_value_and_move(state, me, is_bad, my_belief, k_beliefs):
    if state.finished():
        return 0.0, {}

    if my_belief is None:
        return 0.0, get_zero_move_set(state, me, is_bad)

    move_probabilities = get_move_probabilities(state, my_belief, k_beliefs)

    value_if_move = defaultdict(lambda: 0.0)
    for bad_guy, (bad_guy_prob, moves_if_bad) in enumerate(move_probabilities):
        for move_list in itertools.product(*moves_if_bad):
            move = move_list[me]
            move_prob = product(moves_if_bad[player][m] if player != me else 1.0 for player, m in enumerate(move_list))
            new_state = state.move(move_list)
            immediate_payoff = state.payoff(move_list, bad_guy, is_bad)

            new_k_beliefs = k_belief_update(state, new_state, k_beliefs)
            my_new_belief = single_belief_update(state, new_state, my_belief, k_beliefs)

            future_payoff, _ = get_value_and_move(new_state, me, is_bad, my_new_belief, new_k_beliefs)
            value_if_move[move] += bad_guy_prob * move_prob * (immediate_payoff + BETA*future_payoff)

    moves, values = zip(*value_if_move.items())
    values = np.exp(np.array(values))
    probabilities = values / np.sum(values)

    moveset = { move: probability for move, probability in zip(moves, probabilities) }
    overall_value = sum(moveset[move]*value_if_move[move] for move in moves)
    return overall_value, moveset



def initial_belief(player, is_bad):
    if is_bad:
        return tuple([0.0 if p != player else 1.0 for p in range(NUM_PLAYERS)])
    else:
        return tuple([0.0 if p == player else 1.0/(NUM_PLAYERS - 1) for p in range(NUM_PLAYERS)])


def initial_k_belief(k):
    return tuple([
        tuple([
            tuple([
                initial_belief(player, int(is_bad))
                for is_bad in range(2)
            ])
            for player in range(NUM_PLAYERS)
        ])
        for _ in range(k)
    ])
