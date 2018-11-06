import numpy as np
np.random.seed(11)

from game import PhysicalGameState, NUM_PLAYERS, display_move
from moves import *
import itertools
from forward_solver import get_value_and_move, initial_belief, initial_k_belief, single_belief_update, k_belief_update, product

from collections import namedtuple
Round = namedtuple('Round', ['number', 'proposal', 'result'])

VERBOSE = False

def print_v(message):
    if VERBOSE:
        print message


def solve_forward(player, bad, k):
    assert k >= 1
    k -= 1
    state = PhysicalGameState()
    is_bad = player == bad
    initial = initial_belief(player, is_bad)
    initial_k = initial_k_belief(k)
    get_value_and_move(state, player, is_bad, initial, initial_k)


def get_move(moveset):
    moves, probabilities = zip(*moveset.items())
    index = np.random.choice(np.arange(len(moves)), p=np.array(probabilities))
    return moves[index]


def update_observer(observer_belief, state, new_state, player_beliefs):
    new_belief = np.array([0.0]*NUM_PLAYERS)

    for bad_guy, probability in enumerate(observer_belief):
        if probability == 0.0:
            continue
        moves_if_bad_guy = []
        for player in range(NUM_PLAYERS):
            player_is_bad = player == bad_guy
            _, move = get_value_and_move(state, player, player_is_bad, player_beliefs[bad_guy][player][0], player_beliefs[bad_guy][player][1])
            moves_if_bad_guy.append(move)

        for move_list in itertools.product(*moves_if_bad_guy):
            move_prob = product(moves_if_bad_guy[player][move] for player, move in enumerate(move_list))

            sprime = state.move(move_list)

            if new_state == sprime:
                new_belief[bad_guy] = probability*move_prob

    return new_belief / np.sum(new_belief)


def rounds_to_states(rounds):
    states = []
    fails = 0
    succeeds = 0
    for i, r in enumerate(rounds):
        states.append(
            PhysicalGameState(round_=i, proposal=None, fails=fails, succeeds=succeeds)
        )
        states.append(
            PhysicalGameState(round_=i, proposal=tuple([p - 1 for p in r.proposal]), fails=fails, succeeds=succeeds)
        )
        if r.result == 'fail':
            fails += 1
        else:
            succeeds += 1

    states.append(PhysicalGameState(round_=len(rounds), proposal=None, fails=fails, succeeds=succeeds))
    return states


def backfill_possible(state, next_state):
    possible_moves = []
    for bad_guy in range(NUM_PLAYERS):
        moves = []
        if state.proposal is not None:
            if state.fails == (next_state.fails - 1) and bad_guy not in state.proposal:
                continue
            for player in range(NUM_PLAYERS):
                if player in state.proposal:
                    moves.append((MISSION_FAIL, ) if bad_guy == player else (MISSION_SUCCEED,))
                else:
                    moves.append((OBSERVE,))
        else:
            for player in range(NUM_PLAYERS):
                moves.append((PROPOSE, next_state.proposal) if player == state.round else (OBSERVE,))

        possible_moves.append((bad_guy, tuple(moves)))
    return possible_moves




def observer_belief_runthrough(ks, rounds):
    states = rounds_to_states(rounds)

    for s in states:
        print s


    state = PhysicalGameState()

    total_payoff = 0.0

    player_beliefs = [
        [
            [initial_belief(player, player == bad) if k >= 1 else None, initial_k_belief(k - 1)]
            for player, k in enumerate(ks)
        ]
        for bad in range(NUM_PLAYERS)
    ]

    observer_belief = np.array([1.0]*NUM_PLAYERS)/NUM_PLAYERS

    state = states[0]
    states = states[1:]

    while len(states) > 0:
        print '================='
        print state
        print "Observer belief: {}".format(observer_belief)

        next_state = states[0]
        possible_moves = backfill_possible(state, next_state)

        observer_belief = update_observer(observer_belief, state, next_state, player_beliefs)

        for bad_guy, moves in possible_moves:
            for player, k in enumerate(ks):
                if k != 0:
                    player_beliefs[bad_guy][player][0] = single_belief_update(
                        state,
                        tuple(moves),
                        player_beliefs[bad_guy][player][0],
                        player_beliefs[bad_guy][player][1]
                    )
                    player_beliefs[bad_guy][player][1] = k_belief_update(
                        state,
                        tuple(moves),
                        player_beliefs[bad_guy][player][1]
                    )

        state = next_state
        states = states[1:]

    print "========= Final belief: ========="
    print "Observer belief: {}".format(observer_belief)



def play_main():
    ks = [1, 1, 2]
    for bad in range(NUM_PLAYERS):
        for player, k in enumerate(ks):
            print ("=== Solving player {}, k={} {}".format(player, k, "BAD" if bad == player else ""))
            solve_forward(player, bad, k)

    rounds = [
        Round(1, [1, 2], 'fail'),
        Round(2, [2, 3], 'success'),
        Round(3, [2, 3], 'success'),
    ]

    observer_belief_runthrough(ks, rounds)


if __name__ == "__main__":
    play_main()
