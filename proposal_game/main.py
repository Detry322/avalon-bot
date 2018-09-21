import numpy as np
np.random.seed(10)

from game import PhysicalGameState, NUM_PLAYERS, display_move
import moves
from forward_solver import get_value_and_move, initial_belief, initial_k_belief, single_belief_update, k_belief_update


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


def play_game(k, bad):
    state = PhysicalGameState()

    total_payoff = 0.0

    player_beliefs = [
        [initial_belief(player, player == bad) if k >= 1 else None, initial_k_belief(k - 1)]
        for player in range(NUM_PLAYERS)
    ]

    while not state.finished():
        print "=========== {}".format(state)
        moves = [
            get_move(get_value_and_move(
                state,
                player,
                player == bad,
                player_beliefs[player][0],
                player_beliefs[player][1]
            )[1])
            for player in range(NUM_PLAYERS)
        ]
        for player, move in enumerate(moves):
            print "Player {} ({}): {}".format(player, 'BAD' if player == bad else '   ', display_move(move))
        payoff = state.payoff(moves, bad, False)
        print "Immediate payoff: {}".format(payoff)
        total_payoff += payoff
        print "Cumulative payoff: {}".format(total_payoff)
        new_state = state.move(moves)
        if k == 0:
            state = new_state
            continue

        print "Updating beliefs"
        for player in range(NUM_PLAYERS):
            player_beliefs[player][0] = single_belief_update(state, new_state, player_beliefs[player][0], player_beliefs[player][1])
            player_beliefs[player][1] = k_belief_update(state, new_state, player_beliefs[player][1])

        state = new_state



def main():
    k = 3
    bad = 1
    if k >= 1:
        print "===== Solving game at k={}, player {} is bad".format(k, bad)
        for player in range(NUM_PLAYERS):
            print "=== Solving player {}".format(player)
            solve_forward(player, bad, k)
    play_game(k, bad)


if __name__ == "__main__":
    main()
