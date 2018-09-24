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


def explain_belief(player, belief_value):
    result = "Player {} (b: {:0.2f}) ".format(player, belief_value)
    if belief_value == 0.0:
        return result + "is definitely not bad"
    elif belief_value < 0.1:
        return result + "is almost certainly not bad"
    elif belief_value < 0.4:
        return result + "is probably not bad"
    elif belief_value < 0.6:
        return result + "could be either"
    elif belief_value < 0.9:
        return result + "is probably bad"
    elif belief_value != 1.0:
        return result + "is almost certainly bad"
    else:
        return result + "is definitely bad"


def explain_move(moveset):
    if len(moveset) == 1:
        return "has to play {}".format(display_move(moveset.keys()[0]))

    best = max(moveset, key=lambda m: moveset[m])
    return "will probably ({:0.2f}) play {}".format(moveset[best], display_move(best))


def explain(state, perspective, perspective_belief, perspective_k_beliefs):
    print "====== Player {} thinks:".format(perspective)

    for player in range(NUM_PLAYERS):
        belief = perspective_belief[player]
        print " === " + explain_belief(player, belief)
        if player == perspective:
            continue
        if belief != 0.0:
            b = perspective_k_beliefs[0][player][1] if len(perspective_k_beliefs) != 0 else None
            k_belief = perspective_k_beliefs[1:]
            _, move_if_bad = get_value_and_move(state, player, True, b, k_belief)
            print "If player {} is bad, then he ".format(player) + explain_move(move_if_bad)
        if belief != 1.0:
            b = perspective_k_beliefs[0][player][0] if len(perspective_k_beliefs) != 0 else None
            k_belief = perspective_k_beliefs[1:]
            _, move_if_good = get_value_and_move(state, player, False, b, k_belief)
            print "If player {} is good, then he ".format(player) + explain_move(move_if_good)



def play_game(k, bad):
    state = PhysicalGameState()

    total_payoff = 0.0

    player_beliefs = [
        [initial_belief(player, player == bad) if k >= 1 else None, initial_k_belief(k - 1)]
        for player in range(NUM_PLAYERS)
    ]

    while not state.finished():
        print "=========== {}".format(state)

        for player, (belief, _) in enumerate(player_beliefs):
            print "Player {} ({}) BELIEF: {}".format(player, 'BAD' if player == bad else '   ', belief)

        for player, (belief, k_belief) in enumerate(player_beliefs):
            explain(state, player, belief, k_belief)

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
            player_beliefs[player][0] = single_belief_update(state, tuple(moves), player_beliefs[player][0], player_beliefs[player][1])
            player_beliefs[player][1] = k_belief_update(state, tuple(moves), player_beliefs[player][1])

        state = new_state



def play_main():
    k = 2
    bad = 1
    if k >= 1:
        print "===== Solving game at k={}, player {} is bad".format(k, bad)
        for player in range(NUM_PLAYERS):
            print "=== Solving player {}".format(player)
            solve_forward(player, bad, k)
    play_game(k, bad)


def solve_main():
    solve_forward(0, 1, 2)


if __name__ == "__main__":
    # solve_main()
    play_main()
