import numpy as np
np.random.seed(11)

from game import PhysicalGameState, NUM_PLAYERS, display_move
import moves
from forward_solver import get_value_and_move, initial_belief, initial_k_belief, single_belief_update, k_belief_update

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
    print_v("====== Player {} thinks:".format(perspective))

    for player in range(NUM_PLAYERS):
        belief = perspective_belief[player]
        print_v(" === " + explain_belief(player, belief))
        if player == perspective:
            continue
        if belief != 0.0:
            b = perspective_k_beliefs[0][player][1] if len(perspective_k_beliefs) != 0 else None
            k_belief = perspective_k_beliefs[1:]
            _, move_if_bad = get_value_and_move(state, player, True, b, k_belief)
            print_v("If player {} is bad, then he ".format(player) + explain_move(move_if_bad))
        if belief != 1.0:
            b = perspective_k_beliefs[0][player][0] if len(perspective_k_beliefs) != 0 else None
            k_belief = perspective_k_beliefs[1:]
            _, move_if_good = get_value_and_move(state, player, False, b, k_belief)
            print_v("If player {} is good, then he ".format(player) + explain_move(move_if_good))



def play_game(ks, bad):
    state = PhysicalGameState()

    total_payoff = 0.0

    player_beliefs = [
        [initial_belief(player, player == bad) if k >= 1 else None, initial_k_belief(k - 1)]
        for player, k in enumerate(ks)
    ]

    while not state.finished():
        print_v("=========== {}".format(state))

        for player, (belief, _) in enumerate(player_beliefs):
            print_v("Player {} ({}) (k={}) BELIEF: {}".format(player, 'BAD' if player == bad else '   ', ks[player], belief))

        for player, (belief, k_belief) in enumerate(player_beliefs):
            explain(state, player, belief, k_belief)

        moves = [
            get_value_and_move(
                state,
                player,
                player == bad,
                player_beliefs[player][0],
                player_beliefs[player][1]
            )[1]
            for player in range(NUM_PLAYERS)
        ]

        print_v("===== move probs")

        for player, moveset in enumerate(moves):
            explained_moveset = { display_move(m): prob for m, prob in moveset.items() }
            print_v("Player {} ({}): {}".format(player, 'BAD' if player == bad else '   ', explained_moveset))

        print_v("===== actual moves")

        moves = [get_move(moveset) for moveset in moves]

        for player, move in enumerate(moves):
            print_v("Player {} ({}): {}".format(player, 'BAD' if player == bad else '   ', display_move(move)))
        payoff = state.payoff(moves, bad, False)
        print_v("Immediate payoff: {}".format(payoff))
        total_payoff += payoff
        print_v("Cumulative payoff: {}".format(total_payoff))
        new_state = state.move(moves)
        if k == 0:
            state = new_state
            continue

        print_v("Updating beliefs")
        for player in range(NUM_PLAYERS):
            player_beliefs[player][0] = single_belief_update(state, tuple(moves), player_beliefs[player][0], player_beliefs[player][1])
            player_beliefs[player][1] = k_belief_update(state, tuple(moves), player_beliefs[player][1])

        state = new_state

    return total_payoff



def play_main():
    ks = [7, 7, 8]
    bad = 2
    num_games = 1000
    for player, k in enumerate(ks):
        print_v("=== Solving player {}, k={}".format(player, k))
        solve_forward(player, bad, k)

    total_payoff = 0.0
    for _ in range(num_games):
        total_payoff += play_game(ks, bad)
    print ("Average payoff: {}".format(total_payoff/num_games))


def solve_main():
    solve_forward(0, 1, 2)


if __name__ == "__main__":
    # solve_main()
    play_main()
