from solver import Solver
import numpy as np
import itertools as it


def all_actions(possible_actions):
    for moves in it.product(*possible_actions):
        prob = 1.0
        for player, move in enumerate(moves):
            p, _ = possible_actions[player][move]
            prob *= p
        yield (moves, prob)


def observer_update_belief(solver, belief, belief_tensor, state, next_state, observation):
    new_belief = np.zeros(len(solver.game.HIDDEN_STATES))

    for h, h_prob in enumerate(belief):
        if h_prob == 0:
            continue
        hidden_state = solver.game.HIDDEN_STATES[h]
        possible_actions = solver._get_opponent_moves(state, h, belief_tensor)
        for moves, prob in all_actions(possible_actions):
            if solver.game.transition(state, hidden_state, moves) != next_state:
                continue
            if solver.game.observation(state, hidden_state, moves) != observation:
                continue

            new_belief[h] += h_prob * prob

    new_belief /= np.sum(new_belief)
    return new_belief


def observe_playthrough(solver, playthrough, k=2):
    belief = np.ones(len(solver.game.HIDDEN_STATES))
    belief /= np.sum(belief)

    belief_tensor = solver.game.get_starting_belief_tensor(k)

    for state, next_state, observation in playthrough:
        print " =========== "
        print " State: {}".format(state)
        print " Belief: {}".format(belief)
        print " Next : {}".format(next_state)
        print " Obs  : {}".format(observation)

        new_belief = observer_update_belief(solver, belief, belief_tensor, state, next_state, observation)
        print " New B: {}".format(new_belief)
        belief = new_belief
        belief_tensor = solver.update_tensor(belief_tensor, state, next_state, observation)


    print "Final belief: {}".format(belief)
