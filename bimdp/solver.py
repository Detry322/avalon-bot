import numpy as np
import itertools as it
from collections import namedtuple

from proposal_game import ProposalGame

def probability_of_action_set(action_set, probabilities):
    '''
    Given an action set, return the probability of that action set being played
    by opponents.

    Inputs:
        action_set    : a (N, 1) matrix of indices of actions for a given player
        probabilities : a (N, 1) matrix of lists of log probabilities of actions for any given player

    Output:
        probability   : the log probability of the action set to prevent underflow

    '''
    probability = 0
    for i in range(len(action_set)):
        if probabilities[i] is not None:
            probability += np.log(probabilities[i][action_set[i]])
    return probability

def get_action_sets(state, hidden_state, player):
    '''
    Get all possible action sets for opponents.
    '''
    return it.product(*[range(len(ProposalGame.__possible_moves_for_player(state, hidden_state, i))) if i != player else [None] for i in range(N)])

class Solver:

    def solve_for_player(cls, player, state, belief_tensor):
        '''
        Given a player, a state, and a k-belief tensor, this function computes the
        set of rewards associated with optimal play for the player at level k.

        Inputs:
            player        : a number in {0, ..., N-1} corresponding to player index
            state         : the current state of the game
            belief_tensor : a (H, N, H, k) matrix corresponding to the belief state of the player.
                           Entry B_T[:][i][h][l] is the current player's model of the belief state of player i
                           conditioned on hidden state h at level l, and is a H-length array of probabilities
                           corresponding to what player i assigns likelihood of each hidden state h_i.

        Outputs:
            rewards       : a (H, A) matrix corresponding to payoffs for action a given hidden state h is true
        '''
        # belief_tensor is B_t^k[0:H][0:N][0:H][0:k]
        N, H, k = belief_tensor.shape
        lower_belief_tensor = belief_tensor[:][:][k-2]
        opponent_payoffs = [solve_for_player(cls, i, state, lower_belief_tensor) if i != player else None for i in range(N)]
        rewards = []
        for h, hidden_state in enumerate(ProposalGame.HIDDEN_STATES):
            opponent_moves = [get_move(cls, lower_belief_tensor[:][h][k-2], opponent_payoffs[i]) if i != player else None for i in range(N)]
            next_states = [simulate_state(cls, state, h, belief_tensor, opponent_moves, action) for action in ProposalGame.__possible_moves_for_player(state, hidden_state, player)]
            sets_of_opponent_actions = get_action_sets(state, hidden_state, player)
            reward = sum([probability_of_action_set(action_set, opponent_moves) for action_set in sets_of_opponent_actions])

        return rewards

    def get_move(cls, belief, payoff_matrix):
        pass

    def simulate_state(cls, state, h, belief_tensor, opponent_actions, player_action):
        pass

    def single_update(cls, player, h, belief_tensor, opponent_actions):
        pass

    def get_beliefs(cls, state, belief_tensor, next_state, observation):
        pass


