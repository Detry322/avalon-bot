import numpy as np
import itertools as it
from collections import namedtuple

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

# def get_action_sets(state, hidden_state, player):
#     '''
#     Get all possible action sets for opponents.
#     '''
#     return it.product(*[range(len(ProposalGame.__possible_moves_for_player(state, hidden_state, i))) if i != player else [None] for i in range(N)])

class Solver:
    def __init__(self, game):
        self.game = game


    def _get_opponent_moves(self, state, h, lower_belief_tensor, opponent_payoffs):
        opponent_moves = []
        for i in range(self.game.NUM_PLAYERS):
            if len(lower_belief_tensor) == 0:
                actions = set(self.game.possible_moves(i, state, self.game.HIDDEN_STATES[h]))
                # Return uniform over all actions
                opponent_moves.append({action : 1.0/len(actions) for action in actions})
            else:
                opponent_moves.append(
                    self.get_move(lower_belief_tensor[-1][h][i], opponent_payoffs[i])
                )
        return opponent_moves


    def _get_my_action_space(self, player, state, belief):
        my_possible_actions = None
        for h, p in enumerate(belief):
            if p == 0:
                continue
            hidden_state = self.game.HIDDEN_STATES[h]
            actions = set(self.game.possible_moves(player, state, hidden_state))
            if my_possible_actions is None:
                my_possible_actions = actions
            else:
                my_possible_actions &= actions
        return sorted(my_possible_actions)


    def solve_for_player(self, player, state, belief_tensor):
        '''
        Given a player, a state, and a k-belief tensor, this function computes the
        set of rewards associated with optimal play for the player at level k.

        Inputs:
            player        : a number in {0, ..., N-1} corresponding to player index
            state         : the current state of the game
            belief_tensor : a (k, H, N, H) matrix corresponding to the belief state of the player.
                           Entry B_T[l][h][i][:] is the current player's model of the belief state of player i
                           conditioned on hidden state h at level l, and is a H-length array of probabilities
                           corresponding to what player i assigns likelihood of each hidden state h_i.

        Outputs:
            rewards       : a (H, A) matrix corresponding to payoffs for action a given hidden state h is true
        '''
        k, H, N, Hprime = belief_tensor.shape
        assert H == Hprime, "u dun goof"

        if k == 0:
            return None
        lower_belief_tensor = belief_tensor[:k-1]
        opponent_payoffs = [self.solve_for_player(i, state, lower_belief_tensor) for i in range(self.game.NUM_PLAYERS)]
        rewards = []
        for h, hidden_state in enumerate(self.game.HIDDEN_STATES):
            opponent_moves = self._get_opponent_moves(state, h, lower_belief_tensor, opponent_payoffs)

            my_action_reward_pairs = []
            for my_action in self.game.possible_moves(player, state, hidden_state):
                next_mdp_states = self.simulate_state(player, state, h, belief_tensor, opponent_moves, my_action)

                my_current_reward_for_taking_action = 0.0

                for new_state, new_belief_tensor, r, p in next_mdp_states:
                    future_rewards = self.solve_for_player(player, new_state, new_belief_tensor)
                    my_future_actions = self.get_move(new_belief_tensor[-1][h][player], future_rewards)

                    my_future_reward = 0.0
                    for action, p_action in my_future_actions:
                        my_future_reward += future_rewards[h][action]*p_action

                    my_current_reward_for_taking_action = p*(r + my_future_reward)

                my_action_reward_pairs.append((my_action, my_current_reward_for_taking_action))

            rewards.append(my_action_reward_pairs)
        return rewards


    def get_move(self, belief, payoff_matrix):
        '''
        Given a belief (probability distribution over hidden states) and a payoff matrix of
        expected future rewards, return a probability distribution over actions. The distribution
        will be argmax of the weighted expected future rewards plus trembling hand noise.

        Inputs:
            belief        : a (H,) matrix summing to 1 of probabilities over hidden states.
            payoff_matrix : a (H,) matrix consisting of lists of expected future rewards given actions.
        Outputs:
            strategy      : a (A,) matrix consisting of a probability distribution over action states.
        '''
        pass


    def simulate_state(self, player, state, h, belief_tensor, opponent_actions, player_action):
        '''
        Given a MDP state, a hidden state, a player and a set of actions governing a transition between states,
        return a list of next MDP states, with probabilities and rewards associated with them.

        Inputs:
            MDP state (state, belief_tensor)
            player          : player index
            h               : hidden state index
            opponent_action : a (N,) array of maps from action to move probability for each player
            player_action   : a map of action to move probability corresponding to opponent_action[player]
        Output:
            next_states     : a list of tuples (state', belief_tensor', r, p) corresponding to new MDP state and
                              output
        '''
        pass


    def single_update(self, player, h, belief_tensor, opponent_actions):
        '''
        Compute an update for a single player's belief conditioned on a single hidden state given
        previous belief tensor and a set of actions governing a transition.
        '''
        pass


    def get_beliefs(self, state, belief_tensor, next_state, observation):
        '''
        Given state, belief tensor, and output of transition (i.e., new state and observation), compute the
        belief update for the new belief tensor.
        '''
        pass


