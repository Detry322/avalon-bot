import numpy as np
import itertools as it
from collections import defaultdict

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

def all_actions(player, action, opponent_possible_actions):
    everyone_actions = [
        opp_actions if p != player else { action: (1.0, None) }
        for p, opp_actions in enumerate(opponent_possible_actions)
    ]    
    for moves in it.product(*everyone_actions):
        prob = 1.0
        for player, move in enumerate(moves):
            p, _ = everyone_actions[player][move]
            prob *= p
        yield (moves, prob)

NotPossible = ("not possible",)

class Solver:
    def __init__(self, game):
        self.game = game
        self.memoized_player_solve = {}


    def _get_opponent_moves(self, state, h, belief_tensor):
        lower_belief_tensor = belief_tensor[:-1]
        opponent_moves = []
        for i in range(self.game.NUM_PLAYERS):
            if len(belief_tensor) == 0:
                actions = set(self.game.possible_moves(i, state, self.game.HIDDEN_STATES[h]))
                # Return uniform over all actions
                opponent_moves.append({action : (1.0/len(actions), None) for action in actions})
            else:
                opponent_moves.append(
                    self.solve_for_player(i, belief_tensor[-1][h][i], state, lower_belief_tensor)
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


    def solve_for_player(self, player, player_belief, state, belief_tensor):
        key = (player, player_belief.tostring(), state, belief_tensor.tostring())
        if key not in self.memoized_player_solve:
            result = self.__solve_for_player(player, player_belief, state, belief_tensor)
            self.memoized_player_solve[key] = result
        
        return self.memoized_player_solve[key]


    def __solve_for_player(self, player, player_belief, state, belief_tensor):
        '''
        Given a player, a state, and a k-belief tensor, this function computes the
        set of rewards associated with optimal play for the player at level k.

        Inputs:
            player        : a number in {0, ..., N-1} corresponding to player index
            player_belief : player's belief about the hidden states.
            state         : the current state of the game
            belief_tensor : a (k, H, N, H) matrix corresponding to the belief state of the player.
                           Entry B_T[l][h][i][:] is the current player's model of the belief state of player i
                           conditioned on hidden state h at level l, and is a H-length array of probabilities
                           corresponding to what player i assigns likelihood of each hidden state h_i.

        Outputs:
            move_probs    : a map from actions -> (probability, rewards), which describes:
                            the player's probability of taking the given action,
                            as well as the expected future reward for taking that action if the hidden state turns out to be a certain thing.
        '''
        k, H, N, Hprime = belief_tensor.shape
        assert H == Hprime, "u dun goof"

        rewards = []
        for h, h_prob in enumerate(player_belief):
            if h_prob == 0:
                rewards.append(NotPossible)
                continue

            hidden_state = self.game.HIDDEN_STATES[h]

            opponent_moves = self._get_opponent_moves(state, h, belief_tensor)

            my_action_reward_pairs = {}
            for my_action in self.game.possible_moves(player, state, hidden_state):

                my_current_reward_for_taking_action = None

                for new_belief, new_state, new_belief_tensor, r, p in self.simulate_state(player, player_belief, state, h, belief_tensor, opponent_moves, my_action):               
                    assert p != 0, "Transitioning to nonexistent state"
                    my_future_reward = 0.0

                    if not self.game.state_is_final(new_state):
                        future_actions = self.solve_for_player(player, new_belief, new_state, new_belief_tensor)

                        for action, (p_action, rs) in future_actions.items():
                            my_future_reward += rs[h] * p_action

                    if my_current_reward_for_taking_action is None:
                        my_current_reward_for_taking_action = p * (r + my_future_reward)
                    else:
                        my_current_reward_for_taking_action += p * (r + my_future_reward)

                if my_current_reward_for_taking_action is not None:
                    my_action_reward_pairs[my_action] = my_current_reward_for_taking_action

            assert len(my_action_reward_pairs) != 0, "Empty reward action pairs"
            rewards.append(my_action_reward_pairs)
        return self._get_move(player_belief, rewards)


    def _get_move(self, belief, payoff_matrix):
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
        acceptable_moves = None
        for prob, move_payoffs in zip(belief, payoff_matrix):
            if prob == 0:
                continue
            if acceptable_moves is None:
                acceptable_moves = set(move_payoffs.keys())
            else:
                acceptable_moves &= set(move_payoffs.keys())

        if acceptable_moves is None or len(acceptable_moves) == 0:
            return {}

        actual_payoffs = defaultdict(lambda: 0.0)
        for prob, move_payoffs in zip(belief, payoff_matrix):
            if prob == 0:
                continue
            for move in acceptable_moves:
                actual_payoffs[move] += prob*move_payoffs[move]

        actions, payoffs = zip(*actual_payoffs.items())
        # Do some softmax stuff
        payoffs = np.array(payoffs)
        payoffs -= np.max(payoffs)
        probs = np.exp(payoffs)
        probs /= np.sum(probs)
        return {
            move: (prob, np.array([
                float('nan') if payoff_for_hidden is NotPossible else payoff_for_hidden[move]
                for payoff_for_hidden in payoff_matrix
            ]))
            for move, prob in zip(actions, probs)
        }


    def simulate_state(self, player, player_belief, state, h, belief_tensor, opponent_actions, player_action):
        '''
        Given a MDP state, a hidden state, a player and a set of actions governing a transition between states,
        return a list of next MDP states, with probabilities and rewards associated with them.

        Inputs:
            MDP state (state, belief_tensor)
            player          : player index
            player_belief   : the belief of the player
            h               : hidden state index
            opponent_action : a (N,) array of maps from action to move probability for each player
            player_action   : a map of action to move probability corresponding to opponent_action[player]
        Output:
            next_states     : a list of tuples (belief', state', belief_tensor', r, p) corresponding to new MDP state and
                              output
        '''
        assert abs(np.sum(player_belief) - 1) < 0.000001, "PLAYER's belief doesn't sum to 1"
        hidden_state = self.game.HIDDEN_STATES[h]
        yielded = False
        for moves, prob in all_actions(player, player_action, opponent_actions):
            next_state = self.game.transition(state, hidden_state, moves)
            observation = self.game.observation(state, hidden_state, moves)
            rewards = self.game.rewards(state, hidden_state, moves)
            player_reward = rewards[player]
            new_belief, new_belief_tensor = self.get_beliefs(player, player_belief, belief_tensor, state, next_state, observation, player_action)
            
            assert abs(np.sum(new_belief) - 1) < 0.000001, "New belief doesn't sum to 1"

            yielded = True
            yield (new_belief, next_state, new_belief_tensor, player_reward, prob)

        assert yielded, "Simulate state didn't yield anything"


    def get_beliefs(self, player, player_belief, belief_tensor, state, next_state, observation, player_action):
        new_belief = self.single_update(player, player_belief, player_action, belief_tensor, state, next_state, observation)
        new_belief_tensor = self.update_tensor(belief_tensor, state, next_state, observation)
        return new_belief, new_belief_tensor


    def update_tensor(self, belief_tensor, state, next_state, observation):
        '''
        Given state, belief tensor, and output of transition (i.e., new state and observation), compute the
        belief update for the new belief tensor.
        '''
        if belief_tensor.shape[0] == 0:
            return belief_tensor

        lower_belief_tensor = belief_tensor[:-1]
        new_lower_belief_tensor = self.update_tensor(lower_belief_tensor, state, next_state, observation)
        new_beliefs = np.zeros((1, len(self.game.HIDDEN_STATES), self.game.NUM_PLAYERS, len(self.game.HIDDEN_STATES)))
        new_belief_tensor = np.concatenate([new_lower_belief_tensor, new_beliefs])

        for h, hidden_state in enumerate(self.game.HIDDEN_STATES):
            moves = self.game.infer_possible_actions(state, hidden_state, observation)

            for player in range(self.game.NUM_PLAYERS):
                player_belief = belief_tensor[-1, h, player]
                player_move = moves[player]
                new_belief_tensor[-1, h, player, :] = self.single_update(player, player_belief, player_move, lower_belief_tensor, state, next_state, observation)

        assert not np.all(new_belief_tensor == 0), "new belief tensor isn't valid"
        return new_belief_tensor


    def single_update(self, player, my_belief, my_move, belief_tensor, state, next_state, observation):
        '''
        Compute an update for a single player's belief conditioned on a single hidden state given
        previous belief tensor and a set of actions governing a transition.
        
        Inputs: 
            player           : the player who is updating
            my_belief.       : the original belief
            my_move          : the move this player took
            belief_tensor    : the matrix that determines the opponents beliefs
            state            : the current state
            next_state       : the next state the occured
            observation      : the observation that occured
        '''
        new_belief = np.zeros(len(self.game.HIDDEN_STATES))
        for h, prob in enumerate(my_belief):
            if prob == 0:
                continue
            hidden_state = self.game.HIDDEN_STATES[h]
            opponent_actions = self._get_opponent_moves(state, h, belief_tensor)
            for moves, moves_prob in all_actions(player, my_move, opponent_actions):
                if next_state != self.game.transition(state, hidden_state, moves):
                    continue
                if observation != self.game.observation(state, hidden_state, moves):
                    continue

                new_belief[h] += prob * moves_prob

        s = np.sum(new_belief)
        if s == 0:
            return new_belief
        else:
            return new_belief / s
