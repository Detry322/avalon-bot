import numpy as np
from collections import namedtuple

import util
from proposal_game import ProposalGame

class Solver:

    def solve_for_player(cls, player, state, belief_tensor):

        # belief_tensor is B_t^k[0:N][0:H][0:k]
        N, H, k = belief_tensor.shape
        lower_belief_tensor = belief_tensor[:][:][k-2]
        opponent_payoffs = [solve_for_player(cls, i, state, lower_belief_tensor) if i != player else None for i in range(N)]
        for h, hidden_state in enumerate(ProposalGame.HIDDEN_STATES):
            opponent_moves = [get_move(cls, lower_belief_tensor[:][h][k-2], opponent_payoffs[i]) if i != player else None for i in range(N)]
            next_states = [simulate_state(cls, state, h, belief_tensor, opponent_moves, action) for action in ProposalGame.__possible_moves_for_player(state, hidden_state, player)]
            rewards = sum([])
        return rewards

    def get_move(cls, belief, payoff_matrix):
        pass

    def simulate_state(cls, state, h, belief_tensor, opponent_actions, player_action):
        pass

    def single_update(cls, player, h, belief_tensor, opponent_actions):
        pass

    def get_beliefs(cls, state, belief_tensor, next_state, observation):
        pass


