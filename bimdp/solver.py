import numpy as np
from collections import namedtuple

import util
from proposal_game import ProposalGame



class Solver:

    def solve_for_player(cls, player, state, belief_tensor):
        pass

    def get_move(cls, belief, payoff_matrix):
        pass

    def simulate_state(cls, player, state, belief_tensor, opponent_actions, player_action):
        pass

    def single_update(cls, player, state, belief_tensor, opponent_actions):
        pass

    def get_beliefs(cls, state, belief_tensor, next_state, observation):
        pass


