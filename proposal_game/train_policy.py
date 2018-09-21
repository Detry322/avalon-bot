import numpy as np
from glob import glob

from moves import *
from policy import *
from game import *

from collections import defaultdict

def belief_update(physical_state, new_physical_state, belief_state, bad_guy_moves, good_guy_moves):
    for bad_guy in range(NUM_PLAYERS):
        pass




# def observable_moves(good_guy_moves, bad_guy_moves):
#     pass


# def calculate_reachable_states(start, player, is_bad, good_policies, bad_policies):
#     if not is_bad:
#         initial_belief = tuple([1.0/(NUM_PLAYERS - 1) if p != player else 0.0 for p in range(NUM_PLAYERS)])
#     else:
#         initial_belief = tuple([0.0 if p != player else 1.0 for p in range(NUM_PLAYERS)])

#     initial_state = (start, initial_belief)
#     to_explore = [initial_state]
#     explored = set([])
#     while len(to_explore) > 0:
#         explored.add(to_explore[-1])
#         physical_state, belief_state = to_explore.pop()

#         bad_guy_moves = [ bad_policy.get_move_set(physical_state) for bad_policy in bad_policies ]
#         good_guy_moves = [ good_policy.get_move_set(physical_state) for good_policy in good_policies ]

#         for new_physical_state in physical_state.next_states():
#             new_belief = belief_update(physical_state, new_physical_state, belief_state, bad_guy_moves, good_guy_moves)
#             new_state = (new_physical_state, new_belief)



#         moves = [ policy.get_move_set(physical_state) for policy in policies ]

#     return explored




# def train_player_policy


def train_level_one_policies():
    pass
