import numpy as np
import random

def determinization_iterator(possible_hidden_states, num_iterations):
    i = 0
    hidden = list(possible_hidden_states)
    while i < num_iterations:
        random.shuffle(hidden)
        for h in hidden:
            if i >= num_iterations:
                return
            yield i, h
            i += 1



def random_choice(values, p=None):
    return values[np.random.choice(range(len(values)), p=p)]


def simulate(game_state, hidden_state):
    while not game_state.is_terminal():
        moves = tuple([
            random_choice(game_state.legal_actions(player, hidden_state))
            for player in game_state.moving_players()
        ])
        game_state, hidden_state, _ = game_state.transition(moves, hidden_state)
    return game_state.terminal_value(hidden_state)
