import random
import os
import json

from battlefield.bots.bot import Bot
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, VoteAction, ProposeAction, MissionAction

GAME_DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'games.json'))

human_data = None

def perspective_from_hidden_states(hidden_states):
    roles = [set([]) for _ in range(len(hidden_states[0]))]
    for hidden_state in hidden_states:
        for p, role in hidden_state:
            roles[p].add(role)
    return tuple([ frozenset(possible) for possible in roles ])


def parse_human_data(human_json):
    pass


def load_human_data():
    global human_data
    if human_data is not None:
        return human_data

    with open(GAME_DATAFILE, 'r') as f:
        human_json = json.load(f)

    result = parse_human_data(human_json)
    human_data = result
    return result


class HumanBot(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.is_evil = role in EVIL_ROLES
        self.data = load_human_data()
        self.current_node = self.data.get((self.game.NUM_PLAYERS, player, perspective_from_hidden_states(hidden_states)))


    def handle_transition(self, old_state, new_state, observation, move=None):
        if self.current_node is not None:
            self.current_node = self.current_node['transitions'].get((new_state.as_key(), observation, move))
            if self.current_node is None:
                print "Warning: exited human play..."


    def get_action(self, state, legal_actions):
        if self.current_node is None:
            return random.choice(legal_actions)

        return max(self.current_node['move_counts'], key=self.current_node['move_counts'].get)
