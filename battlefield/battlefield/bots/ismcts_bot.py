import random
import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, VoteAction, ProposeAction, MissionAction
from battlefield.bots.ismcts.ismcts import search_ismcts
from battlefield.bots.ismcts.moismcts import search_moismcts
from battlefield.bots.ismcts.mtmoismcts import search_mtmoismcts

class ISMCTSBot(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'run':
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)


    def get_action(self, state, legal_actions):
        if len(legal_actions) == 1:
            return legal_actions[0]

        action, _ = search_ismcts(self.player, state, self.hidden_states, 100)
        return action


    def get_move_probabilities(self, state, legal_actions):
        raise NotImplemented



class MOISMCTSBot(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'run':
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)


    def get_action(self, state, legal_actions):
        if len(legal_actions) == 1:
            return legal_actions[0]

        actions, _ = search_mtmoismcts(self.player, state, self.hidden_states, 100)
        return actions[self.role in EVIL_ROLES]


    def get_move_probabilities(self, state, legal_actions):
        raise NotImplemented
