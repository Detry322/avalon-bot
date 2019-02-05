import random

from battlefield.bots.bot import Bot
from battlefield.avalon_types import VoteAction

# Plays randomly
class RandomBot(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states


    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        return random.choice(legal_actions)


class RandomBotUV(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states


    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        if state.status == 'vote':
            return VoteAction(up=True)

        return random.choice(legal_actions)
