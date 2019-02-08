"""Base bot class"""
class Bot:
    def __init__(self, game, player, role, hidden_states):
        pass


    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        raise NotImplemented


    def get_move_probabilities(self, state, legal_actions):
        raise NotImplemented
