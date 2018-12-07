import numpy as np

class Game(object):
    # Required fields
    NUM_PLAYERS = None
    HIDDEN_STATES = []
    PHYSICAL_STATES = []
    START_PHYSICAL_STATE = None

    @classmethod
    def get_starting_belief_tensor(cls, k):
        h = len(cls.HIDDEN_STATES)
        belief = np.zeros((k, h, cls.NUM_PLAYERS, h))
        for ki in range(k):
            for hi, hidden_state in enumerate(cls.HIDDEN_STATES):
                for player in range(cls.NUM_PLAYERS):
                    belief[ki, hi, player] = cls.initial_belief(player, hidden_state)
        return belief


    @classmethod
    def initial_belief(cls, player, hidden_state):
        raise NotImplemented


    @classmethod
    def possible_moves(cls, player, state, hidden_state):
        raise NotImplemented


    @classmethod
    def rewards(cls, state, hidden_state, moves):
        raise NotImplemented


    @classmethod
    def observation(cls, state, hidden_state, moves):
        raise NotImplemented


    @classmethod
    def transition(cls, state, hidden_state, moves):
        raise NotImplemented


    @classmethod
    def state_is_final(cls, state):
        raise NotImplemented
