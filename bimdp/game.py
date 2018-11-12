class Game(object):
    # Required fields
    NUM_PLAYERS = None
    HIDDEN_STATES = []
    PHYSICAL_STATES = []
    START_PHYSICAL_STATE = None

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
