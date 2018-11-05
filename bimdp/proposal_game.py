import numpy as np
from collections import namedtuple

from game import Game

N = 3

Move = namedtuple('Move', ['type', 'extra'])
HiddenState = namedtuple('HiddenState', ['evil'])
PhysicalState = namedtuple('PhysicalState', ['round', 'proposal'])

def all_physical_states():
    states = []
    for r in range(N):
        states.append(PhysicalState(round=r, proposal=None))
        for odd_one_out in range(N):
            proposal = frozenset(set(range(N)) - set([odd_one_out]))
            states.append(PhysicalState(round=r, proposal=proposal))

    states.append(PhysicalState(round=N, proposal=None))
    states.append(PhysicalState(round=N+1, proposal=None))


class ProposalGame(Game):
    NUM_PLAYERS = N
    HIDDEN_STATES = [HiddenState(evil=player) for player in range(NUM_PLAYERS)]
    PHYSICAL_STATES = all_physical_states()
    START_PHYSICAL_STATE = PHYSICAL_STATES[0]

    @classmethod
    def initial_belief(cls, player, hidden_state):
        if hidden_state.evil == player:
            return np.array([0.0 if p != player else 1.0 for p in range(cls.NUM_PLAYERS)])
        return np.array([1.0/(cls.NUM_PLAYERS - 1) if p != player else 0.0 for p in range(cls.NUM_PLAYERS)])


    @classmethod
    def __possible_moves_for_player(cls, state, hidden_state, player):
        if state.proposal is None and state.round == player:
            return [
                Move(type='Propose', extra=frozenset(set(range(cls.NUM_PLAYERS)) - set([odd_one_out])))
                for odd_one_out in range(cls.NUM_PLAYERS)
            ]
        
        if state.proposal is not None and player in state.proposal:
            if hidden_state.evil == player:
                return [Move(type='Pass', extra=None), Move(type='Fail', extra=None)]
            else:
                return [Move(type='Pass', extra=None)]

        if state.round == cls.NUM_PLAYERS and hidden_state.evil != player:
            return [Move(type='Pick', extra=p) for p in range(cls.NUM_PLAYERS)]

        return [Move(type=None, extra=None)]



    @classmethod
    def possible_moves(cls, state, hidden_state):
        return [cls.__possible_moves_for_player(state, hidden_state, player) for player in range(cls.NUM_PLAYERS)]


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
        return state.round == (N + 1)
