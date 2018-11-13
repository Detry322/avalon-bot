import numpy as np
from collections import namedtuple

from game import Game

N = 3
GOOD_REWARD = 1
EVIL_REWARD = -1

Move = namedtuple('Move', ['type', 'extra'])
HiddenState = namedtuple('HiddenState', ['evil'])
PhysicalState = namedtuple('PhysicalState', ['round', 'proposal'])
Observation = namedtuple('Observation', ['success', 'proposal', 'bad_picks'])

def all_physical_states():
    states = []
    for r in range(N):
        states.append(PhysicalState(round=r, proposal=None))
        for odd_one_out in range(N):
            proposal = frozenset(set(range(N)) - set([odd_one_out]))
            states.append(PhysicalState(round=r, proposal=proposal))

    states.append(PhysicalState(round=N, proposal=None))
    states.append(PhysicalState(round=N+1, proposal=None))
    return states


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
    def possible_moves(cls, player, state, hidden_state):
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
            moves = [Move(type='Pick', extra=p) for p in range(cls.NUM_PLAYERS)]
            moves.append(Move(type='Pick', extra=None))
            return moves

        return [Move(type=None, extra=None)]


    @classmethod
    def rewards(cls, state, hidden_state, moves):
        rewards = [0 for _ in range(cls.NUM_PLAYERS)]
        success = 1 if not any([move.type == 'Fail' for move in moves]) else -1
        if state.proposal is not None and any([move.type is not None for move in moves]):
            for player in range(cls.NUM_PLAYERS):
                if hidden_state.evil == player:
                    rewards[player] = EVIL_REWARD * success
                else:
                    rewards[player] = GOOD_REWARD * success
        return rewards


    @classmethod
    def observation(cls, state, hidden_state, moves):
        if state.proposal is None and state.round < N:
            return Observation(success=None, proposal=moves[state.round].extra, bad_picks=None)
        elif state.proposal is None and state.round == N:
            return Observation(success=None, proposal=None, bad_picks=tuple(move.extra for move in moves))
        elif state.proposal is not None:
            success = not any([move.type == 'Fail' for move in moves])
            return Observation(success=success, proposal=state.proposal, bad_picks=None)
        raise Exception("Execution should never reach here - you didn't check state_is_final correctly")


    @classmethod
    def transition(cls, state, hidden_state, moves):
        if state.round == N:
            return PhysicalState(round=N+1, proposal=None)
        elif state.round == N+1:
            return None
        else:
            if state.proposal is None:
                # search through moves to find the proposal
                proposer = state.round
                proposal = moves[proposer].extra
                return PhysicalState(round=state.round, proposal=proposal)
            else:
                return PhysicalState(round=state.round+1, proposal=None)

    @classmethod
    def state_is_final(cls, state):
        return state.round == (N + 1)


    @classmethod
    def infer_possible_actions(cls, state, hidden_state, obs):
        actions = [Move(type=None, extra=None) for _ in range(cls.NUM_PLAYERS)]
        if state.proposal is None:
            if state.round == N:
                for player, bad_pick in enumerate(obs.bad_picks):
                    if player == hidden_state.evil:
                        actions[player] = Move(type='Pick', extra=bad_pick)
            else:
                actions[state.round] = Move(type='Propose', extra=obs.proposal)
        else:
            if obs.success:
                proposer = state.round
                p1, p2 = (proposer + 1) % 3, (proposer + 2) % 3
                actions[p1], actions[p2] = Move(type='Pass', extra=None), Move(type='Pass', extra=None)
            else:
                traitor = hidden_state.evil
                passer = list(set(state.proposal) - set([traitor]))[0]
                actions[traitor] = Move(type='Fail', extra=None)
                actions[passer] = Move(type='Pass', extra=None)
        return actions
