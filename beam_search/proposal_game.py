import numpy as np
from collections import namedtuple
import itertools as it

from game import Game

N = 3
PROP_SIZE = 2
GOOD_REWARD = 1
EVIL_REWARD = -1

Move = namedtuple('Move', ['type', 'extra'])
HiddenState = namedtuple('HiddenState', ['evil'])
PhysicalState = namedtuple('PhysicalState', ['round', 'proposal'])
Observation = namedtuple('Observation', ['success', 'proposal', 'bad_picks'])
Round = namedtuple('Round', ['number', 'proposal', 'result'])


def create_playthrough(rounds):
    playthrough = []
    for i, r in enumerate(rounds):
        proposal = frozenset([p - 1 for p in r.proposal])
        playthrough.append(
            (
                PhysicalState(round=i, proposal=None),
                PhysicalState(round=i, proposal=proposal),
                Observation(success=None, proposal=proposal, bad_picks=None)
            )
        )
        playthrough.append(
            (
                PhysicalState(round=i, proposal=proposal),
                PhysicalState(round=i+1, proposal=None),
                Observation(success=(r.result != 'fail'), proposal=proposal, bad_picks=None)
            )
        )
    return playthrough


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
                Move(type='Propose', extra=frozenset(group))
                for group in it.combinations(range(cls.NUM_PLAYERS), r=PROP_SIZE)
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
        good_reward = 0
        if state.proposal is not None:
            good_reward = 1 if not any([move.type == 'Fail' for move in moves]) else -1
        elif state.round == N:
            correct = sum(1 if move.type == 'Pick' and move.extra == hidden_state.evil else 0 for move in moves)
            incorrect = sum(1 if move.type == 'Pick' and move.extra != hidden_state.evil and move.extra is not None else 0 for move in moves)
            good_reward = 10*correct - 100*incorrect

        return [(EVIL_REWARD if player == hidden_state.evil else GOOD_REWARD) * good_reward for player in range(cls.NUM_PLAYERS)]


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
                for player in state.proposal:
                    actions[player] = Move(type='Pass', extra=None)
            else:
                traitor = hidden_state.evil
                for player in state.proposal:
                    actions[player] = Move(type='Fail', extra=None) if traitor == player else Move(type='Pass', extra=None)
        return actions

    @classmethod
    def infer_action_sets(cls, state, obs, hidden_state):
        action_sets = [[] for _ in range(cls.NUM_PLAYERS)]
        if state.proposal is None:
            if state.round == N:
                for player, bad_pick in enumerate(obs.bad_picks):
                    if player != hidden_state.evil:
                        for pick in range(cls.NUM_PLAYERS):
                            action_sets[player].append(Move(type='Pick', extra=pick))
                    else:
                        action_sets[player].append(Move(type='Pick', extra=bad_pick))
            else:
                action_sets[state.round].append(Move(type='Propose', extra=obs.proposal))
                for player in range(cls.NUM_PLAYERS):
                    action_sets[player].append(Move(type=None, extra=None))
        else:
            if obs.success:
                for player in state.proposal:
                    action_sets[player].append(Move(type='Pass', extra=None))
            else:
                traitor = hidden_state.evil
                if traitor not in state.proposal:
                    return [[]]
                else:
                    for player in state.proposal:
                        action_sets[player].append(Move(type='Fail', extra=None) if traitor == player else Move(type='Pass', extra=None))
            for player in range(cls.NUM_PLAYERS):
                if player not in state.proposal:
                    action_sets[player].append(Move(type=None, extra=None))
        return list(it.product(*action_sets))

