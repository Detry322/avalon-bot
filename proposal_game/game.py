import itertools
import copy

from moves import *

NUM_ROUNDS = 3
NUM_PLAYERS = 3
MISSION_SIZE = 2

PROPOSE = 0
MISSION_FAIL = 1
MISSION_SUCCEED = 2
OBSERVE = 3
PICK_BAD = 4

def display_move(move):
    type_ = move[0]
    moves = ["PROPOSE", "MISSION_FAIL", "MISSION_SUCCEED", "OBSERVE", "PICK_BAD"]
    return (moves[move[0]],) + move[1:]


class PhysicalGameState(object):
    def __init__(self, round_=0, proposal=None, fails=0, succeeds=0):
        self.round = round_
        self.proposal = proposal
        self.fails = fails
        self.succeeds = succeeds

    def is_final(self):
        return self.round == NUM_ROUNDS

    def finished(self):
        return self.round > NUM_ROUNDS

    def is_proposing(self, player):
        return self.proposal is None and not self.is_final() and self.round % NUM_PLAYERS == player

    def is_on_mission(self, player):
        return self.proposal is not None and player in self.proposal

    def get_moves(self, player, is_bad):
        if self.finished():
            return []
        if self.round == NUM_ROUNDS:
            if is_bad:
                return [(PICK_BAD, None)]
            return [(PICK_BAD, person) for person in itertools.chain([None], range(NUM_PLAYERS))]
        if self.proposal is not None:
            if player in self.proposal:
                return [(MISSION_SUCCEED,), (MISSION_FAIL,)]
            else:
                return [(OBSERVE,)]
        if self.round % NUM_PLAYERS == player:
            return [(PROPOSE, mission) for mission in itertools.combinations(range(NUM_PLAYERS), r=MISSION_SIZE)]
        else:
            return [(OBSERVE,)]


    def _good_guy_payoff(self, moves, bad_guy):
        if self.round == NUM_ROUNDS:
            num_correct = sum(1 if move[1] == bad_guy else 0 for move in moves)
            num_abstain = sum(1 if move[1] == None else 0 for move in moves)
            return 10.0*num_correct - 100.0*(NUM_PLAYERS - num_correct - num_abstain)

        if self.proposal == None:
            return 0.0
        
        failed = any(move[0] == MISSION_FAIL for move in moves)
        if failed:
            return -1.0
        else:
            return 1.0


    def payoff(self, moves, bad_guy, is_bad):
        payoff = self._good_guy_payoff(moves, bad_guy) 
        return -payoff if is_bad else payoff


    def move(self, moves):
        assert len(moves) == NUM_PLAYERS
        new_state = copy.copy(self)

        if self.round == NUM_ROUNDS:
            new_state.round += 1
            return new_state

        if self.proposal == None:
            move = moves[self.round % NUM_PLAYERS]
            assert move[0] == PROPOSE
            new_state.proposal = move[1]
            return new_state

        new_state.proposal = None
        new_state.round += 1
        failed = any(move[0] == MISSION_FAIL for move in moves)
        if failed:
            new_state.fails += 1
            return new_state
        else:
            new_state.succeeds += 1
            return new_state


    def next_states(self):
        if self.is_final() or self.finished():
            return []
        if self.proposal is None:
            return [
                PhysicalGameState(
                    round_=self.round,
                    proposal=proposal,
                    fails=self.fails,
                    succeeds=self.succeeds
                ) for proposal in itertools.combinations(range(NUM_PLAYERS), r=MISSION_SIZE)
            ]
        return [
            PhysicalGameState(round_=self.round + 1, proposal=None, fails=self.fails, succeeds=self.succeeds + 1),
            PhysicalGameState(round_=self.round + 1, proposal=None, fails=self.fails + 1, succeeds=self.succeeds),
        ]

        
    def __internal_rep(self):
        return (self.round, self.proposal, self.succeeds, self.fails)


    def __eq__(self, other):
        return self.__internal_rep() == other.__internal_rep()

    def __ne__(self, other):
        return not (self == other)


    def __hash__(self):
        return hash(self.__internal_rep())


    def __str__(self):
        return "<GameState Round={} Proposal={} Succeeded={} Failed={}>".format(*self.__internal_rep())


    def __repr__(self):
        return self.__str__()
