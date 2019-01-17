from game import GameState
from collections import namedtuple

Propose = namedtuple('Propose', ['people'])
Pass = 'Pass'
Fail = 'Fail'
Abstain = 'Abstain'
GuessBad = namedtuple('GuessBad', ['person'])

class ProposalGameState(GameState):
    NUM_PLAYERS = 3
    HIDDEN_STATES = range(3)

    def __init__(self, round_, proposal, succeeds, fails, picks):
        self.round = round_
        self.proposal = proposal
        self.fails = fails
        self.succeeds = succeeds
        self.picks = picks


    @classmethod
    def start_state(cls):
        """
        Returns the starting state
        """
        return cls(0, None, 0, 0, None)


    def is_terminal(self):
        """
        Returns true if the game has ended
        """
        return self.round == self.NUM_PLAYERS + 1


    def terminal_value(self, bad_player):
        """
        Returns the payoff for each player
        """
        assert self.is_terminal(), "Called terminal_value() on a non terminal node"
        good_payoff = self.succeeds - self.fails
        for _, pick in self.picks.items():
            if pick == bad_player:
                good_payoff += 10.0
            else:
                good_payoff -= 100.0
        return [good_payoff if player != bad_player else (-2*good_payoff) for player in range(self.NUM_PLAYERS)]


    def moving_players(self):
        """
        Returns an array of players whose turn it is.
        """
        assert not self.is_terminal(), "Called moving_players on terminal node"
        if self.proposal is None:
            if self.round < self.NUM_PLAYERS:
                # Proposal round
                return [self.round]
            else:
                # Guess bad round
                return range(self.NUM_PLAYERS)
        else:
            return sorted(self.proposal)


    def legal_actions(self, player, bad_player):
        """
        Returns the legal actions of the player from this state, given a hidden state
        """
        assert not self.is_terminal(), "Called legal actions on terminal node"
        assert player in self.moving_players(), "Tried moving when it's not your turn"
        if self.proposal is None:
            if self.round < self.NUM_PLAYERS:
                # Proposal
                all_players = frozenset(range(self.NUM_PLAYERS))
                return [Propose(all_players - frozenset([p])) for p in range(self.NUM_PLAYERS)]
            else:
                # Guess bad
                moves = [Abstain]
                if player != bad_player:
                    moves.extend(GuessBad(p) for p in range(self.NUM_PLAYERS))
                return moves
        else:
            # Fail/pass
            return [Pass, Fail] if player == bad_player else [Pass]


    def transition(self, moves, bad_player):
        """
        Returns a tuple:
        state': the new state
        bad_player': the new hidden state
        observation: the communal observation made by all of the play
        """
        assert not self.is_terminal(), "Called transition on terminal node"
        assert len(moves) == len(self.moving_players()), "Someone tried to move who isn't allowed to"
        moves = { p: move for p, move in zip(self.moving_players(), moves) }
        assert all(move in self.legal_actions(p, bad_player) for p, move in moves.items()), "Someone played an illegal move"

        if self.proposal is None:
            if self.round < self.NUM_PLAYERS:
                # Proposal
                new_state = ProposalGameState(self.round, moves[self.round].people, self.succeeds, self.fails, None)
                return new_state, bad_player, moves[self.round]
            else:
                # Pick bad
                picks = { player: guess.person for player, guess in moves.items() if guess != Abstain }
                new_state = ProposalGameState(self.round + 1, None, self.succeeds, self.fails, picks)
                return new_state, bad_player, None
        else:
            # Run proposal round
            did_fail = any(move == Fail for move in moves.values())
            new_succeeds = self.succeeds + (1 if not did_fail else 0)
            new_fails = self.fails + (1 if did_fail else 0)
            new_state = ProposalGameState(self.round + 1, None, new_succeeds, new_fails, None)
            return new_state, bad_player, (Fail if did_fail else Pass)


