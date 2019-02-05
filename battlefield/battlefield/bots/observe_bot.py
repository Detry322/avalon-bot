import random

from battlefield.bots.bot import Bot
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, VoteAction, ProposeAction, MissionAction


class ObserveBot(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.is_evil = role in EVIL_ROLES


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'run':
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)


    def get_action(self, state, legal_actions):
        role_guess = random.choice(self.hidden_states)
        if state.status == 'vote':
            if state.propose_count == 4:
                return VoteAction(up=True)

            up_vote = role_guess[state.proposer] in GOOD_ROLES and all(role_guess[p] in GOOD_ROLES for p in state.proposal)
            if self.is_evil:
                up_vote = not up_vote
            return VoteAction(up=up_vote)

        if state.status == 'propose' and not self.is_evil:
            propose_size = len(legal_actions[0].proposal)
            good_players = [ p for p, role in enumerate(role_guess) if role in GOOD_ROLES ]
            random.shuffle(good_players)
            return ProposeAction(proposal=tuple(sorted(good_players[:propose_size])))

        if state.status == 'run' and self.is_evil:
            return MissionAction(fail=True)

        return random.choice(legal_actions)
