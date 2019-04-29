import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, MissionAction, VoteAction, ProposeAction, PickMerlinAction
from battlefield.bots.deeprole.lookup_tables import get_deeprole_perspective, assignment_id_to_hidden_state, print_top_k_belief, print_top_k_viewpoint_belief
from battlefield.bots.deeprole.run_deeprole import run_deeprole_on_node
from battlefield.bots.cfr_bot import proposal_to_bitstring, bitstring_to_proposal

START_NODE = {
    "type": "TERMINAL_PROPOSE_NN",
    "succeeds": 0,
    "fails": 0,
    "propose_count": 0,
    "proposer": 0,
    "new_belief": list(np.ones(60)/60.0)
}

def assert_eq(a, b):
    assert a == b, "{} != {}".format(a, b)

def votes_to_bitstring(vote_moves):
    result = 0
    for i, vote in enumerate(vote_moves):
        if vote.up:
            result |= (1 << i)
    return result

def print_move_probs(probs, legal_actions, cutoff=0.95):
    zipped = zip(probs, legal_actions)
    zipped.sort(reverse=True)
    total = 0.0
    print "----- move probs -----"
    while total < cutoff and len(zipped) > 0:
        prob, move = zipped[0]
        zipped = zipped[1:]
        print move, prob
        total += prob

# Plays randomly, except always fails missions if bad.
class Deeprole(Bot):
    def __init__(self):
        pass


    def reset(self, game, player, role, hidden_states):
        self.node = run_deeprole_on_node(START_NODE)
        self.player = player
        self.perspective = get_deeprole_perspective(player, hidden_states[0])
        # print self.perspective


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'merlin':
            return

        if old_state.status == 'propose':
            proposal = observation
            bitstring = proposal_to_bitstring(proposal)
            child_index = self.node['propose_options'].index(bitstring)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'vote':
            child_index = votes_to_bitstring(observation)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'run':
            num_fails = observation
            self.node = self.node['children'][num_fails]

        if self.node['type'] == 'TERMINAL_PROPOSE_NN':
            # print_top_k_belief(self.node['new_belief'])
            print_top_k_viewpoint_belief(self.node['new_belief'], self.player, self.perspective)
            print "Player {} NN value: {}".format(self.player, self.node['nn_output'][self.player][self.perspective])
            self.node = run_deeprole_on_node(self.node)

        if self.node['type'].startswith("TERMINAL_") and self.node['type'] != "TERMINAL_MERLIN":
            return

        assert_eq(new_state.succeeds, self.node["succeeds"])
        assert_eq(new_state.fails, self.node["fails"])

        if new_state.status == 'propose':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
        elif new_state.status == 'vote':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))
        elif new_state.status == 'run':
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))


    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.array([1.0])

        if state.status == 'propose':
            probs = np.zeros(len(legal_actions))
            propose_strategy = self.node['propose_strat'][self.perspective]
            propose_options = self.node['propose_options']
            for strategy_prob, proposal_bitstring in zip(propose_strategy, propose_options):
                action = ProposeAction(proposal=bitstring_to_proposal(proposal_bitstring))
                probs[legal_actions.index(action)] = strategy_prob
            # print probs
            # print_move_probs(probs, legal_actions)
            return probs
        elif state.status == 'vote':
            probs = np.zeros(len(legal_actions))
            vote_strategy = self.node['vote_strat'][self.player][self.perspective]
            for strategy_prob, vote_up in zip(vote_strategy, [False, True]):
                action = VoteAction(up=vote_up)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'run':
            probs = np.zeros(len(legal_actions))
            mission_strategy = self.node['mission_strat'][self.player][self.perspective]
            for strategy_prob, fail in zip(mission_strategy, [False, True]):
                action = MissionAction(fail=fail)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'merlin':
            # print np.array(self.node['merlin_strat'][self.player][self.perspective])
            return np.array(self.node['merlin_strat'][self.player][self.perspective])


