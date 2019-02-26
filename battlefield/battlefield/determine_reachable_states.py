from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction, PickMerlinAction
from battlefield.avalon import AvalonState



def determine_reachable(base_bot, roles, num_players):
    hidden_state_to_index = {
        hs: i for i, hs in enumerate(possible_hidden_states(roles, num_players))
    }
    print len(hidden_state_to_index)
