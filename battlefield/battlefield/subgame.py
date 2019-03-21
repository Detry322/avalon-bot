import itertools
import random
import numpy as np
import warnings
from collections import defaultdict

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction
from battlefield.avalon import AvalonState
from battlefield.bots import SimpleStatsBot, ObserveBot, RandomBot, HumanLikeBot


def calculate_observation_ll(hidden_state, bot_classes, observation_history, tremble=0.0):
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(hidden_state))
    ]
    state = AvalonState.start_state(len(hidden_state))
    bots = [ bot() for bot in bot_classes ]
    for i, bot in enumerate(bots):
        bot.reset(state, i, hidden_state[i], beliefs[i])

    log_likelihood = 0.0

    for obs_type, obs in observation_history:
        assert obs_type == state.status, "Incorrect matchup {} != {}".format(obs_type, state.status)

        moving_players = state.moving_players()
        moves = []

        if obs_type == 'propose':
            player = moving_players[0]
            legal_actions = state.legal_actions(player, hidden_state)
            move = ProposeAction(proposal=obs)
            index = legal_actions.index(move)
            moves.append(move)
            move_probs = bots[player].get_move_probabilities(state, legal_actions)
            move_probs = (1.0 - tremble) * move_probs + tremble * (np.ones(len(legal_actions))/len(legal_actions))
            log_likelihood += np.log(move_probs[index])
        elif obs_type == 'vote':
            for p, vote_up in zip(moving_players, obs):
                legal_actions = state.legal_actions(p, hidden_state)
                move = VoteAction(up=vote_up)
                index = legal_actions.index(move)
                moves.append(move)
                move_probs = bots[p].get_move_probabilities(state, legal_actions)
                move_probs = (1.0 - tremble) * move_probs + tremble * (np.ones(len(legal_actions))/len(legal_actions))
                log_likelihood += np.log(move_probs[index])
        elif obs_type == 'run':
            bad_guys_on_mission = [p for p in state.proposal if hidden_state[p] in EVIL_ROLES ]
            if len(bad_guys_on_mission) < obs:
                # Impossible - fewer bad than failed
                return np.log(0.0)

            player_fail_probability = {}
            for bad in bad_guys_on_mission:
                legal_actions = state.legal_actions(bad, hidden_state)
                move = MissionAction(fail=True)
                index = legal_actions.index(move)
                move_probs = bots[bad].get_move_probabilities(state, legal_actions)
                move_probs = (1.0 - tremble) * move_probs + tremble * (np.ones(len(legal_actions))/len(legal_actions))
                player_fail_probability[bad] = move_probs[index]


            failure_prob = 0.0
            moves = [ MissionAction(fail=False) ] * len(state.proposal)
            for bad_failers in itertools.combinations(bad_guys_on_mission, r=obs):
                specific_fail_prob = 1.0
                for bad in bad_guys_on_mission:
                    moves[state.proposal.index(bad)] = MissionAction(fail=True) if bad in bad_failers else MissionAction(fail=False)
                    specific_fail_prob *= player_fail_probability[bad] if bad in bad_failers else (1.0 - player_fail_probability[bad])
                failure_prob += specific_fail_prob
            log_likelihood += np.log(failure_prob)

        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)
        state = new_state

    return log_likelihood

def calculate_subgame_ll(roles, num_players, bot_classes, observation_history, tremble=0.0):
    hidden_states = possible_hidden_states(roles, num_players)
    probabilities = np.zeros(len(hidden_states))

    for h, hidden_state in enumerate(hidden_states):
        with np.errstate(divide='ignore'):
            probabilities[h] = calculate_observation_ll(hidden_state, bot_classes, observation_history, tremble=tremble)

    return hidden_states, probabilities


def test_calculate():
    roles = ['merlin', 'assassin', 'minion', 'servant']
    # bot_classes = [ RandomBot, RandomBot, ObserveBot, RandomBot, RandomBot ]
    bot_classes = [ HumanLikeBot ] * 5
    # observation_history = [
    #     # Round 1
    #     ('propose', (1, 2)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (0, 2)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 1),
    #     # Round 2
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 0),
    #     # Round 3
    #     ('propose', (2, 3)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 0),
    #     # Round 4
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 1),
    #     # Round 5
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, True, True, False, False)),
    #     ('propose', (1, 2, 3)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (0, 1, 3)),
    #     ('vote', (True, False, False, False, True)),
    # ]
    observation_history = [
        # Round 1
        ('propose', (0, 4)),
        ('vote', (True, True, True, True, False)),
        ('run', 1),
        # Round 2
        ('propose', (1, 3, 4)),
        ('vote', (True, True, False, True, False)),
        # ('run', 1),
        # Round 3
        # ('propose', (1, 2)),
        # ('vote', (True, True, False, False, False)),
        # ('propose', (2, 3)),
        # ('vote', (False, False, True, True, True)),
        # ('run', 0),
        # # Round 4
        # ('propose', (2, 3, 4)),
        # ('vote', (False, False, True, True, True)),
        # ('run', 0),
        # # # Round 5
        # ('propose', (0, 1, 3)),
        # ('vote', (True, True, False, False, False)),
        # ('propose', (1, 3, 4)),
        # ('vote', (True, True, False, False, False)),
        # ('propose', (0, 2, 3)),
        # ('vote', (True, True, False, False, False)),
        # ('propose', (2, 3, 4)),
        # ('vote', (False, False, False, True, True)),
        # ('propose', (2, 3, 4)),
        # ('vote', (True, True, True, False, True))
    ]

    hidden_states, lls = calculate_subgame_ll(roles, 5, bot_classes, observation_history, tremble=0.0)
    lls -= np.max(lls)
    
    probs = np.exp(lls)
    probs /= np.sum(probs)
    multiple = np.max(probs) / 50
    for hidden_state, prob in zip(hidden_states, probs):
        print "{: >10} {: >10} {: >10} {: >10} {: >10}: {prob}".format(*hidden_state, prob='#' * int(prob / multiple))


