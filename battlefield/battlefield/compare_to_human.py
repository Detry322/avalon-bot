import numpy as np
import json
import os

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction, PickMerlinAction
from battlefield.avalon import create_avalon_game

DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bots', 'data', 'relabeled.json'))
TREMBLING_HAND_PROB = 0.2

ROLES = ['servant', 'merlin', 'percival', 'minion', 'assassin', 'mordred', 'morgana', 'oberon']

human_data = None

def reconstruct_hidden_state(game):
    roles = []
    for player in game['players']:
        roles.append('minion' if player['spy'] else 'servant')

    for role, p in game['roles'].items():
        if role not in set(['assassin', 'merlin', 'mordred', 'percival', 'morgana', 'oberon']):
            continue
        roles[p] = role
    return tuple(roles)


def handle_transition(state, hidden_state, moves, bots):
    p_to_move = { p: move for p, move in zip(state.moving_players(), moves) }
    new_state, _, observation = state.transition(moves, hidden_state)
    for p, bot in enumerate(bots):
        bot.handle_transition(state, new_state, observation, move=p_to_move.get(p))
    return new_state


def get_ll(state, hidden_state, player, bot, move):
    legal_actions = state.legal_actions(player, hidden_state)
    move_probs = bot.get_move_probabilities(state, legal_actions)
    move_probs += np.ones(len(move_probs)) * TREMBLING_HAND_PROB
    move_probs = move_probs / np.sum(move_probs)

    assert move in legal_actions, "Something is amiss"
    return np.log(move_probs[legal_actions.index(move)])


def handle_round(state, hidden_state, bots, round_, stats):
    last_proposal = None
    for proposal_num in ['1', '2', '3', '4', '5']:
        proposal = last_proposal = round_[proposal_num]
        assert state.proposer == proposal['proposer']
        assert state.propose_count == int(proposal_num) - 1
        moves = [ProposeAction(proposal=tuple(sorted(proposal['team'])))]
        for player, move in zip(state.moving_players(), moves):
            ll = get_ll(state, hidden_state, player, bots[player], move)
            stats['all']['total'] += ll
            stats['all']['propose'] += ll
            stats[hidden_state[player]]['total'] += ll
            stats[hidden_state[player]]['propose'] += ll
        state = handle_transition(state, hidden_state, moves, bots)

        assert state.status == 'vote'
        moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
        for player, move in zip(state.moving_players(), moves):
            ll = get_ll(state, hidden_state, player, bots[player], move)
            stats['all']['total'] += ll
            stats['all']['vote'] += ll
            stats[hidden_state[player]]['total'] += ll
            stats[hidden_state[player]]['vote'] += ll
        state = handle_transition(state, hidden_state, moves, bots)

        if state.status == 'run':
            break

    secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
    moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
    for player, move in zip(state.moving_players(), moves):
        ll = get_ll(state, hidden_state, player, bots[player], move)
        stats['all']['total'] += ll
        stats['all']['mission'] += ll
        stats[hidden_state[player]]['total'] += ll
        stats[hidden_state[player]]['mission'] += ll
    state = handle_transition(state, hidden_state, moves, bots)

    if state.status == 'merlin':
        assert 'findMerlin' in round_
        find_merlin = round_['findMerlin']
        assert hidden_state[find_merlin['assassin']] == 'assassin'
        moves = [
            PickMerlinAction(merlin=find_merlin['merlin_guess'])
            for _ in hidden_state
        ]
        for player, move in zip(state.moving_players(), moves):
            ll = get_ll(state, hidden_state, player, bots[player], move)
            stats['all']['total'] += ll
            stats['all']['merlin'] += ll
            stats[hidden_state[player]]['total'] += ll
            stats[hidden_state[player]]['merlin'] += ll
        state = handle_transition(state, hidden_state, moves, bots)
    return state




def process_game(game, bot_class, stats, verbose=True, num_players=None, max_num_players=7):
    try:
        hidden_state = reconstruct_hidden_state(game)
        if num_players is not None:
            if len(hidden_state) != num_players:
                return
        else:
            if len(hidden_state) >= max_num_players:
                return
        if verbose:
            print game['id']
        possible = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
        perspectives = [
            starting_hidden_states(player, hidden_state, possible)
            for player, _ in enumerate(hidden_state)
        ]
        state = create_avalon_game(len(hidden_state)).start_state()
        bots = [
            bot_class(state, player, role, perspectives[player])
            for player, role in enumerate(hidden_state)
        ]
        for round_ in game['log']:
            state = handle_round(state, hidden_state, bots, round_, stats)
    except AssertionError:
        if verbose:
            print game['id'], 'is bad'


def create_stats():
    stats = {
        role: { 'total': 0.0, 'propose': 0.0, 'vote': 0.0, 'mission': 0.0, 'merlin': 0.0 }
        for role in ROLES
    }
    stats['all'] = { 'total': 0.0, 'propose': 0.0, 'vote': 0.0, 'mission': 0.0, 'merlin': 0.0 }
    return stats


def load_human_data(cache=True):
    global human_data
    if human_data is not None:
        return human_data
    with open(DATAFILE, 'r') as f:
        result = json.load(f)
    if cache:
        human_data = result
    return result


def compute_human_statistics(bot_class, cache=True, verbose=True, num_players=None, max_num_players=7):
    stats = create_stats()
    if verbose:
        print "Loading human data"
    data = load_human_data(cache=cache)
    for game in data:
        process_game(game, bot_class, stats, verbose=verbose, num_players=num_players, max_num_players=max_num_players)
    return stats


def print_human_statistics(bot_class, stats):
    print "========== Bot statistics for {}".format(bot_class.__name__)
    for role in ['all'] + ROLES:
        print "===== Summary for {}".format(role)
        print "All actions LL: {: >10.04f}".format(stats[role]['total'])
        print "    Propose LL: {: >10.04f}".format(stats[role]['propose'])
        print "       Vote LL: {: >10.04f}".format(stats[role]['vote'])
        print "    Mission LL: {: >10.04f}".format(stats[role]['mission'])
        print "Pick Merlin LL: {: >10.04f}".format(stats[role]['merlin'])

def print_human_statistics_csv(bot_class, stats):
    result = [ bot_class.__name__ ]
    for role in ['all'] + ROLES:
        for category in ['total', 'propose', 'vote', 'mission', 'merlin']:
            result.append(stats[role][category])

    print ','.join(str(x) for x in result)


def print_header_row():
    res = ['bot_class'] + [ "{}_{}_ll".format(role, category) for role in ['all'] + ROLES for category in ['total', 'propose', 'vote', 'mission', 'merlin']]
    print ",".join(res)
