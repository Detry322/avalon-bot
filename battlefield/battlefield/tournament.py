from collections import defaultdict

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES
from battlefield.avalon import create_avalon_game, possible_hidden_states, starting_hidden_states

def run_game(game, bots):
    return True


def run_tournament(config, num_games=1000):
    tournament_statistics = [
        { 'bot': bot['bot'].__name__, 'role': bot['role'], 'wins': 0, 'total': 0, 'win_percent': 0 }
        for bot in config
    ]

    game = create_avalon_game(num_players=len(config))
    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]

    for i in range(num_games):
        if i % 100 == 0:
            print "Running game {}".format(i)
        bots = [
            bot['bot'](game, player, bot['role'], beliefs[player])
            for player, bot in enumerate(config)
        ]
        good_win = run_game(game, bots)
        for b in tournament_statistics:
            b['wins'] += 1 if not (good_win ^ (b['role'] in GOOD_ROLES)) else 0
            b['total'] += 1

    for b in tournament_statistics:
        b['win_percent'] = float(b['wins'])/float(b['total'])

    return tournament_statistics


def check_config(config):
    role_counts = defaultdict(lambda: 0)

    game = create_avalon_game(num_players=len(config))

    for bot in config:
        # Count roles
        role_counts[bot['role']] += 1

    assert 'merlin' in role_counts
    assert 'assassin' in role_counts
    for role, count in role_counts.items():
        assert role == 'servant' or role == 'minion' or count == 1
        assert role in GOOD_ROLES or role in EVIL_ROLES


def print_statistics(tournament_statistics):
    print "       Role |            Bot |      Evil |      Winrate "
    print "--------------------------------------------------------"
    for bot in tournament_statistics:
        print "{: >11} | {: >14} | {: >9} | {: >12.04f}".format(bot['role'], bot['bot'], 'Yes' if bot['role'] in EVIL_ROLES else '', 100*bot['win_percent'])
