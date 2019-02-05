from collections import defaultdict

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states
from battlefield.avalon import create_avalon_game

def run_game(state, hidden_state, bots):
    while not state.is_terminal():
        moving_players = state.moving_players()
        moves = [
            bots[player].get_action(state, state.legal_actions(player, hidden_state))
            for player in moving_players
        ]
        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)
        state = new_state
    return state.terminal_value(hidden_state), state.game_end


def run_tournament(config, num_games=1000, granularity=100):
    tournament_statistics = {
        'bots': [
            { 'bot': bot['bot'].__name__, 'role': bot['role'], 'wins': 0, 'total': 0, 'win_percent': 0, 'payoff': 0.0 }
            for bot in config
        ],
        'end_types': {}
    }

    game = create_avalon_game(num_players=len(config))
    start_state = game.start_state()
    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]

    for i in range(num_games):
        if i % granularity == 0:
            print "Running game {}".format(i)
        bots = [
            bot['bot'](start_state, player, bot['role'], beliefs[player])
            for player, bot in enumerate(config)
        ]
        payoffs, end_type = run_game(start_state, hidden_state, bots)

        tournament_statistics['end_types'][end_type] = 1 + tournament_statistics['end_types'].get(end_type, 0)

        for b, payoff in zip(tournament_statistics['bots'], payoffs):
            b['wins'] += 1 if payoff > 0.0 else 0
            b['payoff'] += payoff
            b['total'] += 1

    for b in tournament_statistics['bots']:
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
    print "       Role |            Bot |      Evil |      Winrate |        Payoff "
    print "------------------------------------------------------------------------"
    for bot in tournament_statistics['bots']:
        print "{: >11} | {: >14} | {: >9} | {: >11.02f}% | {: >13.02f} ".format(bot['role'], bot['bot'], 'Yes' if bot['role'] in EVIL_ROLES else '', 100*bot['win_percent'], bot['payoff'])

    for game_end, count in sorted(tournament_statistics['end_types'].items(), key=lambda x: x[1], reverse=True):
        print "{}: {} - {}".format(count, game_end[0], game_end[1])

