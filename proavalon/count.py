import json
import glob
import numpy as np
from collections import defaultdict

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def analyze_game(game):
    roles = game['game_info']['roles']
    roles = roles[:1] + roles[1:][::-1]
    res_bots = 0
    spy_bots = 0
    for (player, role) in zip(game['session_info']['players'], roles):
        if 'DeepRole#' not in player:
            continue
        if role in ['Resistance', 'Merlin']:
            res_bots += 1
        else:
            spy_bots += 1

    res_payoff = -1.0 if game['game_info']['winner'] == 'Spy' else 1.0
    spy_payoff = res_payoff * -1.5

    bot_payoff = res_payoff * res_bots + spy_payoff * spy_bots
    human_payoff = -bot_payoff
    total_bots = res_bots + spy_bots
    total_humans = 5 - total_bots

    return (res_bots + spy_bots, bot_payoff/total_bots, human_payoff/total_humans)


def all_equal(l):
    for a in l:
        for b in l:
            if a != b:
                return False
    return True


def load_games():
    games = defaultdict(lambda: {})
    for game_file in glob.glob('results_dir/*.json'):
        with open(game_file) as f:
            g = json.load(f)

            key = (
                tuple(g['session_info']['players']),
                g['game_info']['roomId'],
                tuple(g['game_info']['roles']),
                g['game_info']['missionNum'],
                g['game_info']['teamLeader'],
                tuple(g['game_info']['proposedTeam']),
                g['game_info']['winner'],
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][0]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][1]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][2]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][3]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][4]]),
            )
            games[key][game_file] = g

    games_by_bot_count = defaultdict(lambda: [])
    for _, similar_games in games.items():
        filenames = similar_games.keys()
        game_results = [analyze_game(similar_games[filename]) for filename in filenames]
        assert all_equal(game_results)
        result = game_results[0]
        bot_count, _, _ = result
        games_by_bot_count[bot_count].append(result)
    return games_by_bot_count


def compute_prob(data):
    values, counts = np.unique(data, return_counts=True)
    if 0 in list(counts):
        return float('nan')
    better_counts = 10*np.ones(len(counts)) + counts
    samples = np.random.dirichlet(better_counts, 1000000)
    return np.mean((np.dot(samples, values) > 0.0).astype(np.float))



def calculate_human_statistics():
    result = {}
    game_results = load_games()
    for num_bots in sorted(game_results.keys()):
        _, bot_payoffs, human_payoffs = zip(*game_results[num_bots])
        result[num_bots] = {
            'n_bots': num_bots,
            'n_humans': 5 - num_bots,
            'bot_avg_payoff': sum(bot_payoffs) / len(bot_payoffs),
            'human_avg_payoff': sum(human_payoffs) / len(human_payoffs),
            'n_games': len(game_results[num_bots]),
            'confidence': compute_prob(bot_payoffs)
        }
    return result



def print_statistics():
    all_statistics = calculate_human_statistics()
    print " N_Bots | N_Humans | Bot_payoff | human_payoff | N_games | P(bot_payoff > 0.0) "
    print "==========================================================================================="
    for num_bots, stats in sorted(all_statistics.items()):
        print " {: <6} | {: <8} | {:.08f} | {:.09f} | {: <7} | {}".format(
            stats['n_bots'],
            stats['n_humans'],
            stats['bot_avg_payoff'],
            stats['human_avg_payoff'],
            stats['n_games'],
            stats['confidence']
        )


if __name__ == '__main__':
    print_statistics()

