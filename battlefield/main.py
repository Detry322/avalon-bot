from battlefield.avalon import create_avalon_game
from battlefield.bots import RandomBot, RandomBotUV, SimpleBot, ObserveBot, ISMCTSBot, MOISMCTSBot, HumanBot, NNBot, NNBotWithObservePropose
from battlefield.tournament import run_tournament, print_tournament_statistics, check_config
from battlefield.compare_to_human import compute_human_statistics, print_human_statistics
import multiprocessing
import pandas as pd
import sys

TOURNAMENT_CONFIG = [
    {
        'bot': ObserveBot,
        'role': 'merlin'
    },
    {
        'bot': ObserveBot,
        'role': 'servant'
    },
    {
        'bot': ObserveBot,
        'role': 'assassin'
    },
    {
        'bot': ObserveBot,
        'role': 'servant'
    },
    {
        'bot': ObserveBot,
        'role': 'minion'
    }
]

def tournament():
    check_config(TOURNAMENT_CONFIG)
    tournament_results = run_tournament(TOURNAMENT_CONFIG, num_games=100, granularity=10)
    print_tournament_statistics(tournament_results)


def parallel_human_compare():
    pool = multiprocessing.Pool()
    try:
        results = []
        for bot in [ObserveBot]:
            for tremble in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results.append(pool.apply_async(compute_human_statistics, (bot, tremble, False, 5)))

        print "Waiting for results"
        sys.stdout.flush()
        results = [ result.get() for result in results ]
        print "Concatenating results"
        sys.stdout.flush()
        data = pd.concat(results)
        print "Saving results"
        sys.stdout.flush()
        data.to_pickle("human_comparison_data.pkl.gz")
    except KeyboardInterrupt:
        print 'terminating early'
        pool.terminate()


def human_compare():
    stats = compute_human_statistics(NNBot, verbose=False, num_players=5)
    print_human_statistics(stats)


if __name__ == "__main__":
    parallel_human_compare()
