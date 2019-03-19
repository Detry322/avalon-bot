from battlefield.bots import (
    RandomBot, RandomBotUV,
    SimpleBot,
    ObserveBot, ExamineAgreementBot,
    ISMCTSBot, MOISMCTSBot,
    HumanBot,
    NNBot, NNBotWithObservePropose,
    SimpleStatsBot,
    SingleMCTSPlayoutBot, SingleMCTSHeuristicBot, SingleMCTSBaseOpponentBot,
    CFRBot,
    LearningBot,
    ObserveBeaterBot
)
from battlefield.tournament import (
    run_simple_tournament,
    run_large_tournament,
    run_all_combos_parallel,
    print_tournament_statistics,
    check_config,
    run_learning_tournament,
    run_single_threaded_tournament
)
from battlefield.compare_to_human import compute_human_statistics, print_human_statistics
from battlefield.predict_roles import predict_evil_over_human_data, predict_evil_using_voting, grid_search
from battlefield.determine_reachable_states import determine_reachable
import multiprocessing
import pandas as pd
import sys
import gzip
import cPickle as pickle
from collections import defaultdict

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
        'bot': ObserveBeaterBot,
        'role': 'servant'
    },
    {
        'bot': ObserveBot,
        'role': 'minion'
    }
]

def tournament():
    check_config(TOURNAMENT_CONFIG)
    print "hidden", tuple(x['role'] for x in TOURNAMENT_CONFIG)
    tournament_results = run_single_threaded_tournament(TOURNAMENT_CONFIG, num_games=2000, granularity=200)
    print_tournament_statistics(tournament_results)


def join_and_write(results):
    result_by_bot = defaultdict(lambda: [])
    for result in results:
        bot_name = result.loc[0].bot
        result_by_bot[bot_name].append(result)

    for bot, results in result_by_bot.items():
        print "Saving", bot
        with gzip.open('human/{}.msg.gz'.format(bot), 'w') as f:
            data = pd.concat(results)
            data.reset_index(drop=True, inplace=True)
            data.to_msgpack(f)


def parallel_human_compare():
    pool = multiprocessing.Pool()
    try:
        results = []
        for bot in [CFRBot(4000000)]:
            for tremble in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
                results.append(pool.apply_async(compute_human_statistics, (bot, tremble, False, 5)))

        print "Waiting for results"
        sys.stdout.flush()
        results = [ result.get() for result in results ]
        print "Concatenating and writing results"
        sys.stdout.flush()
        join_and_write(results)
    except KeyboardInterrupt:
        print 'terminating early'
        pool.terminate()


def human_compare():
    stats = compute_human_statistics(CFRBot(1000000), verbose=False, num_players=5)
    print_human_statistics(stats)


STRING_TO_BOT = {
    "RandomBot": RandomBot,
    "RandomBotUV": RandomBotUV,
    "SimpleBot": SimpleBot,
    "ObserveBot": ObserveBot,
    "SimpleStatsBot": SimpleStatsBot,
    "CFRBot_3000000": CFRBot(3000000),
    "CFRBot_6000000": CFRBot(6000000),
    "CFRBot_10000000": CFRBot(10000000)
}

def predict_roles(bot, tremble):
    bot = STRING_TO_BOT[bot]
    tremble = float(tremble)
    dataframe, particles = predict_evil_over_human_data(bot, tremble)
    print "Writing dataframe..."
    with gzip.open('predict_roles/{}_{}_df.msg.gz'.format(bot.__name__, tremble), 'w') as f:
        dataframe.to_msgpack(f)

    print "Writing particles..."
    with gzip.open('predict_roles/{}_{}_particles.pkl.gz'.format(bot.__name__, tremble), 'w') as f:
        pickle.dump(particles, f)


if __name__ == "__main__":
    # bots = [ ObserveBeaterBot, ObserveBot, ObserveBot, ObserveBot, ObserveBot ]
    # run_learning_tournament(bots, winrate_track=0)
    # grid_search()
    # predict_evil_using_voting()
    tournament()
   # determine_reachable(RandomBot, set(['merlin', 'minion', 'assassin', 'servant']), 5)
