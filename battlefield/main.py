from battlefield.avalon import create_avalon_game
from battlefield.bots import RandomBot, RandomBotUV, SimpleBot, ObserveBot, ISMCTSBot, MOISMCTSBot, HumanBot, NNBot, NNBotWithObservePropose
from battlefield.tournament import run_tournament, print_tournament_statistics, check_config
from battlefield.compare_to_human import compute_human_statistics, print_human_statistics, print_header_row, print_human_statistics_csv

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


def human_compare():
    bots = [ NNBot, NNBotWithObservePropose ]
    print_header_row()
    for bot in bots:
        stats = compute_human_statistics(bot, verbose=False, num_players=5)
        print_human_statistics_csv(bot, stats)


if __name__ == "__main__":
    human_compare()
