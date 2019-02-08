from battlefield.avalon import create_avalon_game
from battlefield.bots import RandomBot, RandomBotUV, SimpleBot, ObserveBot, ISMCTSBot, MOISMCTSBot, HumanBot
from battlefield.tournament import run_tournament, print_tournament_statistics, check_config
from battlefield.compare_to_human import compute_human_statistics, print_human_statistics, print_header_row, print_human_statistics_csv

TOURNAMENT_CONFIG = [
    {
        'bot': RandomBotUV,
        'role': 'merlin'
    },
    {
        'bot': RandomBotUV,
        'role': 'servant'
    },
    {
        'bot': MOISMCTSBot,
        'role': 'assassin'
    },
    {
        'bot': RandomBotUV,
        'role': 'servant'
    },
    {
        'bot': MOISMCTSBot,
        'role': 'minion'
    }
]

def tournament():
    check_config(TOURNAMENT_CONFIG)
    tournament_results = run_tournament(TOURNAMENT_CONFIG, num_games=40, granularity=1)
    print_tournament_statistics(tournament_results)


def human_compare():
    bots = [ RandomBot, RandomBotUV, SimpleBot, ObserveBot ]
    print_header_row()
    for bot in bots:
        stats = compute_human_statistics(bot, verbose=False)
        print_human_statistics_csv(bot, stats)


if __name__ == "__main__":
    human_compare()
