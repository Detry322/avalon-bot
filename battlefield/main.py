from battlefield.avalon import create_avalon_game
from battlefield.bots import RandomBot, RandomBotUV, SimpleBot, ObserveBot, ISMCTSBot, MOISMCTSBot, HumanBot
from battlefield.tournament import run_tournament, print_statistics, check_config

TOURNAMENT_CONFIG = [
    {
        'bot': HumanBot,
        'role': 'merlin'
    },
    {
        'bot': HumanBot,
        'role': 'servant'
    },
    {
        'bot': RandomBot,
        'role': 'assassin'
    },
    {
        'bot': HumanBot,
        'role': 'servant'
    },
    {
        'bot': RandomBot,
        'role': 'minion'
    }
]

def main():
    check_config(TOURNAMENT_CONFIG)
    tournament_results = run_tournament(TOURNAMENT_CONFIG, num_games=1000, granularity=100)
    print_statistics(tournament_results)


if __name__ == "__main__":
    main()
