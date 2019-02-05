from battlefield.avalon import create_avalon_game
from battlefield.bots import RandomBot, RandomBotUV, SimpleBot, ObserveBot, ISMCTSBot, MOISMCTSBot
from battlefield.tournament import run_tournament, print_statistics, check_config

TOURNAMENT_CONFIG = [
    {
        'bot': ObserveBot,
        'role': 'merlin'
    },
    {
        'bot': MOISMCTSBot,
        'role': 'servant'
    },
    {
        'bot': ISMCTSBot,
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

def main():
    check_config(TOURNAMENT_CONFIG)
    tournament_results = run_tournament(TOURNAMENT_CONFIG, num_games=40, granularity=1)
    print_statistics(tournament_results)


if __name__ == "__main__":
    main()
