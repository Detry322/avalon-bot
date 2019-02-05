from battlefield.avalon import create_avalon_game
from battlefield.bots import RandomBot, SimpleBot
from battlefield.tournament import run_tournament, print_statistics, check_config

TOURNAMENT_CONFIG = [
    {
        'bot': SimpleBot,
        'role': 'merlin'
    },
    {
        'bot': SimpleBot,
        'role': 'servant'
    },
    {
        'bot': RandomBot,
        'role': 'assassin'
    },
    {
        'bot': SimpleBot,
        'role': 'servant'
    },
    {
        'bot': RandomBot,
        'role': 'minion'
    }
]

def main():
    check_config(TOURNAMENT_CONFIG)
    tournament_results = run_tournament(TOURNAMENT_CONFIG, num_games=1000)
    print_statistics(tournament_results)


if __name__ == "__main__":
    main()
