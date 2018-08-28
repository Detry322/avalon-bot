from collections import defaultdict

GAMES = []

PROPOSAL_1 = 0
PROPOSAL_1_SUCCEED = 1
PROPOSAL_2 = 2
PROPOSAL_2_SUCCEED = 3
PROPOSAL_3 = 4
PROPOSAL_3_SUCCEED = 5

PROPOSALS = [(0, 1), (0, 2), (1, 2)]

for bad in range(3):
  for proposal1 in PROPOSALS:
    for succeed1 in [True, False]:
      for proposal2 in PROPOSALS:
        for succeed2 in [True, False]:
          for proposal3 in PROPOSALS:
            for succeed3 in [True, False]:
              if not succeed1 and bad not in proposal1:
                continue
              if not succeed2 and bad not in proposal2:
                continue
              if not succeed3 and bad not in proposal3:
                continue
              GAMES.append((bad, (proposal1, succeed1, proposal2, succeed2, proposal3, succeed3)))


def create_p1_strategies():
    bad_strategies = []
    good_strategies = []
    for proposal in PROPOSALS:
        for succeed1 in [True, False]:
            for succeed2 in [True, False]:
                for succeed3 in [True, False]:
                    strategy = {}
                    for _, game in GAMES:
                        strategy[game[:0]] = proposal
                        if 0 in game[0]:
                            strategy[game[:1]] = succeed1
                        if 0 in game[2]:
                            strategy[game[:3]] = succeed2
                        if 0 in game[4]:
                            strategy[game[:5]] = succeed3
                    bad_strategies.append(strategy)
    for proposal in PROPOSALS:
        for succeed1 in [True]:
            for succeed2 in [True]:
                for succeed3 in [True]:
                    strategy = {}
                    for _, game in GAMES:
                        strategy[game[:0]] = proposal
                        if 0 in game[0]:
                            strategy[game[:1]] = succeed1
                        if 0 in game[2]:
                            strategy[game[:3]] = succeed2
                        if 0 in game[4]:
                            strategy[game[:5]] = succeed3
                    good_strategies.append(strategy)
    return good_strategies, bad_strategies


def create_p2_strategies():
    bad_strategies = []
    good_strategies = []
    for proposal in PROPOSALS:
        for succeed1 in [True, False]:
            for succeed2 in [True, False]:
                for succeed3 in [True, False]:
                    strategy = {}
                    for _, game in GAMES:
                        strategy[game[:2]] = proposal
                        if 1 in game[0]:
                            strategy[game[:1]] = succeed1
                        if 1 in game[2]:
                            strategy[game[:3]] = succeed2
                        if 1 in game[4]:
                            strategy[game[:5]] = succeed3
                    bad_strategies.append(strategy)
    for proposal in PROPOSALS:
        for succeed1 in [True]:
            for succeed2 in [True]:
                for succeed3 in [True]:
                    strategy = {}
                    for _, game in GAMES:
                        strategy[game[:2]] = proposal
                        if 1 in game[0]:
                            strategy[game[:1]] = succeed1
                        if 1 in game[2]:
                            strategy[game[:3]] = succeed2
                        if 1 in game[4]:
                            strategy[game[:5]] = succeed3
                    good_strategies.append(strategy)
    return good_strategies, bad_strategies


def create_p3_strategies():
    bad_strategies = []
    good_strategies = []
    for proposal in PROPOSALS:
        for succeed1 in [True, False]:
            for succeed2 in [True, False]:
                for succeed3 in [True, False]:
                    strategy = {}
                    for _, game in GAMES:
                        strategy[game[:4]] = proposal
                        if 2 in game[0]:
                            strategy[game[:1]] = succeed1
                        if 2 in game[2]:
                            strategy[game[:3]] = succeed2
                        if 2 in game[4]:
                            strategy[game[:5]] = succeed3
                    bad_strategies.append(strategy)
    for proposal in PROPOSALS:
        for succeed1 in [True]:
            for succeed2 in [True]:
                for succeed3 in [True]:
                    strategy = {}
                    for _, game in GAMES:
                        strategy[game[:4]] = proposal
                        if 2 in game[0]:
                            strategy[game[:1]] = succeed1
                        if 2 in game[2]:
                            strategy[game[:3]] = succeed2
                        if 2 in game[4]:
                            strategy[game[:5]] = succeed3
                    good_strategies.append(strategy)
    return good_strategies, bad_strategies


STRATEGIES = { 0: {}, 1: {}, 2: {} }
STRATEGIES[0]['good'], STRATEGIES[0]['bad'] = create_p1_strategies()
STRATEGIES[1]['good'], STRATEGIES[1]['bad'] = create_p2_strategies()
STRATEGIES[2]['good'], STRATEGIES[2]['bad'] = create_p3_strategies()

def play_game(p1_strategy, p2_strategy, p3_strategy):
    strats = { 0: p1_strategy, 1: p2_strategy, 2: p3_strategy }
    game_state = ()
    proposed = p1_strategy[game_state]
    game_state += (proposed,)
    pass_or_fail = strats[proposed[0]][game_state] and strats[proposed[1]][game_state]
    game_state += (pass_or_fail,)
    proposed = p2_strategy[game_state]
    game_state += (proposed,)
    pass_or_fail = strats[proposed[0]][game_state] and strats[proposed[1]][game_state]
    game_state += (pass_or_fail,)
    proposed = p3_strategy[game_state]
    game_state += (proposed,)
    pass_or_fail = strats[proposed[0]][game_state] and strats[proposed[1]][game_state]
    game_state += (pass_or_fail,)
    return (game_state, (int(game_state[PROPOSAL_1_SUCCEED]) + int(game_state[PROPOSAL_2_SUCCEED]) + int(game_state[PROPOSAL_3_SUCCEED])) >= 2)


def replace_strat(new_strat, p1_strategy, p2_strategy, p3_strategy, select):
    args = {
        0: (new_strat, p2_strategy, p3_strategy),
        1: (p1_strategy, new_strat, p3_strategy),
        2: (p1_strategy, p2_strategy, new_strat)
    }
    return args[select]

def is_equilibrium(p1_strategy, p2_strategy, p3_strategy, bad):
    p1_strategy_set = STRATEGIES[0]['bad' if bad == 0 else 'good']
    p2_strategy_set = STRATEGIES[1]['bad' if bad == 1 else 'good']
    p3_strategy_set = STRATEGIES[2]['bad' if bad == 2 else 'good']
    _, good_win = play_game(p1_strategy, p2_strategy, p3_strategy)

    bad_strategy_set = STRATEGIES[bad]['bad']
    good1, good2 = sorted({0, 1, 2} - {bad})
    good_strategy_set1 = STRATEGIES[good1]['good']
    good_strategy_set2 = STRATEGIES[good2]['good']

    strat_selector = { 0: p1_strategy, 1: p2_strategy, 2: p3_strategy }

    if good_win:
        bad_strategy = strat_selector[bad]
        for bad_strat in bad_strategy_set:
            if bad_strat is bad_strategy:
                continue
            args = replace_strat(bad_strat, p1_strategy, p2_strategy, p3_strategy, bad)
            _, good_win = play_game(*args)
            if not good_win:
                return False
    else:
        good_strategy1 = strat_selector[good1]
        good_strategy2 = strat_selector[good2]
        for good_strat1 in good_strategy_set1:
            if good_strat1 is good_strategy1:
                continue
            args = replace_strat(good_strat1, p1_strategy, p2_strategy, p3_strategy, good1)
            _, good_win = play_game(*args)
            if good_win:
                return False
        for good_strat2 in good_strategy_set2:
            if good_strat2 is good_strategy2:
                continue
            args = replace_strat(good_strat2, p1_strategy, p2_strategy, p3_strategy, good2)
            _, good_win = play_game(*args)
            if good_win:
                return False
    return True


def find_equilibria():
    equilibria = defaultdict(lambda: [])
    for bad in range(3):
        p1_strategy_set = STRATEGIES[0]['bad' if bad == 0 else 'good']
        p2_strategy_set = STRATEGIES[1]['bad' if bad == 1 else 'good']
        p3_strategy_set = STRATEGIES[2]['bad' if bad == 2 else 'good']
        for p1_strategy in p1_strategy_set:
            for p2_strategy in p2_strategy_set:
                for p3_strategy in p3_strategy_set:
                    game_outcome, good_win = play_game(p1_strategy, p2_strategy, p3_strategy)
                    if is_equilibrium(p1_strategy, p2_strategy, p3_strategy, bad):
                        key = (bad, good_win, game_outcome)
                        equilibria[key].append((p1_strategy, p2_strategy, p3_strategy))
    return equilibria


results = find_equilibria()
for key in results:
    print(key)
