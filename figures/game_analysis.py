import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import functools
from scipy.special import comb

# Wellman stuff
def filter_bot_v_bot_games(all_games, acceptable_bots):
    for i in range(5):
        if i == 0:
            selector = all_games['bot_{}'.format(i)].isin(acceptable_bots)
        else:
            selector &= all_games['bot_{}'.format(i)].isin(acceptable_bots)
    return all_games[selector]


def _group_by_bot(unchunked_games, bots, chunksize=100000):
    results = []
    for i in range(0, len(unchunked_games), chunksize):
        games = unchunked_games[i:i+chunksize]
        new_dataframe = {}
        for bot in bots:
            for i in range(5):
                selector = games['bot_{}'.format(i)] == bot
                if i == 0:
                    bot_count = selector.astype(int)
                    payoff = games['bot_{}_payoff'.format(i)][selector]
                else:
                    bot_count = bot_count.add(selector.astype(int), fill_value=0.0)
                    payoff = payoff.add(games['bot_{}_payoff'.format(i)][selector], fill_value=0.0)
            new_dataframe['{}_count'.format(bot)] = bot_count
            new_dataframe['{}_payoff'.format(bot)] = payoff.divide(bot_count)
        results.append(pd.DataFrame(new_dataframe))
    return pd.concat(results)


def compute_payoff_matrix(games, bots):
    grouped = _group_by_bot(games, bots, 100000)
    payoff = grouped.groupby(['{}_count'.format(bot) for bot in bots]).mean()
    return payoff[['{}_payoff'.format(bot) for bot in bots]].fillna(0.0)


def scipy_multinomial(params):
    if len(params) == 1:
        return 1
    coeff = (comb(np.sum(params), params[-1], exact=True) *
             scipy_multinomial(params[:-1]))
    return coeff


def P(N_i, x):
    x = np.array(x)
    N_i = np.array(N_i)
    return scipy_multinomial(N_i) * np.prod( x ** N_i )


def r(x, table):
    x = np.array(x)
    numerator = np.zeros(len(x))
    for index, payoff in table.iterrows():
        numerator += P(index, x) * np.array(payoff)
    denominator = 1.0 - (1.0 - x) ** 5
    return np.nan_to_num(numerator / denominator)


def calc_xdot(x, table):
    rx = r(x, table)
    xtAx = np.sum(x * rx)
    xdot = x * (rx - xtAx)
    return xdot


LEARNING_RATE = 0.1
def calc_new_strat(x, table):
    return x + LEARNING_RATE*calc_xdot(x, table)


def find_nash(table, num_bots=4, start=None, iters=100000, freq=300):
    if start is not None:
        x = start[:]
    else:
        x = np.ones(num_bots) / num_bots
    for i in range(iters):
        x += LEARNING_RATE*calc_xdot(x, table)
        if i % freq == 0:
            print x


# Confidence interval stuff

from scipy.special import betaln
def confidence_b_gr_a(pos_a, neg_a, pos_b, neg_b):
    alpha_a = pos_a + 1
    beta_a = neg_a + 1
    alpha_b = pos_b + 1
    beta_b = neg_b + 1
    
    total = 0.0
    for i in range(alpha_b):
        total += np.exp(
            betaln(alpha_a + i, beta_b + beta_a)
            - np.log(beta_b + i)
            - betaln(1 + i, beta_b)
            - betaln(alpha_a, beta_a)
        )
    return total


def compare_humans_and_bots(df, fields):
    human_stats = df[df.is_bot == False].groupby(fields + ['num_bots', 'win']).count().game
    bot_stats = df[df.is_bot == True].groupby(fields + ['num_bots', 'win']).count().game
    
    results = []
    for idx in human_stats.index:
        if not idx[-1]:
            continue
        num_bots = idx[-2]
        idx = idx[:-2]
        human_win = human_stats.loc[idx + (num_bots, True)]
        human_loss = human_stats.loc[idx + (num_bots, False)]
        bot_win = bot_stats.loc[idx + (num_bots + 1, True)]
        bot_loss = bot_stats.loc[idx + (num_bots + 1, False)]
        new_dict = {
            'num_other_bots': num_bots,
            'bot_winrate': bot_win / float(bot_win + bot_loss),
            'bot_n': bot_win + bot_loss,
            'human_winrate': human_win / float(human_win + human_loss),
            'human_n': human_win + human_loss,
            'bot_better_confidence': confidence_b_gr_a(human_win, human_loss, bot_win, bot_loss)
        }
        for k, v in zip(fields, idx):
            new_dict[k] = v
        results.append(new_dict)
    
    result = pd.DataFrame(results, columns=['num_other_bots'] + fields + ['bot_winrate', 'human_winrate', 'bot_better_confidence', 'bot_n', 'human_n'])
    return result.groupby(['num_other_bots'] + fields).mean()
