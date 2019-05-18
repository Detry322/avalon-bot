import numpy as np
import pandas as pd
import gzip
import glob
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import logging
import multiprocessing


BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

TOPOLOGICALLY_SORTED_LAYERS = [
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4),
    (0, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3), (1, 0, 4),
    (0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 1, 4),
    (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4),
    (0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 2, 3), (0, 2, 4),
    (2, 0, 0), (2, 0, 1), (2, 0, 2), (2, 0, 3), (2, 0, 4),
    (2, 1, 0), (2, 1, 1), (2, 1, 2), (2, 1, 3), (2, 1, 4),
    (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4),
    (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4),
]

def load_tensorboard_loss_data(subfolder, num_succeeds, num_fails, propose_count):
    logging.disable(logging.CRITICAL)

    event_acc = EventAccumulator(os.path.join(
        BASE_DIR,
        'tensorboard',
        subfolder,
        '{}_{}_{}'.format(num_succeeds, num_fails, propose_count)
    ))
    event_acc.Reload()
    _, _, loss = zip(*event_acc.Scalars('epoch_val_loss'))
    logging.disable(logging.NOTSET)
    return loss


def _load_tensorboard_final_loss(args):
    loss = load_tensorboard_loss_data(args[0], *args[1])
    return {
        'index': TOPOLOGICALLY_SORTED_LAYERS.index(args[1]),
        'part': args[1],
        'loss': loss[-1]
    }


def load_tensorboard_final_losses(subfolder):
    logging.disable(logging.CRITICAL)
    data = []

    pool = multiprocessing.Pool(16)
    work_args = zip([subfolder]*len(TOPOLOGICALLY_SORTED_LAYERS), TOPOLOGICALLY_SORTED_LAYERS)
    results = pool.map(_load_tensorboard_final_loss, work_args)
    
    return pd.DataFrame(results)


ALL_GAMES = None
def load_bot_v_bot_games():
    global ALL_GAMES
    if ALL_GAMES is None:
        with gzip.open(os.path.join(BASE_DIR, 'bot_v_bot_games.msg.gz'), 'r') as f:
            ALL_GAMES = pd.read_msgpack(f)
    return ALL_GAMES


HUMAN_GAMES = None
def load_human_v_bot_games():
    global HUMAN_GAMES
    if HUMAN_GAMES is None:
        HUMAN_GAMES = pd.read_csv(os.path.join(BASE_DIR, 'human_v_bots_data.csv'))
    return HUMAN_GAMES
