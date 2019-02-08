import os
import gzip
import cPickle as pickle
import json
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking
from sklearn.utils import shuffle

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(PARENT_DIR)

from battlefield.avalon import create_avalon_game
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, ProposeAction, VoteAction, MissionAction, PickMerlinAction, possible_hidden_states, starting_hidden_states

VOTE_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vote_data.npz'))
PROPOSE_OUTPUT_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'propose_data.npz'))


def create_vote_model():
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(None, 79)),
        Dense(120, activation='relu'),
        LSTM(120),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])
    return model



def train():
    print "Loading data"
    data = np.load(VOTE_FILENAME)
    X = data['arr_0']
    y = data['arr_1']
    print "Shuffling data"
    X, y = shuffle(X, y)
    model = create_vote_model()
    model.fit(x=X, y=y, batch_size=128, epochs=100, validation_split=0.1)


if __name__ == "__main__":
    train()
