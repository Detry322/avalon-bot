import numpy as np
import pandas
from nltk.probability import DictionaryProbDist
import random
import itertools
from collections import defaultdict, deque

from util import ncr

NUM_PLAYERS = 5
NUM_BAD = 1
NUM_GOOD = 3
MANY = False
MANY_COUNT = 1000

print "Pre-making lookup table for {} players with {} bad people and {} good people".format(NUM_PLAYERS, NUM_BAD, NUM_GOOD)
possible_bad = list(itertools.combinations(range(NUM_PLAYERS), NUM_BAD))
possible_good = list(itertools.combinations(range(NUM_PLAYERS), NUM_GOOD))
possible_beliefs = [
    (good, bad) for good, bad in itertools.product(possible_good, possible_bad)
    if len(set(good) & set(bad)) == 0
]

lookup_table = { combo: i for i, combo in enumerate(possible_beliefs) }

def belief_to_index(belief):
    global lookup_table
    return lookup_table[belief]


def index_to_belief(index):
    global possible_beliefs
    return possible_beliefs[index]


def get_type(player, (good, bad)):
    if player in good:
        return 'good'
    elif player in bad:
        return 'bad'
    else:
        return 'lemming'


def initial_belief(player, player_type):
    belief = np.zeros(len(possible_beliefs))
    indices = []
    if player_type == 'good':
        for i, (good, bad) in enumerate(possible_beliefs):
            if player in good:
                indices.append(i)
    elif player_type == 'bad':
        for i, (good, bad) in enumerate(possible_beliefs):
            if player in bad:
                indices.append(i)
    else:
        for i, (good, bad) in enumerate(possible_beliefs):
            if (player not in good) and (player not in bad):
                indices.append(i)

    belief[indices] = 1.0/len(indices)
    return belief


def good_strategy(good_outcome):
    return DictionaryProbDist({ good_outcome: 0.51, (not good_outcome): 0.49 })

def bad_strategy(good_outcome):
    return DictionaryProbDist({ good_outcome: 0.49, (not good_outcome): 0.51 })

def lemming_strategy(good_outcome):
    return DictionaryProbDist({ True: 0.5, False: 0.5 })

STRATEGIES = {'good': good_strategy, 'bad': bad_strategy, 'lemming': lemming_strategy }


def update_belief(belief, outcome, vote_results):
    new_belief = np.zeros(len(belief))
    for index, (good, bad) in enumerate(possible_beliefs):
        probability = 1.0
        for player, vote in enumerate(vote_results):
            probability *= STRATEGIES[get_type(player, (good, bad))](outcome).prob(vote)
        new_belief[index] = belief[index]*probability

    return new_belief / np.sum(new_belief)


def run_iteration(players, beliefs):
    outcome = random.choice([True, False])
    votes = [STRATEGIES[player](outcome).generate() for player in players]
    return [update_belief(belief, outcome, votes) for belief in beliefs]


def pp(message):
    if not MANY:
        print message

def ppm(message):
    if MANY:
        print message

def print_belief(players, beliefs):
    confidences = []
    for i, (player_type, belief) in enumerate(zip(players, beliefs)):
        top_belief = np.argmax(belief)
        confidence = belief[top_belief]
        confidences.append(confidence)
        good, bad = index_to_belief(top_belief)
        pp("Player {} {: <9}: CONFIDENCE: {:0.3f}, GOOD: {}, BAD: {}".format(i, "({})".format(player_type), confidence, good, bad))
    return min(confidences)


def run_big_iteration():
    correct_belief = random.choice(possible_beliefs)
    players = [get_type(player, correct_belief) for player in range(NUM_PLAYERS)]
    player_beliefs = [initial_belief(player, player_type) for player, player_type in enumerate(players)]
    print_belief(players, player_beliefs)
    for i in itertools.count():
        pp("============== Round {}".format(i))
        player_beliefs = run_iteration(players, player_beliefs)
        lowest_confidence = print_belief(players, player_beliefs)
        if lowest_confidence > 0.99:
            break
    return i + 1

def main():
    print("============== Starting voting only game.")
    print("      NUM_PLAYERS: {}".format(NUM_PLAYERS))
    print("          NUM_BAD: {}".format(NUM_BAD))
    print("         NUM_GOOD: {}".format(NUM_GOOD))
    ppm(  "       MANY_COUNT: {}".format(MANY_COUNT))
    print("=======================")

    if not MANY:
        run_big_iteration()
        return

    data = []
    for i in range(MANY_COUNT):
        data.append(run_big_iteration())


    df = pandas.DataFrame({'trials_needed': data})

    print df.describe()



if __name__ == "__main__":
    main()

