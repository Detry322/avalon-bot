import numpy as np
import pandas
from collections import defaultdict, deque

import itertools

# If they pass a winning dude, they get payoff 1
# If they pass a failing dude, they get payoff -1
# If the vote doesn't pass, they get payoff 0

NUM_PLAYERS = 7
NUM_BAD = 3
MISSION_SIZE = 5

print "Pre-making lookup table for {} players with {} bad people".format(NUM_PLAYERS, NUM_BAD)
bad_people_combos = list(itertools.combinations(range(NUM_PLAYERS), NUM_BAD))
lookup_table = { combo: i for i, combo in enumerate(bad_people_combos) }

import operator as op
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, xrange(n, n-r, -1), 1)
    denom = reduce(op.mul, xrange(1, r+1), 1)
    return numer//denom


def bad_people_to_index(bad_people):
    global lookup_table
    return lookup_table[bad_people]


def index_to_bad_people(index):
    global bad_people_combos
    return bad_people_combos[index]


def initial_belief(person):
    possible_bad_groups = float(len(bad_people_combos) - ncr(NUM_PLAYERS - 1, NUM_BAD - 1))
    bad_prob = 1.0/possible_bad_groups
    belief = np.zeros(len(bad_people_combos))
    count = 0
    for i, bad_people in enumerate(bad_people_combos):
        if person in bad_people:
            continue
        count += 1
        belief[i] = bad_prob
    assert count == possible_bad_groups
    return belief


def bad_mask(mission):
    return np.array([len(set(mission) & set(bad_people)) > 0 for bad_people in bad_people_combos])


def observe_mission(belief, mission, has_bad):
    mask = bad_mask(mission) if has_bad else (1.0 - bad_mask(mission))
    not_normalized_belief = belief * mask
    return not_normalized_belief / np.sum(not_normalized_belief)


def run_iteration(starter_belief, potential_bad_people):
    bad_people = tuple(sorted(np.random.choice(potential_bad_people, replace=False, size=NUM_BAD)))

    belief = starter_belief
    for mission_number in itertools.count():
        mission = tuple(sorted(np.random.choice(range(NUM_PLAYERS), replace=False, size=MISSION_SIZE)))
        has_bad = len(set(mission) & set(bad_people)) != 0
        belief = observe_mission(belief, mission, has_bad)
        best_guess = np.argmax(belief)
        confidence = belief[best_guess]
        bad_people_guess = index_to_bad_people(best_guess)
        if confidence == 1:
            break

    assert bad_people == bad_people_guess
    return mission_number + 1

def main():
    print("============== Starting new iterated game.")
    print("      NUM_PLAYERS: {}".format(NUM_PLAYERS))
    print("          NUM_BAD: {}".format(NUM_BAD))
    print("     MISSION_SIZE: {}".format(MISSION_SIZE))
    print("=======================")
    perspective = 0
    potential_bad_people = list(set(range(NUM_PLAYERS)) - set([perspective]))

    starter_belief = initial_belief(perspective)

    data = []
    for i in range(1000):
        data.append(run_iteration(starter_belief, potential_bad_people))


    df = pandas.DataFrame({'trials_needed': data})

    print df.describe()


if __name__ == "__main__":
    main()
