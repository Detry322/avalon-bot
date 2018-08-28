import numpy as np
from collections import defaultdict, deque

# If they pass a winning dude, they get payoff 1
# If they pass a failing dude, they get payoff -1
# If the vote doesn't pass, they get payoff 0

NUM_PLAYERS = 9
NUM_BAD = 2
MISSION_SIZE = 5

MISSION_PASSED = 1
MISSION_FAILED = -1
MISSION_SKIPPED = 0

def update(belief, is_bad):
    belief = np.clip(belief, 1e-20, 1.0 - 1e-15)
    undo = np.log(belief/(1.0 - belief))
    undo += is_bad - 0.5
    result = 1.0/(1 + np.exp(-undo))
    return result/np.sum(result)



class RunningAverage(object):
    def __init__(self, k=1000):
        self.values = defaultdict(lambda: 0)
        self.k = k
        self.last_k = deque([])

    def add(self, key):
        self.last_k.append(key)
        self.values[key] += 1
        if len(self.last_k) >= self.k:
            other = self.last_k.popleft()
            self.values[other] -= 1


    def average(self, key):
        return self.values[key]/float(len(self.last_k))



class LevelKGood(object):
    def __init__(self, me, k):
        self.me = me
        self.k = k
        self.belief = np.array([ 0.0 if i == me else 1/float(NUM_PLAYERS - 1) for i in range(NUM_PLAYERS)])

    def vote(self, mission):
        bad_guys_guess = set(np.random.choice(range(NUM_PLAYERS), replace=False, size=NUM_BAD, p=self.belief))
        return len(mission & bad_guys_guess) == 0


    def passed_belief_update(self, votes):
        votes = np.array(votes) ^ 1
        self.belief = update(self.belief, votes)

    def failed_belief_update(self, votes):
        votes = np.array(votes) ^ 0
        self.belief = update(self.belief, votes)

    def belief_update(self, mission, votes, result):
        if result == MISSION_SKIPPED:
            return
        if result == MISSION_PASSED:
            self.passed_belief_update(votes)
        else:
            self.failed_belief_update(votes)


    def has_figured_out(self):
        return sum(self.belief > 1e-5) == NUM_BAD



class LevelKBad(object):
    def __init__(self, me, k, bad_guys):
        self.me = me
        self.k = k
        self.bad_guys = bad_guys
        self.beliefs = []
        for i in range(NUM_PLAYERS):
            self.beliefs.append([ 0.0 if i == j else 1/float(NUM_PLAYERS - 1) for j in range(NUM_PLAYERS)])
        self.beliefs = np.array(self.beliefs)

    def vote(self, mission):
        return len(mission & self.bad_guys) != 0

    def belief_update(self, mission, votes, result):
        if result == MISSION_SKIPPED:
            return
        if result == MISSION_PASSED:
            xor = 1
        else:
            xor = 0
        votes = np.array(votes) ^ xor
        for i in range(len(self.beliefs)):
            self.beliefs[i] = update(self.beliefs[i], votes)

    def has_figured_out(self):
        return True


def did_vote_pass(votes):
    return sum(votes) >= (NUM_PLAYERS+1)/2


def play_game(players, mission):
    return [player.vote(mission) for player in players]


def calc_payoff(mission, vote_success, bad_guys):
    if not vote_success:
        return MISSION_SKIPPED
    else:
        if len(mission & bad_guys) == 0:
            return MISSION_PASSED
        else:
            return MISSION_FAILED


def iteration(players, bad_guys):
    mission = set(np.random.choice(range(NUM_PLAYERS), replace=False, size=MISSION_SIZE))
    votes = play_game(players, mission)
    vote_success = did_vote_pass(votes)
    payoff = calc_payoff(mission, vote_success, bad_guys)

    for player in players:
        player.belief_update(mission, votes, payoff)

    return payoff





def main():
    print("============== Starting new iterated game.")
    print("      NUM_PLAYERS: {}".format(NUM_PLAYERS))
    print("          NUM_BAD: {}".format(NUM_BAD))
    print("     MISSION_SIZE: {}".format(MISSION_SIZE))
    print("=======================")
    bad_guys = {7, 8}#set(np.random.choice(range(NUM_PLAYERS), replace=False, size=NUM_BAD))
    players = [LevelKBad(me=i, k=0, bad_guys=bad_guys) if i in bad_guys else LevelKGood(me=i, k=0) for i in range(NUM_PLAYERS)]
    average_payoff = 0
    counts = RunningAverage(k=10)
    i = 1
    while True:
        payoff = iteration(players, bad_guys)
        counts.add(payoff)
        average_payoff = (i-1)/float(i)*average_payoff + 1/float(i)*payoff
        if i % 10 == 0:
            figured_out = sum([player.has_figured_out() for player in players])
            print("{: 9d}: {: >8.4f}, PASS: {: >6.2f}%, FAIL: {: >6.2f}%, SKIP: {: >6.2f}%, FIGURED_OUT: {}".format(i, average_payoff, counts.average(MISSION_PASSED)*100, counts.average(MISSION_FAILED)*100, counts.average(MISSION_SKIPPED)*100, figured_out))
        if i > 20 and all([player.has_figured_out() for player in players]):
            print(i)
            break
        i += 1


if __name__ == "__main__":
    main()
