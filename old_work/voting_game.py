import numpy as np
from collections import defaultdict, deque

# If they pass a winning dude, they get payoff 1
# If they pass a failing dude, they get payoff -1
# If the vote doesn't pass, they get payoff 0

NUM_PLAYERS = 10
NUM_BAD = 2
MISSION_SIZE = 3

MISSION_PASSED = 1
MISSION_FAILED = -1
MISSION_SKIPPED = 0

PLAYTHROUGH_DEPTH = 5
PLAYTHROUGH_ITERATIONS = 5

GOOD_GUY_K_LEVEL = 0
BAD_GUY_K_LEVEL = 0

def update(belief, is_bad):
    belief = np.clip(belief, 1e-20, 1.0 - 1e-15)
    undo = np.log(belief/(1.0 - belief))
    undo += is_bad - 0.5
    result = 1.0/(1 + np.exp(-undo))
    return result/np.sum(result)



class RunningAverageCounter(object):
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

class RunningAverage(object):
    def __init__(self, k=1000):
        self.k = k
        self.last_k = deque([])

    def add(self, value):
        self.last_k.append(value)
        if len(self.last_k) >= self.k:
            self.last_k.popleft()

    def average(self):
        return sum(self.last_k)/float(len(self.last_k))


def belief_sample(belief):
    return set(np.random.choice(range(NUM_PLAYERS), replace=False, size=NUM_BAD, p=belief))


class LevelKGood(object):
    def __init__(self, me, k, belief=None):
        self.me = me
        self.k = k
        if belief is not None:
            self.belief = belief
        else:
            self.belief = np.array([ 0.0 if i == me else 1/float(NUM_PLAYERS - 1) for i in range(NUM_PLAYERS)])

    def vote(self, mission):
        bad_guys_guess = belief_sample(self.belief)
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

    def guess_bad(self):
        result = np.argwhere(self.belief > 1e-5)
        return set(result.T[0])



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
        strategy = self._pick_strategy(depth=PLAYTHROUGH_DEPTH)
        return self._vote(mission, self.beliefs, strategy)

    def _vote(self, mission, beliefs, strategy):
        if strategy == 'normal':
            return len(mission & self.bad_guys) != 0
        elif strategy == 'anti':
            return len(mission & self.bad_guys) == 0
        elif strategy == 'belief_sample':
            belief = np.sum(self.beliefs, axis=0)
            belief /= np.sum(belief)
            bad_guys_guess = belief_sample(belief)
            return len(mission & bad_guys_guess) == 0
        elif strategy == 'random':
            return np.random.choice([True, False])

    def belief_update(self, mission, votes, result):
        if self.k != 1:
            return

        self.beliefs = self._belief_update(self.beliefs, mission, votes, result)


    def _belief_update(self, beliefs, mission, votes, result):
        if result == MISSION_SKIPPED:
            return beliefs
        if result == MISSION_PASSED:
            xor = 1
        else:
            xor = 0
        votes = np.array(votes) ^ xor
        return np.array([ update(beliefs[i], votes) for i in range(len(beliefs)) ])


    def _pick_strategy(self, depth):
        if self.k == 0:
            return 'normal'
        strat = max(['normal', 'anti', 'belief_sample', 'random'], key=lambda strat: self._evaluate_strategy(strat, depth))
        return strat


    def _evaluate_strategy(self, strategy, depth):
        payoff = 0
        for _ in range(PLAYTHROUGH_ITERATIONS):
            payoff += self._playthrough_with_strategy(strategy, depth)
        return payoff

    def _playthrough_with_strategy(self, strategy, depth):
        total_payoff = 0
        beliefs = self.beliefs
        for _ in range(depth):
            mission = set(np.random.choice(range(NUM_PLAYERS), replace=False, size=MISSION_SIZE))
            votes = []
            belief_index = 0
            for i in range(NUM_PLAYERS):
                if i in self.bad_guys:
                    votes.append(self._vote(mission, beliefs, strategy))
                else:
                    voter = LevelKGood(me=i, k=0, belief=self.beliefs[belief_index])
                    votes.append(voter.vote(mission))
                    belief_index += 1
            result = calc_payoff(mission, did_vote_pass(votes), self.bad_guys)
            total_payoff += result
            beliefs = self._belief_update(beliefs, mission, votes, result)
        return total_payoff


    def has_figured_out(self):
        return True

    def guess_bad(self):
        return self.bad_guys


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
    bad_guys = set(np.random.choice(range(NUM_PLAYERS), replace=False, size=NUM_BAD))
    players = [
        LevelKBad(me=i, k=BAD_GUY_K_LEVEL, bad_guys=bad_guys)
            if i in bad_guys else 
                LevelKGood(me=i, k=GOOD_GUY_K_LEVEL) 
                for i in range(NUM_PLAYERS)
    ]
    average_payoff = RunningAverage(k=100)
    counts = RunningAverageCounter(k=100)
    i = 1
    while True:
        payoff = iteration(players, bad_guys)
        counts.add(payoff)
        average_payoff.add(payoff)
        if i % 10 == 0:
            figured_out = sum([player.has_figured_out() for player in players])
            print(
                "{: 9d}: {: >8.4f}, PASS: {: >6.2f}%, FAIL: {: >6.2f}%, SKIP: {: >6.2f}%, FIGURED_OUT: {}".format(
                    i, 
                    average_payoff.average(),
                    counts.average(MISSION_PASSED)*100,
                    counts.average(MISSION_FAILED)*100,
                    counts.average(MISSION_SKIPPED)*100,
                    figured_out
                )
            )
        if i > 20 and all([player.has_figured_out() for player in players]):
            print(i)
            print("Actual bad: {}".format(bad_guys))
            for i, player in enumerate(players):
                print("Player {}: {}".format(i, player.guess_bad()))
            break
        i += 1


if __name__ == "__main__":
    main()
