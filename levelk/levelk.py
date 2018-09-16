from world import load_world, ACTIONS, STAY
from collections import defaultdict

import cPickle as pickle

K_LEVEL = 0
VALUE_ITERATION = 100
GAMMA = 0.95

class Policy(object):
    def __init__(self, policy_file=None):
        self.policy = None
        if policy_file is None:
            return
        self.policy = pickle.load(open(policy_file))

    def execute(self, state):
        if self.policy is None:
            return STAY
        else:
            value, action = self.policy[state]
            return action


def k_0_policy(world):
    return STAY


def value_iteration(reachable_states, player, lower_policies):
    values = { state: (0, STAY) for state in reachable_states }
    for _ in range(VALUE_ITERATION):
        if _ % 10 == 0:
            print("= iter: {}".format(_))
        for state in values:
            if state.is_game_end():
                continue
            moves = [policy.execute(state) for policy in lower_policies]
            best_response = None
            best_value = -float('inf')
            for move in ACTIONS:
                moves[player] = move
                next_state, reward = state.next_state(moves)
                value = reward[player] + GAMMA*values[next_state][0]
                if value >= best_value:
                    best_value = value
                    best_response = move
            values[state] = (best_value, best_response)
    return values


def save_policy(player, k_level, values):
    print "=== saving Player {}, k = {} policy".format(player, k_level)
    with open('player_{}.k_{}.pickle'.format(player, k_level), 'w') as f:
        pickle.dump(values, f)


def load_policy(player, k_level):
    print "=== loading Player {}, k = {} policy".format(player, k_level)
    if k_level == 0:
        return Policy()
    return Policy('player_{}.k_{}.pickle'.format(player, k_level))


def train_main():
    starting_world = load_world('world.txt', 3)

    print("Calculating reachable states...")
    reachable_states = starting_world.all_reachable_states()

    for k_level in range(10):
        print("====== Creating policies for k={}".format(k_level))
        policies = [load_policy(player, k_level=k_level) for player in range(3)]
        for player in range(3):
            print("=== Iterating for player {}".format(player))
            save_policy(player, k_level + 1, value_iteration(reachable_states, player, policies))


def play_main():
    k_levels = [2, 0, 0]
    policies = [load_policy(player, k_level=k_level) for player, k_level in enumerate(k_levels)]
    world = load_world('world.txt', 3)

    while not world.is_game_end():
        print world
        world, _ = world.next_state([policy.execute(world) for policy in policies])

    print world



if __name__ == "__main__":
    play_main()
